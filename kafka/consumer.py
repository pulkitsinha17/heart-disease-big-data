"""
Kafka Consumer for ingesting and validating streaming heart disease records.

Subscribes to 'heart-disease-records', parses each JSON payload, verifies
structural completeness, checks record uniqueness and binary target validity,
and reports consumption statistics.
"""

import argparse
import json
import sys
import time
import warnings
from typing import Dict, Any, Set, List

warnings.filterwarnings("ignore", category=DeprecationWarning)

from kafka import KafkaConsumer
from kafka.errors import KafkaError, KafkaConnectionError

# Import shared configuration
try:
    from kafka.config import (
        DEFAULT_BOOTSTRAP_SERVERS,
        DEFAULT_TOPIC,
        DEFAULT_GROUP_ID,
        EXPECTED_RECORD_COUNT,
        RECORD_ID_COLUMN,
        TARGET_COLUMN,
        validate_record,
    )
except ImportError:
    from config import (
        DEFAULT_BOOTSTRAP_SERVERS,
        DEFAULT_TOPIC,
        DEFAULT_GROUP_ID,
        EXPECTED_RECORD_COUNT,
        RECORD_ID_COLUMN,
        TARGET_COLUMN,
        validate_record,
    )


def create_consumer(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    timeout_ms: int = 15000,
    from_beginning: bool = True,
) -> KafkaConsumer:
    """Create and return a KafkaConsumer instance."""
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            auto_offset_reset="earliest" if from_beginning else "latest",
            enable_auto_commit=True,
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        return consumer
    except (KafkaConnectionError, KafkaError) as e:
        print(f"\n[ERROR] Could not connect to Kafka broker at '{bootstrap_servers}': {e}", file=sys.stderr)
        print("Please verify that Kafka is running (e.g., 'docker compose ps').", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error initializing KafkaConsumer: {e}", file=sys.stderr)
        sys.exit(1)


def consume_and_validate(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    max_messages: int = EXPECTED_RECORD_COUNT,
    timeout_ms: int = 15000,
    from_beginning: bool = True,
    verbose: bool = False,
) -> int:
    """
    Consume messages from Kafka topic and perform rigorous validation.
    Returns the number of successfully validated messages.
    """
    print(f"Connecting to Kafka at {bootstrap_servers}...")
    print(f"Subscribing to topic:  '{topic}'")
    print(f"Consumer Group ID:     '{group_id}'")
    print(f"Target Record Count:   {max_messages}")
    print(f"Poll Timeout:          {timeout_ms / 1000:.1f}s")
    print(f"Auto Offset Reset:     {'earliest' if from_beginning else 'latest'}")
    print("\nWaiting for incoming messages (Ctrl+C to interrupt)...\n")

    consumer = create_consumer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        group_id=group_id,
        timeout_ms=timeout_ms,
        from_beginning=from_beginning,
    )

    consumed_count = 0
    valid_count = 0
    seen_ids: Set[Any] = set()
    duplicate_ids: List[Any] = []
    validation_errors: List[str] = []

    target_counts = {0: 0, 1: 0}
    null_counts = {"ca": 0, "thal": 0}

    start_time = time.time()

    try:
        for message in consumer:
            consumed_count += 1
            payload = message.value

            # Validate structural schema
            is_valid, err_msg = validate_record(payload)
            if not is_valid:
                validation_errors.append(f"Message {consumed_count}: {err_msg}")
                if verbose:
                    print(f"[INVALID] Record {consumed_count}: {err_msg}", file=sys.stderr)
                continue

            rec_id = payload[RECORD_ID_COLUMN]
            if rec_id in seen_ids:
                duplicate_ids.append(rec_id)
                validation_errors.append(f"Duplicate record_id detected: {rec_id}")
            else:
                seen_ids.add(rec_id)

            # Track target distribution
            target = payload[TARGET_COLUMN]
            target_counts[target] = target_counts.get(target, 0) + 1

            # Track null values in permitted features
            if payload.get("ca") is None:
                null_counts["ca"] += 1
            if payload.get("thal") is None:
                null_counts["thal"] += 1

            valid_count += 1

            if verbose or consumed_count % 50 == 0 or consumed_count == max_messages:
                print(
                    f"Consumed {consumed_count}/{max_messages} records "
                    f"(Offset: {message.offset}, Partition: {message.partition}, Last ID: {rec_id})"
                )

            # Exit once target message threshold is satisfied
            if max_messages > 0 and consumed_count >= max_messages:
                print(f"\nReached target limit of {max_messages} records. Stopping consumer.")
                break

    except KeyboardInterrupt:
        print("\nConsumer interrupted by user.")
    finally:
        consumer.close()
        print("KafkaConsumer closed cleanly.")

    elapsed = time.time() - start_time

    # Summary Report
    print("\n" + "=" * 50)
    print("        CONSUMER VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Topic Name:               {topic}")
    print(f"Total Messages Consumed:  {consumed_count}")
    print(f"Valid Records:            {valid_count}")
    print(f"Invalid Records:          {len(validation_errors)}")
    print(f"Unique Record IDs:        {len(seen_ids)}")
    print(f"Duplicate Record IDs:     {len(duplicate_ids)}")
    print(f"Target Distribution:      Class 0 = {target_counts.get(0, 0)}, Class 1 = {target_counts.get(1, 0)}")
    print(f"Null Values Observed:     ca = {null_counts['ca']}, thal = {null_counts['thal']}")
    print(f"Elapsed Time:             {elapsed:.2f} seconds")
    print("=" * 50)

    if duplicate_ids:
        print(f"\n[WARNING] Duplicate IDs found: {duplicate_ids[:10]}", file=sys.stderr)

    if validation_errors:
        print(f"\n[ERROR] Encountered {len(validation_errors)} validation errors:", file=sys.stderr)
        for err in validation_errors[:5]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    if consumed_count < max_messages:
        print(
            f"\n[WARNING] Consumed fewer messages ({consumed_count}) than requested ({max_messages}). "
            "Consumer poll timed out.",
            file=sys.stderr,
        )
        return valid_count

    print("\n[SUCCESS] All records successfully consumed and verified.")
    return valid_count


def main():
    parser = argparse.ArgumentParser(
        description="Consume and validate heart disease records from Apache Kafka."
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=DEFAULT_BOOTSTRAP_SERVERS,
        help=f"Kafka bootstrap servers (default: {DEFAULT_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help=f"Kafka topic to consume (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=f"{DEFAULT_GROUP_ID}-{int(time.time())}",
        help="Kafka consumer group ID (default: unique group per validation run)",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=EXPECTED_RECORD_COUNT,
        help=f"Maximum messages to consume before exiting (default: {EXPECTED_RECORD_COUNT})",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="Poll timeout in milliseconds when no messages arrive (default: 15000)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Start consuming from latest offset instead of earliest",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each consumed record detail",
    )

    args = parser.parse_args()
    consume_and_validate(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        max_messages=args.max_messages,
        timeout_ms=args.timeout_ms,
        from_beginning=not args.latest,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
