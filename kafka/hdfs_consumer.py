"""
Kafka-to-HDFS Integration Consumer for Heart Disease Big Data Pipeline.

Subscribes to 'heart-disease-records', validates each incoming clinical record,
persists records to Hadoop HDFS as JSON Lines (.jsonl), and manually commits
Kafka offsets ONLY after successful HDFS persistence (at-least-once / verified persistence).
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, Any, List, Set

warnings.filterwarnings("ignore", category=DeprecationWarning)

from kafka import KafkaConsumer
from kafka.errors import KafkaError, KafkaConnectionError

# Import local modules robustly against site-package namespace collisions
_KAFKA_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_KAFKA_DIR)

if _KAFKA_DIR not in sys.path:
    sys.path.insert(0, _KAFKA_DIR)
if os.path.join(_PROJECT_ROOT, "hdfs") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "hdfs"))

try:
    from config import (
        DEFAULT_BOOTSTRAP_SERVERS,
        DEFAULT_TOPIC,
        EXPECTED_RECORD_COUNT,
        RECORD_ID_COLUMN,
        TARGET_COLUMN,
        validate_record,
    )
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("kconfig", os.path.join(_KAFKA_DIR, "config.py"))
    _kcfg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_kcfg)
    DEFAULT_BOOTSTRAP_SERVERS = _kcfg.DEFAULT_BOOTSTRAP_SERVERS
    DEFAULT_TOPIC = _kcfg.DEFAULT_TOPIC
    EXPECTED_RECORD_COUNT = _kcfg.EXPECTED_RECORD_COUNT
    RECORD_ID_COLUMN = _kcfg.RECORD_ID_COLUMN
    TARGET_COLUMN = _kcfg.TARGET_COLUMN
    validate_record = _kcfg.validate_record

from hdfs_writer import HDFSWriter, DEFAULT_NAMENODE_URL, DEFAULT_HDFS_FILE


def run_hdfs_consumer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    topic: str = DEFAULT_TOPIC,
    group_id: str = f"heart-disease-hdfs-group-{int(time.time())}",
    hdfs_url: str = DEFAULT_NAMENODE_URL,
    hdfs_path: str = DEFAULT_HDFS_FILE,
    max_messages: int = EXPECTED_RECORD_COUNT,
    timeout_ms: int = 15000,
    overwrite: bool = True,
    from_beginning: bool = True,
    verbose: bool = False,
) -> int:
    """
    Ingest records from Kafka, validate schema, persist to HDFS, and commit offsets.
    Returns the count of successfully persisted records.
    """
    print("=" * 60)
    print("STARTING KAFKA-TO-HDFS STREAMING CONSUMER")
    print("=" * 60)
    print(f"Kafka Broker:       {bootstrap_servers}")
    print(f"Topic:              {topic}")
    print(f"Consumer Group:     {group_id}")
    print(f"HDFS NameNode URL:  {hdfs_url}")
    print(f"HDFS Target Path:   {hdfs_path}")
    print(f"Max Messages:       {max_messages} (0 for continuous)")
    print(f"Poll Timeout:       {timeout_ms / 1000:.1f}s")
    print(f"Offset Reset:       {'earliest' if from_beginning else 'latest'}")
    print(f"Commit Strategy:    Manual commit AFTER HDFS write verification")
    print("=" * 60)

    # 1. Initialize and verify HDFS Writer
    hdfs_writer = HDFSWriter(namenode_url=hdfs_url)
    healthy, health_msg = hdfs_writer.is_healthy()
    if not healthy:
        print(f"\n[ERROR] HDFS health check failed: {health_msg}", file=sys.stderr)
        print("Please verify that NameNode and DataNode containers are running.", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] HDFS is healthy: {health_msg}")

    # 2. Connect to Kafka Consumer with manual offset management
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            auto_offset_reset="earliest" if from_beginning else "latest",
            enable_auto_commit=False,  # MANUAL COMMIT STRATEGY: only commit after HDFS write
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
    except (KafkaConnectionError, KafkaError) as e:
        print(f"\n[ERROR] Could not connect to Kafka broker at '{bootstrap_servers}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error initializing KafkaConsumer: {e}", file=sys.stderr)
        sys.exit(1)

    consumed_records: List[Dict[str, Any]] = []
    seen_ids: Set[Any] = set()
    duplicate_ids: List[Any] = []
    validation_errors: List[str] = []

    target_counts = {0: 0, 1: 0}
    null_counts = {"ca": 0, "thal": 0}

    start_time = time.time()
    print("\nListening for streaming records from Kafka...\n")

    try:
        for message in consumer:
            payload = message.value

            # Validate record structure
            is_valid, err_msg = validate_record(payload)
            if not is_valid:
                validation_errors.append(f"Offset {message.offset}: {err_msg}")
                if verbose:
                    print(f"[INVALID] Offset {message.offset}: {err_msg}", file=sys.stderr)
                continue

            rec_id = payload[RECORD_ID_COLUMN]
            if rec_id in seen_ids:
                duplicate_ids.append(rec_id)
                validation_errors.append(f"Duplicate record_id: {rec_id}")
            else:
                seen_ids.add(rec_id)

            # Track distributions
            target = payload[TARGET_COLUMN]
            target_counts[target] = target_counts.get(target, 0) + 1

            if payload.get("ca") is None:
                null_counts["ca"] += 1
            if payload.get("thal") is None:
                null_counts["thal"] += 1

            consumed_records.append(payload)
            count_so_far = len(consumed_records)

            if verbose or count_so_far % 50 == 0 or count_so_far == max_messages:
                print(
                    f"Received & Validated {count_so_far}/{max_messages} records "
                    f"(Offset: {message.offset}, ID: {rec_id})"
                )

            if max_messages > 0 and count_so_far >= max_messages:
                print(f"\nTarget count of {max_messages} records reached. Finishing ingestion batch.")
                break

    except KeyboardInterrupt:
        print("\nConsumer interrupted by user.")

    total_validated = len(consumed_records)

    # 3. Persist batch to HDFS and commit Kafka offsets
    if total_validated > 0:
        print(f"\nPersisting {total_validated} records to HDFS at: {hdfs_path}...")
        try:
            written_count = hdfs_writer.write_jsonl(
                hdfs_path=hdfs_path,
                records=consumed_records,
                overwrite=overwrite,
            )
            print(f"[OK] Successfully wrote {written_count} JSONL lines to HDFS.")

            # CRITICAL COMMIT STEP: Only commit Kafka offsets once HDFS write succeeds!
            print("Committing Kafka consumer offsets (HDFS write confirmed)...")
            consumer.commit()
            print("[OK] Kafka offsets successfully committed.")

        except Exception as e:
            print(f"\n[CRITICAL ERROR] HDFS write failed: {e}", file=sys.stderr)
            print("Kafka offsets will NOT be committed to prevent data loss.", file=sys.stderr)
            consumer.close()
            sys.exit(1)
    else:
        print("\n[WARNING] No records were received from Kafka.")

    consumer.close()
    print("Kafka consumer closed cleanly.")
    elapsed = time.time() - start_time

    # 4. Summary & Verification
    print("\n" + "=" * 60)
    print("        KAFKA-TO-HDFS INGESTION SUMMARY")
    print("=" * 60)
    print(f"HDFS Target Location:     {hdfs_path}")
    print(f"Records Consumed:         {total_validated}")
    print(f"Records Stored in HDFS:   {total_validated}")
    print(f"Validation Errors:        {len(validation_errors)}")
    print(f"Unique Record IDs:        {len(seen_ids)}")
    print(f"Duplicate Record IDs:     {len(duplicate_ids)}")
    print(f"Target Distribution:      Class 0 = {target_counts.get(0, 0)}, Class 1 = {target_counts.get(1, 0)}")
    print(f"Null Values Stored:       ca = {null_counts['ca']}, thal = {null_counts['thal']}")
    print(f"Elapsed Time:             {elapsed:.2f} seconds")
    print("=" * 60)

    if validation_errors:
        print(f"\n[ERROR] Encountered {len(validation_errors)} validation errors during run:", file=sys.stderr)
        for err in validation_errors[:5]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    if total_validated < max_messages:
        print(
            f"\n[WARNING] Consumed {total_validated} records, less than target {max_messages}.",
            file=sys.stderr,
        )

    return total_validated


def main():
    parser = argparse.ArgumentParser(
        description="Stream patient records from Kafka into Hadoop HDFS (.jsonl)."
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
        help=f"Kafka topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=f"heart-disease-hdfs-group-{int(time.time())}",
        help="Kafka consumer group ID (default: unique group per run)",
    )
    parser.add_argument(
        "--hdfs-url",
        type=str,
        default=DEFAULT_NAMENODE_URL,
        help=f"HDFS NameNode URL (default: {DEFAULT_NAMENODE_URL})",
    )
    parser.add_argument(
        "--hdfs-path",
        type=str,
        default=DEFAULT_HDFS_FILE,
        help=f"Destination path in HDFS (default: {DEFAULT_HDFS_FILE})",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=EXPECTED_RECORD_COUNT,
        help=f"Target records to ingest before exiting (default: {EXPECTED_RECORD_COUNT})",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="Poll timeout in milliseconds (default: 15000)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing HDFS file instead of overwriting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose record-level logging",
    )

    args = parser.parse_args()
    run_hdfs_consumer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        hdfs_url=args.hdfs_url,
        hdfs_path=args.hdfs_path,
        max_messages=args.max_messages,
        timeout_ms=args.timeout_ms,
        overwrite=not args.append,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
