"""
Kafka Producer for streaming patient clinical records.

Reads the processed UCI Cleveland dataset (data/processed/heart_disease.csv)
and publishes each patient record as a serialized JSON message to Kafka.
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, Any, List

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaConnectionError

# Import shared configuration
try:
    from kafka.config import (
        DEFAULT_BOOTSTRAP_SERVERS,
        DEFAULT_TOPIC,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        RECORD_ID_COLUMN,
    )
except ImportError:
    from config import (
        DEFAULT_BOOTSTRAP_SERVERS,
        DEFAULT_TOPIC,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        RECORD_ID_COLUMN,
    )

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "heart_disease.csv"
)


def serialize_record(row: pd.Series, record_id: int) -> Dict[str, Any]:
    """
    Convert a Pandas row into a clean dictionary suitable for strict JSON serialization.
    Missing/NaN values are explicitly mapped to Python None (JSON null).
    """
    record: Dict[str, Any] = {RECORD_ID_COLUMN: int(record_id)}

    for col in FEATURE_COLUMNS:
        val = row[col]
        if pd.isna(val) or val is None:
            record[col] = None
        elif isinstance(val, (np.integer, int)):
            record[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            # Retain integer-like floats as integers if clean (e.g. sex=1.0 -> 1.0 or float)
            record[col] = float(val)
        else:
            record[col] = val

    # Ensure target is an integer (0 or 1)
    target_val = row[TARGET_COLUMN]
    record[TARGET_COLUMN] = int(target_val)

    return record


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    """Create and return a KafkaProducer with JSON serialization and delivery guarantees."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v, allow_nan=False).encode("utf-8"),
            acks="all",  # Wait for full acknowledgment from broker
            retries=3,
            max_block_ms=15000,
            request_timeout_ms=10000,
        )
        return producer
    except (KafkaConnectionError, KafkaError) as e:
        print(f"\n[ERROR] Could not connect to Kafka broker at '{bootstrap_servers}': {e}", file=sys.stderr)
        print("Please verify that Kafka is running (e.g., 'docker compose ps').", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error connecting to Kafka: {e}", file=sys.stderr)
        sys.exit(1)


def publish_records(
    data_path: str,
    bootstrap_servers: str,
    topic: str,
    delay: float = 0.0,
) -> int:
    """
    Read dataset from CSV and publish each patient record to Kafka.
    Returns the total number of successfully acknowledged records.
    """
    resolved_path = os.path.abspath(data_path)
    if not os.path.exists(resolved_path):
        print(f"[ERROR] Processed dataset not found at: {resolved_path}", file=sys.stderr)
        print("Please run 'python preprocessing/prepare_dataset.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path)
    total_records = len(df)
    print(f"Loaded {total_records} records with columns: {list(df.columns)}")

    print(f"Connecting to Kafka broker at: {bootstrap_servers}")
    producer = create_producer(bootstrap_servers)

    print(f"Publishing {total_records} patient records to topic '{topic}'...")

    successful_acks = 0
    delivery_errors: List[str] = []

    def on_send_success(record_metadata):
        nonlocal successful_acks
        successful_acks += 1

    def on_send_error(excp):
        nonlocal delivery_errors
        delivery_errors.append(str(excp))
        print(f"[ERROR] Failed to deliver record: {excp}", file=sys.stderr)

    start_time = time.time()

    for idx, row in df.iterrows():
        record_id = idx + 1  # 1-indexed patient identifier (1 to 303)
        payload = serialize_record(row, record_id)

        try:
            future = producer.send(
                topic,
                key=str(record_id).encode("utf-8"),
                value=payload,
            )
            future.add_callback(on_send_success)
            future.add_errback(on_send_error)
        except Exception as e:
            delivery_errors.append(f"Record {record_id}: {e}")
            print(f"[ERROR] Producer send error on record {record_id}: {e}", file=sys.stderr)

        if delay > 0:
            time.sleep(delay)

    # Flush all outstanding messages and wait for broker acknowledgment
    print("Flushing messages to Kafka broker and awaiting acknowledgments...")
    producer.flush(timeout=30)
    producer.close()

    elapsed = time.time() - start_time

    # Report results
    print("\n--- Producer Publishing Summary ---")
    print(f"Topic:                 {topic}")
    print(f"Total Records Read:    {total_records}")
    print(f"Successfully Acked:    {successful_acks}")
    print(f"Failed Messages:       {len(delivery_errors)}")
    print(f"Elapsed Time:          {elapsed:.2f} seconds")

    if delivery_errors:
        print(f"\n[ERROR] Encountered {len(delivery_errors)} delivery failures:", file=sys.stderr)
        for err in delivery_errors[:5]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    if successful_acks != total_records:
        print(
            f"\n[ERROR] Mismatch between records read ({total_records}) and acknowledged ({successful_acks}).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[SUCCESS] All records successfully published and acknowledged by Kafka broker.")
    return successful_acks


def main():
    parser = argparse.ArgumentParser(
        description="Publish heart disease records from CSV to Apache Kafka."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to processed CSV file (default: {DEFAULT_DATA_PATH})",
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
        help=f"Kafka destination topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay between messages in seconds for simulating real-time streaming (default: 0.0)",
    )

    args = parser.parse_args()
    publish_records(
        data_path=args.data_path,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
