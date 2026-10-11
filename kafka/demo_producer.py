"""
Dedicated Demonstration Producer for Heart Disease Kafka-to-HDFS Pipeline.

Terminal 2 Entry Point:
- Reads 303 patient records from data/processed/heart_disease.csv.
- Connects to Kafka broker at localhost:9092.
- Streams patient records one-by-one with a gentle pacing delay (default: 0.01s).
- Flushes and verifies delivery acknowledgments.
"""

import argparse
import os
import sys

_KAFKA_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_KAFKA_DIR)

if _KAFKA_DIR not in sys.path:
    sys.path.insert(0, _KAFKA_DIR)

from config import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_TOPIC,
)
from producer import publish_records, DEFAULT_DATA_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Terminal 2: Live Demonstration Producer (CSV -> Kafka Stream)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to processed CSV (default: {DEFAULT_DATA_PATH})",
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
        default=0.01,
        help="Inter-message streaming pacing delay in seconds (default: 0.01s for realistic live feed)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" DEMONSTRATION MODE: STREAMING PATIENT RECORDS TO KAFKA")
    print("=" * 65)
    print(f" Source CSV:         {args.data_path}")
    print(f" Destination Topic:  {args.topic}")
    print(f" Kafka Broker:       {args.bootstrap_servers}")
    print(f" Streaming Pace:     {args.delay:.3f}s per record (~{args.delay * 303:.1f}s total)")
    print("=" * 65 + "\n")

    publish_records(
        data_path=args.data_path,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
