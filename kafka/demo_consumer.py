"""
Dedicated Demonstration Consumer for Heart Disease Kafka-to-HDFS Pipeline.

Terminal 1 Entry Point:
- Connects to Kafka and determines current log watermark.
- Seeks to the end of all assigned partitions to skip historical backlog.
- Displays clear waiting/listening banner.
- Awaits live streaming of 303 records from Terminal 2 producer.
- Validates structural schema, sequence, class distribution, and null audit.
- Persists exactly 303 records to HDFS (/heart-disease/raw/heart_disease_stream.jsonl).
- Verifies HDFS persistence via WebHDFS read-back before committing Kafka offsets.
"""

import argparse
import os
import sys
import time

_KAFKA_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_KAFKA_DIR)

if _KAFKA_DIR not in sys.path:
    sys.path.insert(0, _KAFKA_DIR)
if os.path.join(_PROJECT_ROOT, "hdfs") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "hdfs"))

from config import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_TOPIC,
    EXPECTED_RECORD_COUNT,
)
from hdfs_consumer import run_hdfs_consumer
from hdfs_writer import DEFAULT_NAMENODE_URL, DEFAULT_HDFS_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Terminal 1: Live Demonstration Consumer (Kafka -> HDFS)"
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
        default=None,
        help="Custom consumer group ID (default: fresh unique group per demo run)",
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
        help=f"HDFS destination file (default: {DEFAULT_HDFS_FILE})",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=EXPECTED_RECORD_COUNT,
        help=f"Target records to ingest (default: {EXPECTED_RECORD_COUNT})",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=90000,
        help="Poll timeout in milliseconds awaiting producer (default: 90000 ms / 90s)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing HDFS file instead of overwriting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose record-by-record logging",
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
        demo_mode=True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
