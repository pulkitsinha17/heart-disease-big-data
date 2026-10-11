"""
End-to-End Integration Test for Kafka to Hadoop HDFS Pipeline.

Validates:
1. Kafka broker connectivity (localhost:9092).
2. HDFS NameNode and DataNode connectivity (http://localhost:9870).
3. Resetting / verifying the Kafka topic 'heart-disease-records'.
4. Launching the HDFS integration consumer in background.
5. Publishing all 303 records from data/processed/heart_disease.csv via Kafka producer.
6. Awaiting consumer completion and verifying clean exit.
7. Reading back the persisted .jsonl file from HDFS.
8. Rigorous validation of records:
   - Exactly 303 records.
   - Unique record_id values (1 to 303).
   - Valid binary target (only 0 and 1).
   - Target distribution: exactly 164 zeros (absence) and 139 ones (presence).
   - Null value audit: exactly 4 nulls in 'ca' and 2 nulls in 'thal'.
   - Ground truth equality comparison against data/processed/heart_disease.csv.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List

import pandas as pd

# Add project paths to avoid site-packages collision
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDFS_DIR = os.path.join(PROJECT_ROOT, "hdfs")
KAFKA_DIR = os.path.join(PROJECT_ROOT, "kafka")

if HDFS_DIR not in sys.path:
    sys.path.insert(0, HDFS_DIR)
if KAFKA_DIR not in sys.path:
    sys.path.insert(0, KAFKA_DIR)

from hdfs_writer import HDFSWriter, DEFAULT_NAMENODE_URL, DEFAULT_HDFS_DIR
from config import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_TOPIC,
    EXPECTED_RECORD_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RECORD_ID_COLUMN,
)

PYTHON_EXE = sys.executable
PRODUCER_SCRIPT = os.path.join(PROJECT_ROOT, "kafka", "producer.py")
HDFS_CONSUMER_SCRIPT = os.path.join(PROJECT_ROOT, "kafka", "hdfs_consumer.py")
PROCESSED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease.csv")
TEST_HDFS_PATH = f"{DEFAULT_HDFS_DIR}/heart_disease_stream.jsonl"


def run_kafka_hdfs_test():
    print("=" * 70)
    print("PHASE 4: KAFKA -> HDFS END-TO-END INTEGRATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Check Kafka connectivity
    # ---------------------------------------------------------
    print("\n[Step 1/8] Verifying Kafka connectivity...")
    try:
        from kafka import KafkaProducer
        test_prod = KafkaProducer(
            bootstrap_servers=[DEFAULT_BOOTSTRAP_SERVERS],
            request_timeout_ms=5000,
        )
        test_prod.close()
        print(f"[PASS] Kafka broker reachable at {DEFAULT_BOOTSTRAP_SERVERS}.")
    except Exception as e:
        print(f"[FAIL] Cannot reach Kafka at {DEFAULT_BOOTSTRAP_SERVERS}: {e}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 2: Check HDFS connectivity & write permission
    # ---------------------------------------------------------
    print("\n[Step 2/8] Verifying HDFS connectivity and write permissions...")
    hdfs_client = HDFSWriter(namenode_url=DEFAULT_NAMENODE_URL)
    healthy, msg = hdfs_client.is_healthy()
    if not healthy:
        print(f"[FAIL] HDFS is not reachable: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] HDFS NameNode reachable at {DEFAULT_NAMENODE_URL}: {msg}")

    # Verify write capability by creating directory
    try:
        hdfs_client.mkdirs(DEFAULT_HDFS_DIR)
        print(f"[PASS] HDFS directory verified/created: {DEFAULT_HDFS_DIR}")
    except Exception as e:
        print(f"[FAIL] Could not create directory {DEFAULT_HDFS_DIR} in HDFS: {e}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 3: Prepare clean Kafka topic and consumer group
    # ---------------------------------------------------------
    test_run_id = int(time.time())
    consumer_group = f"test-hdfs-group-{test_run_id}"
    print(f"\n[Step 3/8] Using Kafka topic '{DEFAULT_TOPIC}' with fresh Consumer Group '{consumer_group}'...")

    # ---------------------------------------------------------
    # Step 4: Start HDFS Consumer in background
    # ---------------------------------------------------------
    print(f"\n[Step 4/8] Starting HDFS Consumer (target path: {TEST_HDFS_PATH})...")
    consumer_cmd = [
        PYTHON_EXE,
        HDFS_CONSUMER_SCRIPT,
        "--bootstrap-servers",
        DEFAULT_BOOTSTRAP_SERVERS,
        "--topic",
        DEFAULT_TOPIC,
        "--group-id",
        consumer_group,
        "--hdfs-path",
        TEST_HDFS_PATH,
        "--max-messages",
        str(EXPECTED_RECORD_COUNT),
        "--timeout-ms",
        "20000",
    ]

    consumer_proc = subprocess.Popen(
        consumer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for consumer group rebalance and partition assignment
    print("Waiting 4 seconds for consumer to initialize and subscribe...")
    time.sleep(4)

    # ---------------------------------------------------------
    # Step 5: Run Producer to stream all 303 records
    # ---------------------------------------------------------
    print(f"\n[Step 5/8] Running Producer to stream {EXPECTED_RECORD_COUNT} records from {PROCESSED_CSV_PATH}...")
    producer_cmd = [
        PYTHON_EXE,
        PRODUCER_SCRIPT,
        "--data-path",
        PROCESSED_CSV_PATH,
        "--topic",
        DEFAULT_TOPIC,
    ]

    prod_res = subprocess.run(producer_cmd, capture_output=True, text=True)
    print(prod_res.stdout)
    if prod_res.stderr:
        print("[PRODUCER STDERR]:", prod_res.stderr, file=sys.stderr)

    if prod_res.returncode != 0:
        print(f"[FAIL] Producer failed with exit code {prod_res.returncode}", file=sys.stderr)
        consumer_proc.terminate()
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 6: Await Consumer completion and HDFS persistence
    # ---------------------------------------------------------
    print("\n[Step 6/8] Awaiting HDFS Consumer completion...")
    try:
        cons_stdout, cons_stderr = consumer_proc.communicate(timeout=30)
        print(cons_stdout)
        if cons_stderr:
            print("[CONSUMER STDERR]:", cons_stderr, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[FAIL] Consumer timed out awaiting messages!", file=sys.stderr)
        consumer_proc.kill()
        sys.exit(1)

    if consumer_proc.returncode != 0:
        print(f"[FAIL] Consumer exited with error code {consumer_proc.returncode}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 7: Read back JSONL file from HDFS
    # ---------------------------------------------------------
    print(f"\n[Step 7/8] Reading back persisted file '{TEST_HDFS_PATH}' from HDFS...")
    try:
        stored_records = hdfs_client.read_jsonl(TEST_HDFS_PATH)
        print(f"[PASS] Successfully retrieved {len(stored_records)} records from HDFS via WebHDFS.")
    except Exception as e:
        print(f"[FAIL] Failed to read '{TEST_HDFS_PATH}' from HDFS: {e}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 8: Rigorous Schema, Distribution, and Ground Truth Validation
    # ---------------------------------------------------------
    print("\n[Step 8/8] Performing strict data validation...")

    # A. Record Count Verification
    actual_count = len(stored_records)
    if actual_count != EXPECTED_RECORD_COUNT:
        print(
            f"[FAIL] Record count mismatch: expected {EXPECTED_RECORD_COUNT}, got {actual_count}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[PASS] Record count check: exactly {actual_count} records.")

    # B. ID Uniqueness Check
    rec_ids = [r[RECORD_ID_COLUMN] for r in stored_records]
    unique_ids = set(rec_ids)
    if len(unique_ids) != EXPECTED_RECORD_COUNT:
        print(
            f"[FAIL] Duplicate IDs detected: {len(unique_ids)} unique out of {actual_count}",
            file=sys.stderr,
        )
        sys.exit(1)
    if sorted(list(unique_ids)) != list(range(1, EXPECTED_RECORD_COUNT + 1)):
        print(f"[FAIL] ID sequence mismatch: expected 1..{EXPECTED_RECORD_COUNT}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] ID uniqueness check: all {len(unique_ids)} IDs are unique (1 to {EXPECTED_RECORD_COUNT}).")

    # C. Target Binary Values and Distribution Check
    targets = [r[TARGET_COLUMN] for r in stored_records]
    invalid_targets = [t for t in targets if t not in (0, 1)]
    if invalid_targets:
        print(f"[FAIL] Invalid target values found: {invalid_targets}", file=sys.stderr)
        sys.exit(1)

    zeros_count = targets.count(0)
    ones_count = targets.count(1)
    if zeros_count != 164 or ones_count != 139:
        print(
            f"[FAIL] Target distribution mismatch: expected 164 zeroes and 139 ones, got {zeros_count} zeros and {ones_count} ones",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[PASS] Target distribution check: exactly 164 zeros (absence) and 139 ones (presence).")

    # D. Missing Values Audit (ca and thal)
    ca_nulls = sum(1 for r in stored_records if r.get("ca") is None)
    thal_nulls = sum(1 for r in stored_records if r.get("thal") is None)
    other_nulls = sum(
        1
        for r in stored_records
        for f in FEATURE_COLUMNS
        if f not in ("ca", "thal") and r.get(f) is None
    )

    if ca_nulls != 4 or thal_nulls != 2 or other_nulls != 0:
        print(
            f"[FAIL] Null value audit failed: ca nulls={ca_nulls} (expected 4), thal nulls={thal_nulls} (expected 2), other nulls={other_nulls} (expected 0)",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[PASS] Null value audit: exactly 4 nulls in 'ca', 2 nulls in 'thal', and 0 unexpected nulls.")

    # E. Exact Ground-Truth Match against CSV
    df_truth = pd.read_csv(PROCESSED_CSV_PATH)
    for idx, row in df_truth.iterrows():
        expected_id = idx + 1
        stored_r = stored_records[idx]
        assert stored_r[RECORD_ID_COLUMN] == expected_id, f"ID mismatch at index {idx}"
        assert stored_r[TARGET_COLUMN] == int(row[TARGET_COLUMN]), f"Target mismatch at index {idx}"

        for feat in FEATURE_COLUMNS:
            truth_val = row[feat]
            stored_val = stored_r[feat]
            if pd.isna(truth_val):
                assert stored_val is None, f"Expected null for {feat} at record {expected_id}, got {stored_val}"
            else:
                assert abs(stored_val - float(truth_val)) < 1e-5, f"Value mismatch for {feat} at record {expected_id}"

    print(f"[PASS] Ground-truth comparison: all {EXPECTED_RECORD_COUNT} records match data/processed/heart_disease.csv perfectly.")

    # ---------------------------------------------------------
    # Final Result
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULT: ALL PHASE 4 INTEGRATION TESTS PASSED (PASS)")
    print("=" * 70)
    print(f"HDFS File Stored:   {TEST_HDFS_PATH}")
    print(f"Total Records:      {actual_count}")
    print(f"Target Breakdown:   0: {zeros_count} (54.13%), 1: {ones_count} (45.87%)")
    print(f"Missing Values:     ca: {ca_nulls}, thal: {thal_nulls}")
    print(f"Storage Format:     JSON Lines (.jsonl)")
    print(f"Commit Strategy:    Manual commit after HDFS write confirmed")
    print("=" * 70)


if __name__ == "__main__":
    run_kafka_hdfs_test()
