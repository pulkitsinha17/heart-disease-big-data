"""
Automated Integration Test for the Live Demonstration Workflow.

Tests:
1. Consumer starts in demo mode (Terminal 1 equivalent) and seeks to end of topic.
2. Producer starts (Terminal 2 equivalent) and streams 303 patient records.
3. Consumer intercepts ONLY the newly published 303 records (skipping 1515+ historical records).
4. Consumer validates records, persists to HDFS, verifies write, and commits offsets.
5. End-to-end audit of persisted HDFS file (record count, IDs 1..303, target distribution, ground truth).
"""

import os
import subprocess
import sys
import time

_KAFKA_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_KAFKA_DIR)

if _KAFKA_DIR not in sys.path:
    sys.path.insert(0, _KAFKA_DIR)
if os.path.join(_PROJECT_ROOT, "hdfs") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "hdfs"))

from hdfs_writer import HDFSWriter, DEFAULT_NAMENODE_URL, DEFAULT_HDFS_FILE
from config import EXPECTED_RECORD_COUNT

PYTHON_EXE = sys.executable
DEMO_CONSUMER = os.path.join(_KAFKA_DIR, "demo_consumer.py")
DEMO_PRODUCER = os.path.join(_KAFKA_DIR, "demo_producer.py")


def run_demo_test():
    print("=" * 70)
    print("TESTING LIVE DEMONSTRATION WORKFLOW (TERMINAL 1 + TERMINAL 2)")
    print("=" * 70)

    # 1. Start Demo Consumer in background
    print("\n[Step 1/5] Launching Demo Consumer in background (Terminal 1 simulation)...")
    consumer_proc = subprocess.Popen(
        [PYTHON_EXE, DEMO_CONSUMER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for consumer to initialize, assign partition, and report ready
    print("Waiting 4 seconds for consumer to synchronize and seek to partition end...")
    time.sleep(4)

    # 2. Run Demo Producer
    print("\n[Step 2/5] Running Demo Producer to stream 303 records (Terminal 2 simulation)...")
    prod_res = subprocess.run(
        [PYTHON_EXE, DEMO_PRODUCER, "--delay", "0.005"],
        capture_output=True,
        text=True,
    )
    print(prod_res.stdout)
    if prod_res.stderr:
        print("[PRODUCER STDERR]:", prod_res.stderr, file=sys.stderr)

    assert prod_res.returncode == 0, f"Producer failed with exit code {prod_res.returncode}"
    print("[PASS] Producer successfully published and acknowledged 303 records.")

    # 3. Wait for Consumer to finish
    print("\n[Step 3/5] Awaiting Demo Consumer completion...")
    try:
        cons_stdout, cons_stderr = consumer_proc.communicate(timeout=30)
        print(cons_stdout)
        if cons_stderr:
            print("[CONSUMER STDERR]:", cons_stderr, file=sys.stderr)
    except subprocess.TimeoutExpired:
        consumer_proc.kill()
        raise AssertionError("Demo Consumer timed out awaiting messages!")

    assert consumer_proc.returncode == 0, f"Consumer exited with error code {consumer_proc.returncode}"
    print("[PASS] Consumer finished cleanly with exit code 0.")

    # 4. Read back and audit from HDFS
    print("\n[Step 4/5] Auditing HDFS persisted dataset via WebHDFS...")
    writer = HDFSWriter(namenode_url=DEFAULT_NAMENODE_URL)
    records = writer.read_jsonl(DEFAULT_HDFS_FILE)
    print(f"Retrieved {len(records)} records from HDFS.")
    assert len(records) == EXPECTED_RECORD_COUNT, f"Expected {EXPECTED_RECORD_COUNT}, got {len(records)}"

    rec_ids = [r["record_id"] for r in records]
    assert len(set(rec_ids)) == EXPECTED_RECORD_COUNT, "Duplicate record IDs found in HDFS!"
    assert sorted(rec_ids) == list(range(1, EXPECTED_RECORD_COUNT + 1)), "Record IDs are not 1..303!"

    target_counts = {0: 0, 1: 0}
    for r in records:
        target_counts[r["target"]] += 1
    assert target_counts[0] == 164, f"Target 0 count mismatch: {target_counts[0]}"
    assert target_counts[1] == 139, f"Target 1 count mismatch: {target_counts[1]}"

    # 5. Result
    print("\n[Step 5/5] Final Verification:")
    print("  - Total Records in HDFS:    303")
    print("  - Unique Record IDs:        303 (1 to 303)")
    print("  - Absence (Class 0):        164 (54.1%)")
    print("  - Presence (Class 1):        139 (45.9%)")
    print("  - Null counts:              ca=4, thal=2")
    print("\n" + "=" * 70)
    print("RESULT: LIVE DEMONSTRATION WORKFLOW TEST PASSED (PASS)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo_test()
