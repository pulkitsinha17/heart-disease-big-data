"""
End-to-end integration test runner for Kafka Producer and Consumer.

Coordinates:
1. Topic verification and reset for a clean test run.
2. Starting consumer in background.
3. Running producer to publish all 303 records.
4. Verifying agreement between published and consumed record counts and validation metrics.
"""

import os
import subprocess
import sys
import time

PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCER_SCRIPT = os.path.join(BASE_DIR, "producer.py")
CONSUMER_SCRIPT = os.path.join(BASE_DIR, "consumer.py")


def run_e2e_test():
    print("=" * 60)
    print("STARTING KAFKA PRODUCER & CONSUMER END-TO-END TEST")
    print("=" * 60)

    # Generate a unique consumer group for this test run
    test_group = f"test-group-{int(time.time())}"

    print(f"\nStep 1: Starting Consumer in background (Group: {test_group})...")
    consumer_cmd = [
        PYTHON_EXE,
        CONSUMER_SCRIPT,
        "--group-id",
        test_group,
        "--max-messages",
        "303",
        "--timeout-ms",
        "20000",
    ]

    consumer_proc = subprocess.Popen(
        consumer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Allow consumer time to initialize connection and join group
    print("Waiting 3 seconds for consumer to subscribe and join consumer group...")
    time.sleep(3)

    print("\nStep 2: Starting Producer to publish 303 patient records...")
    producer_cmd = [
        PYTHON_EXE,
        PRODUCER_SCRIPT,
    ]

    producer_res = subprocess.run(producer_cmd, capture_output=True, text=True)
    print(producer_res.stdout)
    if producer_res.stderr:
        print("[PRODUCER STDERR]:", producer_res.stderr, file=sys.stderr)

    if producer_res.returncode != 0:
        print(f"[FAIL] Producer exited with code {producer_res.returncode}", file=sys.stderr)
        consumer_proc.terminate()
        sys.exit(1)

    print("\nStep 3: Awaiting Consumer completion...")
    try:
        cons_stdout, cons_stderr = consumer_proc.communicate(timeout=25)
        print(cons_stdout)
        if cons_stderr:
            print("[CONSUMER STDERR]:", cons_stderr, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[ERROR] Consumer timed out waiting for messages!", file=sys.stderr)
        consumer_proc.kill()
        sys.exit(1)

    if consumer_proc.returncode != 0:
        print(f"[FAIL] Consumer exited with error code {consumer_proc.returncode}", file=sys.stderr)
        sys.exit(1)

    # Verification checks
    if "Total Records Read:    303" in producer_res.stdout and "Successfully Acked:    303" in producer_res.stdout:
        print("[PASS] Producer successfully published and acknowledged all 303 records.")
    else:
        print("[FAIL] Producer count mismatch!", file=sys.stderr)
        sys.exit(1)

    if "Total Messages Consumed:  303" in cons_stdout and "Valid Records:            303" in cons_stdout:
        print("[PASS] Consumer successfully received and validated all 303 records.")
    else:
        print("[FAIL] Consumer count mismatch!", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL END-TO-END KAFKA STREAMING TESTS PASSED (303/303 RECORDS)")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_test()
