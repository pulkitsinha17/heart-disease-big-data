"""
Kafka configuration and schema validation utilities for Heart Disease data streaming.
"""

import os
from typing import Dict, Any, Tuple, List

DEFAULT_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_TOPIC: str = os.getenv("KAFKA_TOPIC", "heart-disease-records")
DEFAULT_DEMO_TOPIC: str = os.getenv("KAFKA_DEMO_TOPIC", "heart-disease-records")
DEFAULT_GROUP_ID: str = os.getenv("KAFKA_GROUP_ID", "heart-disease-consumer-group")
EXPECTED_RECORD_COUNT: int = 303

FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN: str = "target"
RECORD_ID_COLUMN: str = "record_id"

REQUIRED_FIELDS: List[str] = [RECORD_ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# Columns where null (missing) values are clinically permitted in Cleveland dataset
NULLABLE_COLUMNS: set = {"ca", "thal"}


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a consumed heart disease record against expected schema and ranges.
    Returns (is_valid, error_message).
    """
    if not isinstance(record, dict):
        return False, "Record payload is not a valid JSON object/dictionary"

    # Check for presence of all required fields
    missing_fields = [f for f in REQUIRED_FIELDS if f not in record]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Validate record_id
    rec_id = record[RECORD_ID_COLUMN]
    if not isinstance(rec_id, (int, str)):
        return False, f"Invalid record_id type: expected int or str, got {type(rec_id).__name__}"

    # Validate target (must be 0 or 1)
    target = record[TARGET_COLUMN]
    if target not in (0, 1):
        return False, f"Invalid target value: expected 0 or 1, got {target}"

    # Validate nullable fields: ca and thal can be None / null
    for field in FEATURE_COLUMNS:
        val = record[field]
        if val is None:
            if field not in NULLABLE_COLUMNS:
                return False, f"Field '{field}' cannot be null"
        else:
            if not isinstance(val, (int, float)):
                return False, f"Field '{field}' must be numeric or null, got {type(val).__name__}"

    return True, ""
