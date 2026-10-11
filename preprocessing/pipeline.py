"""
Preprocessing Pipeline Module for Heart Disease Machine Learning.

This module provides:
1. Direct HDFS data ingestion and validation for /heart-disease/raw/heart_disease_stream.jsonl.
2. Data validation (schema, types, row count, null counts, target distribution).
3. Leakage-safe ColumnTransformer definitions for scikit-learn Pipelines.
4. Stratified train-test splitting utilities.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Ensure local hdfs module is discoverable without colliding with site-packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
HDFS_DIR = os.path.join(PROJECT_ROOT, "hdfs")
if HDFS_DIR not in sys.path:
    sys.path.insert(0, HDFS_DIR)

try:
    from hdfs_writer import HDFSWriter, DEFAULT_NAMENODE_URL, DEFAULT_HDFS_FILE
except ImportError as e:
    raise ImportError(f"Failed to import HDFSWriter from {HDFS_DIR}: {e}")

# Feature definitions according to UCI Cleveland clinical specifications
NUMERIC_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: List[str] = ["cp", "restecg", "slope", "thal", "ca"]
BINARY_FEATURES: List[str] = ["sex", "fbs", "exang"]
CLINICAL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET_COLUMN: str = "target"
ID_COLUMN: str = "record_id"


def load_dataset_from_hdfs(
    hdfs_path: str = DEFAULT_HDFS_FILE,
    namenode_url: str = DEFAULT_NAMENODE_URL,
    expected_rows: Optional[int] = 303,
) -> pd.DataFrame:
    """
    Read patient records directly from Hadoop HDFS and parse into a pandas DataFrame.
    
    CRITICAL: Does not silently fall back to CSV if HDFS read fails.
    Raises RuntimeError if HDFS is unhealthy, or FileNotFoundError if file is missing.
    """
    writer = HDFSWriter(namenode_url=namenode_url)
    healthy, status_msg = writer.is_healthy()
    if not healthy:
        raise RuntimeError(
            f"HDFS cluster is unavailable or unhealthy: {status_msg}. "
            f"Please verify Docker containers (namenode, datanode) are running at {namenode_url}."
        )

    if not writer.exists(hdfs_path):
        raise FileNotFoundError(
            f"HDFS dataset file not found at path '{hdfs_path}'. "
            f"Ensure Phase 4 streaming pipeline has completed and persisted the dataset."
        )

    print(f"Reading patient records from HDFS: {hdfs_path} (NameNode: {namenode_url})...")
    records = writer.read_jsonl(hdfs_path)
    if not records:
        raise ValueError(f"HDFS dataset at '{hdfs_path}' is empty.")

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} records from HDFS.")

    # Validate against expected row count if specified
    if expected_rows is not None and len(df) != expected_rows:
        print(f"Notice: Expected {expected_rows} records, but loaded {len(df)}.")

    return df


def validate_and_prepare_features(
    df: pd.DataFrame,
    drop_record_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate DataFrame schema, drop metadata columns (record_id), and split into features (X) and target (y).
    """
    # Check target column
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Required target column '{TARGET_COLUMN}' missing from dataset.")

    # Verify binary target values
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column contains non-binary values: {unique_targets}")

    # Check required clinical features
    missing_features = [f for f in CLINICAL_FEATURES if f not in df.columns]
    if missing_features:
        raise KeyError(f"Dataset is missing required clinical features: {missing_features}")

    # Drop record_id if present
    features_df = df.copy()
    if drop_record_id and ID_COLUMN in features_df.columns:
        features_df = features_df.drop(columns=[ID_COLUMN])

    X = features_df[CLINICAL_FEATURES].copy()
    y = features_df[TARGET_COLUMN].astype(int).copy()

    # Log summary of missing values
    null_counts = X.isnull().sum()
    missing_cols = null_counts[null_counts > 0]
    if len(missing_cols) > 0:
        print(f"Features with missing values to be imputed:")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count} missing values ({count / len(X):.2%})")

    return X, y


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for leakage-safe data transformation.
    
    1. Numeric Features ('age', 'trestbps', 'chol', 'thalach', 'oldpeak'):
       - Median imputation for robustness against outliers.
       - Optional StandardScaler (fitted on training data only).
       
    2. Categorical Features ('cp', 'restecg', 'slope', 'thal', 'ca'):
       - Most-frequent imputation (handles missing 'thal' and 'ca' values).
       - OneHotEncoder with handle_unknown='ignore' (safely handles rare categories like restecg=1).
       
    3. Binary / Discrete Features ('sex', 'fbs', 'exang'):
       - Most-frequent imputation.
       - Passed through as numeric flags without scaling.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve feature names generated by fitted ColumnTransformer.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    return []


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split.
    Guarantees balanced class distribution across train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
