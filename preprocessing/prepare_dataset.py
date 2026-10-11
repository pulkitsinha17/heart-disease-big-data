"""
Dataset preparation script for the UCI Heart Disease (Cleveland) dataset.

This script:
1. Downloads the official processed Cleveland dataset from the UCI Machine Learning Repository
   if it does not already exist in data/raw/.
2. Parses the raw data using official Cleveland attribute names and handles '?' as missing values.
3. Keeps all 13 input clinical features.
4. Binarizes the diagnosis column ('num') into 'target' (0 = absence, 1 = presence).
5. Preserves missing values in 'ca' and 'thal' without row deletion or ML imputation.
6. Saves the cleaned dataset deterministically to data/processed/heart_disease.csv.
"""

import os
import sys
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np

UCI_PROCESSED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

RAW_FILE_PATH = os.path.join(RAW_DATA_DIR, "processed.cleveland.data")
NAMES_FILE_PATH = os.path.join(RAW_DATA_DIR, "heart-disease.names")
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "heart_disease.csv")

COLUMN_NAMES = [
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
    "num",
]


def download_raw_data():
    """Download the raw Cleveland dataset and names documentation from UCI."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. Download processed.cleveland.data if not present
    if not os.path.exists(RAW_FILE_PATH):
        print(f"Downloading raw data from {UCI_PROCESSED_URL}...")
        try:
            req = urllib.request.Request(UCI_PROCESSED_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
            with open(RAW_FILE_PATH, "wb") as f:
                f.write(raw_bytes)
            print(f"Saved raw data to {os.path.abspath(RAW_FILE_PATH)} ({len(raw_bytes)} bytes)")
        except Exception as e:
            print(f"Direct download failed ({e}), attempting fallback via official UCI zip...")
            req_zip = urllib.request.Request(UCI_ZIP_URL, headers=headers)
            with urllib.request.urlopen(req_zip, timeout=30) as resp_zip:
                z = zipfile.ZipFile(io.BytesIO(resp_zip.read()))
                data_bytes = z.read("processed.cleveland.data")
                with open(RAW_FILE_PATH, "wb") as f:
                    f.write(data_bytes)
                if not os.path.exists(NAMES_FILE_PATH):
                    with open(NAMES_FILE_PATH, "wb") as f_names:
                        f_names.write(z.read("heart-disease.names"))
            print("Successfully extracted dataset from official UCI archive zip.")

    # 2. Download heart-disease.names metadata if not present
    if not os.path.exists(NAMES_FILE_PATH):
        try:
            req_zip = urllib.request.Request(UCI_ZIP_URL, headers=headers)
            with urllib.request.urlopen(req_zip, timeout=30) as resp_zip:
                z = zipfile.ZipFile(io.BytesIO(resp_zip.read()))
                with open(NAMES_FILE_PATH, "wb") as f_names:
                    f_names.write(z.read("heart-disease.names"))
            print(f"Saved attribute documentation to {os.path.abspath(NAMES_FILE_PATH)}")
        except Exception as e:
            print(f"Notice: Could not fetch heart-disease.names archive: {e}")


def load_and_prepare_data():
    """Load raw dataset, format columns, convert target to binary, and save processed CSV."""
    download_raw_data()

    if not os.path.exists(RAW_FILE_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_FILE_PATH}")

    print(f"\nLoading raw data from: {os.path.abspath(RAW_FILE_PATH)}")
    df_raw = pd.read_csv(
        RAW_FILE_PATH,
        names=COLUMN_NAMES,
        na_values="?",
        sep=",",
        skipinitialspace=True,
    )

    print(f"Raw dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Validate row count
    if df_raw.shape != (303, 14):
        print(f"WARNING: Expected (303, 14), observed {df_raw.shape}")

    # Inspect missing values in raw dataset
    missing_counts = df_raw.isna().sum()
    print("\nMissing values per column in raw data:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  - {col}: {count} missing ({count / len(df_raw):.2%})")

    # Copy to create processed dataset
    df_processed = df_raw.copy()

    # Binarize target: 0 -> 0 (absence), > 0 -> 1 (presence)
    df_processed["target"] = (df_processed["num"] > 0).astype(int)

    # Drop original 'num' column
    df_processed = df_processed.drop(columns=["num"])

    # Ensure column ordering: 13 features followed by target
    feature_cols = [c for c in COLUMN_NAMES if c != "num"]
    final_cols = feature_cols + ["target"]
    df_processed = df_processed[final_cols]

    # Save to data/processed/heart_disease.csv deterministically
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_processed.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nProcessed dataset saved to: {os.path.abspath(OUTPUT_CSV_PATH)}")

    # Summary report
    print("\n--- Processed Dataset Summary ---")
    print(f"Total Rows:    {len(df_processed)}")
    print(f"Total Columns: {len(df_processed.columns)} ({len(feature_cols)} features + 1 target)")
    print(f"Target Distribution:\n{df_processed['target'].value_counts().sort_index()}")
    pct = df_processed['target'].value_counts(normalize=True).sort_index() * 100
    print(f"  Target 0 (Absence):  {df_processed['target'].value_counts()[0]} ({pct[0]:.2f}%)")
    print(f"  Target 1 (Presence): {df_processed['target'].value_counts()[1]} ({pct[1]:.2f}%)")
    print("\nMissing Values in Processed Dataset:")
    proc_missing = df_processed.isna().sum()
    for col in df_processed.columns:
        if proc_missing[col] > 0:
            print(f"  - {col}: {proc_missing[col]} missing ({proc_missing[col] / len(df_processed):.2%})")

    return df_processed


if __name__ == "__main__":
    load_and_prepare_data()
