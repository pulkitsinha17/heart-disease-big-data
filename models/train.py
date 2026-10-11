"""
Model Training and Orchestration Script for Heart Disease Prediction.

Executes Phase 5 end-to-end:
1. Loads patient records directly from Hadoop HDFS (/heart-disease/raw/heart_disease_stream.jsonl).
2. Validates schema, feature columns, and binary target.
3. Performs stratified 80:20 train-test split (seed=42).
4. Fits leakage-safe preprocessing pipelines and classifiers:
   - Logistic Regression (with StandardScaler and L2 regularization)
   - Random Forest Classifier (with tree-based ensemble)
5. Executes 5-fold Stratified Cross-Validation on the training set.
6. Evaluates final performance on the held-out test set.
7. Saves serialized model pipelines to models/*.joblib.
8. Exports metrics, comparison tables, and diagnostic plots to results/*.
"""

import os
import sys
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate

# Resolve project paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, "preprocessing")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for d in [PROJECT_ROOT, PREPROCESSING_DIR, MODELS_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pipeline import (
    load_dataset_from_hdfs,
    validate_and_prepare_features,
    build_preprocessor,
    split_data,
    get_feature_names_after_preprocessing,
    CLINICAL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)
from evaluate import (
    compute_classification_metrics,
    plot_roc_curve_comparison,
    plot_confusion_matrices,
    plot_feature_importance_comparison,
)


def run_training_pipeline() -> Dict[str, Any]:
    """Execute complete ML pipeline from HDFS ingestion to evaluation and persistence."""
    print("=" * 70)
    print("PHASE 5: MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Ingest dataset directly from HDFS
    print("\n[Step 1] Ingesting dataset from Hadoop HDFS...")
    df = load_dataset_from_hdfs()
    print(f"Loaded DataFrame shape: {df.shape}")

    # 2. Schema validation & feature preparation
    print("\n[Step 2] Validating clinical schema and separating target...")
    X, y = validate_and_prepare_features(df)
    print(f"Total features: {X.shape[1]}, Total records: {len(X)}")
    print(f"Target distribution (Overall):")
    print(f"  Class 0 (Absence):  {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"  Class 1 (Presence): {(y == 1).sum()} ({(y == 1).mean():.1%})")

    # 3. Stratified Train-Test Split (80:20, seed=42)
    print("\n[Step 3] Performing stratified 80:20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
    print(f"  Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
    print(f"Test set:     {len(X_test)} samples ({len(X_test)/len(X):.1%})")
    print(f"  Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")

    # 4. Define Leakage-Safe Pipelines
    print("\n[Step 4] Defining scikit-learn Pipelines with ColumnTransformer...")
    # Logistic Regression: Scaled numeric features + One-Hot encoded categoricals
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])

    # Random Forest: Unscaled numeric features + One-Hot encoded categoricals
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )),
    ])

    # 5. Repeated Stratified 5-Fold Cross-Validation on Training Split Only
    print("\n[Step 5] Evaluating Repeated 5-Fold Stratified Cross-Validation on Training Set...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]

    cv_lr = cross_validate(lr_pipeline, X_train, y_train, cv=rskf, scoring=scoring)
    cv_rf = cross_validate(rf_pipeline, X_train, y_train, cv=rskf, scoring=scoring)

    print(f"Logistic Regression Repeated 5-Fold CV (5 repeats = 25 folds):")
    print(f"  ROC-AUC:  {cv_lr['test_roc_auc'].mean():.4f} (+/- {cv_lr['test_roc_auc'].std():.4f})")
    print(f"  Accuracy: {cv_lr['test_accuracy'].mean():.4f} (+/- {cv_lr['test_accuracy'].std():.4f})")
    print(f"  Precision:{cv_lr['test_precision'].mean():.4f} (+/- {cv_lr['test_precision'].std():.4f})")
    print(f"  Recall:   {cv_lr['test_recall'].mean():.4f} (+/- {cv_lr['test_recall'].std():.4f})")
    print(f"  F1 Score: {cv_lr['test_f1'].mean():.4f} (+/- {cv_lr['test_f1'].std():.4f})")

    print(f"Random Forest Classifier Repeated 5-Fold CV (5 repeats = 25 folds):")
    print(f"  ROC-AUC:  {cv_rf['test_roc_auc'].mean():.4f} (+/- {cv_rf['test_roc_auc'].std():.4f})")
    print(f"  Accuracy: {cv_rf['test_accuracy'].mean():.4f} (+/- {cv_rf['test_accuracy'].std():.4f})")
    print(f"  Precision:{cv_rf['test_precision'].mean():.4f} (+/- {cv_rf['test_precision'].std():.4f})")
    print(f"  Recall:   {cv_rf['test_recall'].mean():.4f} (+/- {cv_rf['test_recall'].std():.4f})")
    print(f"  F1 Score: {cv_rf['test_f1'].mean():.4f} (+/- {cv_rf['test_f1'].std():.4f})")

    # 6. Fit Pipelines on Full Training Set and Evaluate on Held-Out Test Set
    print("\n[Step 6] Fitting final pipelines on training set and evaluating on held-out test set...")
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]

    rf_pred = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    lr_metrics = compute_classification_metrics(y_test.values, lr_pred, lr_proba, "Logistic Regression")
    rf_metrics = compute_classification_metrics(y_test.values, rf_pred, rf_proba, "Random Forest")

    # Display Test Set Results
    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION RESULTS (N = 61)")
    print("=" * 70)
    for m in [lr_metrics, rf_metrics]:
        print(f"\nModel: {m['model_name']}")
        print(f"  ROC-AUC:             {m['roc_auc']:.4f} (Target > 0.85: {'MET [PASS]' if m['roc_auc'] > 0.85 else 'UNMET [FAIL]'})")
        print(f"  Accuracy:            {m['accuracy']:.4f} ({m['accuracy']:.2%})")
        print(f"  Precision:           {m['precision']:.4f}")
        print(f"  Recall (Sens.):      {m['recall_sensitivity']:.4f}")
        print(f"  Specificity:         {m['specificity']:.4f}")
        print(f"  F1 Score:            {m['f1_score']:.4f}")
        print(f"  Confusion Matrix:    TN={m['confusion_matrix']['true_negatives']}, FP={m['confusion_matrix']['false_positives']}, FN={m['confusion_matrix']['false_negatives']}, TP={m['confusion_matrix']['true_positives']}")

    # 7. Save Serialized Pipelines and Metadata
    print("\n[Step 7] Serializing trained model pipelines to models/...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(rf_pipeline, rf_path)
    print(f"Saved: {lr_path}")
    print(f"Saved: {rf_path}")

    # Extract feature names from fitted preprocessor
    fitted_preprocessor = lr_pipeline.named_steps["preprocessor"]
    transformed_feature_names = get_feature_names_after_preprocessing(fitted_preprocessor)

    metadata = {
        "dataset_source": "HDFS /heart-disease/raw/heart_disease_stream.jsonl",
        "total_records": len(X),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "random_state": 42,
        "clinical_features": CLINICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "transformed_features_count": len(transformed_feature_names),
        "transformed_feature_names": transformed_feature_names,
        "models": {
            "logistic_regression": {
                "file": "logistic_regression.joblib",
                "hyperparameters": lr_pipeline.named_steps["classifier"].get_params(),
                "cv_repeated_5fold_roc_auc_mean": round(float(cv_lr["test_roc_auc"].mean()), 4),
                "cv_repeated_5fold_roc_auc_std": round(float(cv_lr["test_roc_auc"].std()), 4),
                "cv_n_splits": 5,
                "cv_n_repeats": 5,
                "cv_total_folds": 25,
                "test_metrics": lr_metrics,
            },
            "random_forest": {
                "file": "random_forest.joblib",
                "hyperparameters": {
                    k: v for k, v in rf_pipeline.named_steps["classifier"].get_params().items()
                    if k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"]
                },
                "cv_repeated_5fold_roc_auc_mean": round(float(cv_rf["test_roc_auc"].mean()), 4),
                "cv_repeated_5fold_roc_auc_std": round(float(cv_rf["test_roc_auc"].std()), 4),
                "cv_n_splits": 5,
                "cv_n_repeats": 5,
                "cv_total_folds": 25,
                "test_metrics": rf_metrics,
            },
        },
    }

    meta_path = os.path.join(MODELS_DIR, "pipeline_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")

    # 8. Export Results & Visualizations
    print("\n[Step 8] Generating diagnostic plots and machine-readable metrics...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(RESULTS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
        }, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Save metrics comparison CSV
    comparison_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "CV Repeated ROC-AUC (Mean)": round(float(cv_lr["test_roc_auc"].mean()), 4),
            "CV Repeated ROC-AUC (Std)": round(float(cv_lr["test_roc_auc"].std()), 4),
            "Test ROC-AUC": lr_metrics["roc_auc"],
            "Test Accuracy": lr_metrics["accuracy"],
            "Test Precision": lr_metrics["precision"],
            "Test Recall": lr_metrics["recall_sensitivity"],
            "Test Specificity": lr_metrics["specificity"],
            "Test F1": lr_metrics["f1_score"],
            "Target > 0.85 Met": "Yes" if lr_metrics["roc_auc"] > 0.85 else "No",
        },
        {
            "Model": "Random Forest",
            "CV Repeated ROC-AUC (Mean)": round(float(cv_rf["test_roc_auc"].mean()), 4),
            "CV Repeated ROC-AUC (Std)": round(float(cv_rf["test_roc_auc"].std()), 4),
            "Test ROC-AUC": rf_metrics["roc_auc"],
            "Test Accuracy": rf_metrics["accuracy"],
            "Test Precision": rf_metrics["precision"],
            "Test Recall": rf_metrics["recall_sensitivity"],
            "Test Specificity": rf_metrics["specificity"],
            "Test F1": rf_metrics["f1_score"],
            "Target > 0.85 Met": "Yes" if rf_metrics["roc_auc"] > 0.85 else "No",
        },
    ])
    comparison_csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved: {comparison_csv_path}")

    # Save test predictions CSV for traceability
    test_preds_df = X_test.copy()
    test_preds_df["y_true"] = y_test.values
    test_preds_df["lr_pred"] = lr_pred
    test_preds_df["lr_proba"] = np.round(lr_proba, 4)
    test_preds_df["rf_pred"] = rf_pred
    test_preds_df["rf_proba"] = np.round(rf_proba, 4)
    test_preds_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved: {test_preds_path}")

    # Plot ROC Curves
    eval_list = [
        {"model_name": "Logistic Regression", "y_true": y_test.values, "y_proba": lr_proba, "metrics": lr_metrics},
        {"model_name": "Random Forest", "y_true": y_test.values, "y_proba": rf_proba, "metrics": rf_metrics},
    ]
    roc_plot_path = os.path.join(RESULTS_DIR, "roc_curve_comparison.png")
    plot_roc_curve_comparison(eval_list, roc_plot_path)
    print(f"Saved: {roc_plot_path}")

    # Plot Confusion Matrices
    cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plot_confusion_matrices(eval_list, cm_plot_path)
    print(f"Saved: {cm_plot_path}")

    # Plot Feature Importances / Coefficients
    lr_coefs = lr_pipeline.named_steps["classifier"].coef_[0]
    rf_importances = rf_pipeline.named_steps["classifier"].feature_importances_
    feat_plot_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plot_feature_importance_comparison(transformed_feature_names, lr_coefs, rf_importances, feat_plot_path)
    print(f"Saved: {feat_plot_path}")

    print("\n" + "=" * 70)
    print("PHASE 5 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    run_training_pipeline()
