"""
Model Evaluation and Visualization Module for Heart Disease Prediction.

Provides utilities for:
1. Calculating comprehensive classification metrics (ROC-AUC, Accuracy, Precision, Recall, Specificity, F1, Confusion Matrix).
2. Generating high-resolution publication-quality diagnostic plots (ROC curves, Confusion Matrices, Feature Importances).
3. Exporting machine-readable evaluation results (JSON and CSV).
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Compute full suite of classification performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": report,
    }


def plot_roc_curve_comparison(
    models_eval: List[Dict[str, Any]],
    output_path: str,
    title: str = "ROC Curve Comparison - Heart Disease Risk Prediction",
) -> str:
    """
    Generate and save ROC curve comparison plot for multiple models.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, model_data in enumerate(models_eval):
        y_true = model_data["y_true"]
        y_proba = model_data["y_proba"]
        name = model_data["model_name"]
        auc = model_data["metrics"]["roc_auc"]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.4f})",
            color=colors[idx % len(colors)],
            linewidth=2.5,
        )

    # Reference chance diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)", linewidth=1.5, alpha=0.7)
    # Target threshold annotation
    plt.axhline(0.85, color="gray", linestyle=":", alpha=0.5, label="Project Target AUC (0.85)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_confusion_matrices(
    models_eval: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Generate and save side-by-side confusion matrix heatmaps.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_models = len(models_eval)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), dpi=300, squeeze=False)

    for idx, model_data in enumerate(models_eval):
        ax = axes[0, idx]
        cm_dict = model_data["metrics"]["confusion_matrix"]
        cm_matrix = np.array([
            [cm_dict["true_negatives"], cm_dict["false_positives"]],
            [cm_dict["false_negatives"], cm_dict["true_positives"]],
        ])

        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Absence (0)", "Presence (1)"],
            yticklabels=["Absence (0)", "Presence (1)"],
        )
        ax.set_title(
            f"{model_data['model_name']}\nAccuracy: {model_data['metrics']['accuracy']:.2%} | F1: {model_data['metrics']['f1_score']:.4f}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Predicted Diagnosis", fontsize=11)
        ax.set_ylabel("True Diagnosis", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_importance_comparison(
    feature_names: List[str],
    lr_coefficients: np.ndarray,
    rf_importances: np.ndarray,
    output_path: str,
) -> str:
    """
    Generate and save bar plots comparing Logistic Regression coefficients and Random Forest feature importances.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300)

    # 1. Logistic Regression Coefficients (Log-Odds)
    lr_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr_coefficients,
    }).sort_values(by="coefficient", key=abs, ascending=True).tail(12)

    colors = ["#d62728" if c > 0 else "#1f77b4" for c in lr_df["coefficient"]]
    ax1.barh(lr_df["feature"], lr_df["coefficient"], color=colors)
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Logistic Regression (Top Coefficients / Log-Odds)\nRed = Positive Risk, Blue = Protective", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Coefficient Weight", fontsize=10)

    # 2. Random Forest Gini Importance
    rf_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_importances,
    }).sort_values(by="importance", ascending=True).tail(12)

    ax2.barh(rf_df["feature"], rf_df["importance"], color="#2ca02c")
    ax2.set_title("Random Forest (Top Gini Feature Importances)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
