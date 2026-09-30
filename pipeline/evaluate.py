"""
Evaluation Engine for AdditiveLLM Models.
Computes multi-label accuracy (mean per-label binary accuracy), exact-match ratio,
macro/micro/weighted F1, per-class metrics, and confusion matrices.
Saves raw predictions, probabilities, and metric summaries to disk.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import torch

CLASS_NAMES = ["None", "Keyhole", "Lack of Fusion", "Balling"]

def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    """
    Computes all standard paper metrics for multi-label classification.
    y_true: shape (N, 4)
    y_pred: shape (N, 4) binary {0, 1}
    y_prob: shape (N, 4) continuous [0, 1]
    """
    num_samples, num_classes = y_true.shape
    
    # 1. Paper's primary metric: Mean per-label binary accuracy
    per_label_acc = []
    for c in range(num_classes):
        acc_c = accuracy_score(y_true[:, c], y_pred[:, c])
        per_label_acc.append(float(acc_c))
    mean_binary_accuracy = float(np.mean(per_label_acc))
    
    # 2. Secondary metric: Exact Match Ratio (Subset Accuracy)
    exact_matches = np.all(y_true == y_pred, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))
    
    # 3. Overall F1 scores
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 4. Per-class metrics (precision, recall, F1, support)
    per_class_metrics = {}
    confusion_matrices = {}
    for c, name in enumerate(CLASS_NAMES):
        prec = float(precision_score(y_true[:, c], y_pred[:, c], zero_division=0))
        rec = float(recall_score(y_true[:, c], y_pred[:, c], zero_division=0))
        f1 = float(f1_score(y_true[:, c], y_pred[:, c], zero_division=0))
        support = int(np.sum(y_true[:, c]))
        
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])
        # cm is [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = map(int, cm.ravel())
        
        per_class_metrics[name] = {
            "binary_accuracy": per_label_acc[c],
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        }
        confusion_matrices[name] = {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "matrix": [[tn, fp], [fn, tp]]
        }

    return {
        "num_samples": num_samples,
        "mean_binary_accuracy": mean_binary_accuracy,
        "exact_match_ratio": exact_match_ratio,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics,
        "confusion_matrices": confusion_matrices,
    }

def save_evaluation_artifacts(
    model_name: str,
    dataset_format: str,
    epoch: str,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str = "reports"
) -> Path:
    """Save metrics JSON, prediction arrays, and summaries to disk."""
    out_dir = Path(output_dir) / "evaluations" / f"{model_name}_{dataset_format}_{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save compact numpy arrays
    np.save(out_dir / "y_true.npy", y_true.astype(np.int8))
    np.save(out_dir / "y_pred.npy", y_pred.astype(np.int8))
    np.save(out_dir / "y_prob.npy", y_prob.astype(np.float32))
    
    print(f"[Evaluate] Saved evaluation artifacts to {out_dir}")
    return out_dir
