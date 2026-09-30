"""
Random Forest Multi-Label Baseline on Identical Group-Safe Splits.
Trains a MultiOutputClassifier(RandomForestClassifier) on the exact same group-aware train split,
and evaluates performance across Tier A (In-Distribution), Tier B (Held-Out Groups), and Tier C (Held-Out Materials).
Saves comparative evaluation metrics to results/random_forest_baseline.csv.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_pipeline import parse_baseline_text, extract_numeric_features, KNOWN_MATERIALS


def prepare_tabular_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numeric feature matrix X and multi-label binary matrix Y."""
    X_list = []
    Y_list = []

    for _, row in df.iterrows():
        p = parse_baseline_text(row["text"])
        feats, mat_idx = extract_numeric_features(p)
        # One-hot encode material (30 materials)
        mat_one_hot = np.zeros(len(KNOWN_MATERIALS), dtype=np.float32)
        if mat_idx < len(KNOWN_MATERIALS):
            mat_one_hot[mat_idx] = 1.0

        full_features = np.concatenate([feats, mat_one_hot])
        X_list.append(full_features)

        lbl = row["label"]
        if isinstance(lbl, str):
            import ast
            lbl = ast.literal_eval(lbl)
        Y_list.append(list(lbl))

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.int32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute exact match ratio, mean binary accuracy, and F1 metrics."""
    exact_match = np.mean(np.all(y_true == y_pred, axis=1))
    mean_binary_acc = np.mean(y_true == y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return {
        "mean_binary_accuracy": float(mean_binary_acc),
        "exact_match_ratio": float(exact_match),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }


def run_random_forest_baseline():
    print("=" * 70)
    print("  RANDOM FOREST MULTI-LABEL BASELINE EVALUATION (TIERS A, B, C)")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "splits_group_safe"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Random Forest] Loading training and tier validation parquets...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    tier_a_df = pd.read_parquet(data_dir / "tier_a.parquet")
    tier_b_df = pd.read_parquet(data_dir / "tier_b.parquet")
    tier_c_df = pd.read_parquet(data_dir / "tier_c.parquet")

    # Sample for training tabular model if large
    if len(train_df) > 30000:
        train_df = train_df.sample(n=30000, random_state=42)

    X_train, Y_train = prepare_tabular_dataset(train_df)
    X_tier_a, Y_tier_a = prepare_tabular_dataset(tier_a_df)
    X_tier_b, Y_tier_b = prepare_tabular_dataset(tier_b_df)
    X_tier_c, Y_tier_c = prepare_tabular_dataset(tier_c_df)

    print(f"[Random Forest] Training on {len(X_train):,} samples with 100 decision trees...")
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_model = MultiOutputClassifier(rf_base)
    rf_model.fit(X_train, Y_train)
    print("[Random Forest] Fitting completed!")

    tiers = [
        ("Tier A (In-Distribution)", X_tier_a, Y_tier_a),
        ("Tier B (Held-Out Physical Groups)", X_tier_b, Y_tier_b),
        ("Tier C (Held-Out Materials OOD)", X_tier_c, Y_tier_c),
    ]

    results_rows = []
    print("\n" + "-" * 70)
    print(f"{'Evaluation Tier':<35} {'Binary Acc':<12} {'Exact Match':<12} {'Macro F1':<10}")
    print("-" * 70)

    for tier_name, X_eval, Y_eval in tiers:
        preds = rf_model.predict(X_eval)
        m = compute_metrics(Y_eval, preds)
        results_rows.append({
            "model": "Random Forest Baseline",
            "tier": tier_name,
            "samples": len(X_eval),
            "mean_binary_accuracy_pct": round(m["mean_binary_accuracy"] * 100, 2),
            "exact_match_pct": round(m["exact_match_ratio"] * 100, 2),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "micro_f1": round(m["micro_f1"], 4),
        })
        print(f"{tier_name:<35} {m['mean_binary_accuracy']*100:<11.2f}% {m['exact_match_ratio']*100:<11.2f}% {m['macro_f1']:<10.4f}")

    print("-" * 70)

    out_csv = results_dir / "random_forest_baseline.csv"
    res_df = pd.DataFrame(results_rows)
    res_df.to_csv(out_csv, index=False)
    print(f"\n[Random Forest] Direct comparison metrics saved to {out_csv}")
    return res_df


if __name__ == "__main__":
    run_random_forest_baseline()
