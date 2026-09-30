"""
All-Tiers Multi-Panel Evaluation & Pooled Aggregate Metric Engine.
Generates:
1. results/overall_aggregate_metrics.csv (Format B: True pooled predictions across Tier A + Tier B + Tier C)
2. results/confusion_matrices/ (Format A: Confusion matrices for Tier A, Tier B, Tier C x Baseline, Prompt)
3. results/calibration_{tier}_{format}.png (Format A: Reliability calibration diagrams for all tiers)
4. results/all_tiers_comparison.png (Format A: Multi-panel publication figure comparing Tiers A, B, C)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_mpl_config"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import GroupSafeDataset, load_raw_geometry_lookup
from src.hybrid_model import HybridAdditiveLLM
from src.evaluate import compute_ece

DEFECT_CLASSES = ["None", "Keyhole", "Lack_of_Fusion", "Balling"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models" / "base_models"
SPLITS_DIR = BASE_DIR / "data" / "splits_group_safe"


def plot_tier_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, tier_label: str, fmt: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=300)
    for idx, (cls_name, ax) in enumerate(zip(DEFECT_CLASSES, axes)):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        cm = confusion_matrix(yt, yp, labels=[0, 1])

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{cls_name}", fontweight="bold", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], fontsize=9)
        ax.set_yticklabels(["Neg", "Pos"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, f"{val:,d}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.suptitle(f"{tier_label} ({fmt.capitalize()} Format) Defect Confusion Matrices", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_tier_calibration(y_true: np.ndarray, y_probs: np.ndarray, tier_label: str, fmt: str, out_png: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()
    colors = ["#1e40af", "#b45309", "#6d28d9", "#be185d"]
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for idx, (cls_name, ax, color) in enumerate(zip(DEFECT_CLASSES, axes, colors)):
        p = y_probs[:, idx]
        y = y_true[:, idx]

        bin_confs, bin_accs = [], []
        for i in range(n_bins):
            b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (p >= b_low) & (p < b_high) if i < n_bins - 1 else (p >= b_low) & (p <= b_high)
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(y[mask]))
                bin_confs.append(np.mean(p[mask]))

        ece_c = compute_ece(p, y, n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
        if bin_confs:
            ax.plot(bin_confs, bin_accs, marker="o", color=color, linewidth=2, label=f"ECE: {ece_c:.4f}")
            ax.bar(bin_confs, bin_accs, width=0.04, alpha=0.25, color=color)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Mean Confidence", fontsize=9)
        ax.set_ylabel("True Fraction", fontsize=9)
        ax.set_title(f"{cls_name} (ECE: {ece_c:.4f})", fontweight="bold", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{tier_label} ({fmt.capitalize()}) Calibration Reliability Diagrams", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def run_all_tiers_evaluation(base_dir: Path):
    print("=" * 80, flush=True)
    print("  RUNNING ALL-TIERS EVALUATION & COMPUTING POOLED OVERALL METRICS", flush=True)
    print("=" * 80, flush=True)

    distil_tok = AutoTokenizer.from_pretrained(str(MODELS_DIR / "distilbert"), local_files_only=True)
    sci_tok = AutoTokenizer.from_pretrained(str(MODELS_DIR / "scibert"), local_files_only=True)
    tok_dict = {"distilbert": distil_tok, "scibert": sci_tok}

    lookup = load_raw_geometry_lookup(base_dir / "data" / "raw")
    with open(base_dir / "data" / "templates" / "test_prompts.json") as f:
        test_templates = list(json.load(f).values())

    model = HybridAdditiveLLM(str(MODELS_DIR)).to(DEVICE)
    ckpt_path = base_dir / "checkpoints" / "checkpoint_best.pt"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    model.eval()

    tier_dfs = [
        ("Tier A (In-Distribution)", "tier_a", pd.read_parquet(SPLITS_DIR / "tier_a.parquet")),
        ("Tier B (Held-Out Groups)", "tier_b", pd.read_parquet(SPLITS_DIR / "tier_b.parquet")),
        ("Tier C (Held-Out Material OOD)", "tier_c", pd.read_parquet(SPLITS_DIR / "tier_c.parquet")),
    ]

    tier_results = []
    pooled_data = {"baseline": {"probs": [], "targets": []}, "prompt": {"probs": [], "targets": []}}

    cm_dir = RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for tier_label, tier_key, df_raw in tier_dfs:
        for fmt in ["baseline", "prompt"]:
            # Evaluate representative sample of tier (1500 for Tier A, 1200 for B/C)
            n_eval = min(1500 if tier_key == "tier_a" else 1200, len(df_raw))
            df_sub = df_raw.sample(n=n_eval, random_state=42).reset_index(drop=True)
            ds = GroupSafeDataset(df_sub, templates=test_templates, mode=fmt, lookup=lookup)
            loader = DataLoader(ds, batch_size=64, shuffle=False)

            all_probs, all_targets = [], []
            with torch.no_grad():
                for batch in loader:
                    texts = batch["text"]
                    num_feats = batch["numeric_features"].to(DEVICE)
                    mat_idx = batch["material_idx"].to(DEVICE)
                    targets = batch["labels"].numpy()

                    enc_d = distil_tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(DEVICE)
                    enc_s = sci_tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(DEVICE)

                    out = model(
                        input_ids_distil=enc_d["input_ids"],
                        attention_mask_distil=enc_d["attention_mask"],
                        input_ids_sci=enc_s["input_ids"],
                        attention_mask_sci=enc_s["attention_mask"],
                        numeric_features=num_feats,
                        material_idx=mat_idx
                    )
                    probs = torch.sigmoid(out["logits"]).cpu().numpy()
                    all_probs.append(probs)
                    all_targets.append(targets)

            yp = np.vstack(all_probs)
            yt = np.vstack(all_targets)
            yp_bin = (yp > 0.5).astype(int)

            # Store for pooled aggregate calculation
            pooled_data[fmt]["probs"].append(yp)
            pooled_data[fmt]["targets"].append(yt)

            acc = (yp_bin == yt).mean() * 100.0
            f1 = f1_score(yt, yp_bin, average="macro", zero_division=0)
            prec = precision_score(yt, yp_bin, average="macro", zero_division=0)
            rec = recall_score(yt, yp_bin, average="macro", zero_division=0)
            ece = compute_ece(yp, yt, n_bins=10)

            tier_results.append({
                "Tier": tier_label,
                "Format": fmt.capitalize(),
                "Accuracy_Pct": round(acc, 2),
                "Macro_F1": round(f1, 4),
                "Macro_Precision": round(prec, 4),
                "Macro_Recall": round(rec, 4),
                "ECE": round(ece, 4),
                "Sample_Count": len(yt)
            })

            # Save individual tier confusion matrix & calibration diagram
            plot_tier_confusion_matrix(yt, yp_bin, tier_label, fmt, cm_dir / f"confusion_matrix_{tier_key}_{fmt}.png")
            plot_tier_calibration(yt, yp, tier_label, fmt, RESULTS_DIR / f"calibration_{tier_key}_{fmt}.png")

            # Save per-class metrics CSV for each tier & format
            class_metrics = []
            for c_idx, c_name in enumerate(DEFECT_CLASSES):
                c_yt = yt[:, c_idx]
                c_yp = yp_bin[:, c_idx]
                c_acc = accuracy_score(c_yt, c_yp) * 100.0
                c_f1 = f1_score(c_yt, c_yp, zero_division=0)
                c_prec = precision_score(c_yt, c_yp, zero_division=0)
                c_rec = recall_score(c_yt, c_yp, zero_division=0)
                cm_c = confusion_matrix(c_yt, c_yp, labels=[0, 1])
                tn, fp, fn, tp = cm_c.ravel()
                class_metrics.append({
                    "defect_class": c_name,
                    "accuracy_pct": round(c_acc, 2),
                    "precision": round(c_prec, 4),
                    "recall": round(c_rec, 4),
                    "f1_score": round(c_f1, 4),
                    "false_negative_rate_pct": round((fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0, 2),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "support": int(tp + fn)
                })
            df_class = pd.DataFrame(class_metrics)
            df_class.to_csv(RESULTS_DIR / f"metrics_{tier_key}_{fmt}.csv", index=False)
            print(f"  [{tier_label} {fmt.capitalize()}] Acc: {acc:.2f}% | F1: {f1:.4f} | ECE: {ece:.4f}", flush=True)

    # --------------------------------------------------------------------------
    # FORMAT B: COMPUTING GENUINE POOLED OVERALL AGGREGATE METRICS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("  COMPUTING FORMAT B: GENUINE POOLED OVERALL AGGREGATE METRICS", flush=True)
    print("=" * 80, flush=True)

    overall_records = []
    for fmt in ["baseline", "prompt"]:
        all_p = np.vstack(pooled_data[fmt]["probs"])
        all_t = np.vstack(pooled_data[fmt]["targets"])
        all_bin = (all_p > 0.5).astype(int)

        pooled_acc = (all_bin == all_t).mean() * 100.0
        pooled_f1 = f1_score(all_t, all_bin, average="macro", zero_division=0)
        pooled_prec = precision_score(all_t, all_bin, average="macro", zero_division=0)
        pooled_rec = recall_score(all_t, all_bin, average="macro", zero_division=0)
        pooled_ece = compute_ece(all_p, all_t, n_bins=10)

        # Plot overall pooled confusion matrix & calibration
        plot_tier_confusion_matrix(all_t, all_bin, "Overall (All Tiers Combined)", fmt, cm_dir / f"confusion_matrix_overall_pooled_{fmt}.png")
        plot_tier_calibration(all_t, all_p, "Overall (All Tiers Combined)", fmt, RESULTS_DIR / f"calibration_overall_pooled_{fmt}.png")

        overall_records.append({
            "Evaluation_Scope": "Overall (All Tiers Pooled: Tier A + B + C)",
            "Format": fmt.capitalize(),
            "Pooled_Accuracy_Pct": round(pooled_acc, 2),
            "Pooled_Macro_F1": round(pooled_f1, 4),
            "Pooled_Macro_Precision": round(pooled_prec, 4),
            "Pooled_Macro_Recall": round(pooled_rec, 4),
            "Pooled_ECE": round(pooled_ece, 4),
            "Total_Test_Samples_Pooled": len(all_t),
            "Notes": "Genuine micro-pooled sample aggregation across seen, held-out groups, and held-out alloy IN625."
        })

    df_overall = pd.DataFrame(overall_records)
    out_overall = RESULTS_DIR / "overall_aggregate_metrics.csv"
    df_overall.to_csv(out_overall, index=False)
    print(f"[Format B] Saved overall aggregate metrics to {out_overall}")
    print(df_overall.to_string())

    # Multi-Tier Side-by-Side Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    df_tiers = pd.DataFrame(tier_results)

    tiers_base = df_tiers[df_tiers["Format"] == "Baseline"]
    tiers_prom = df_tiers[df_tiers["Format"] == "Prompt"]
    x = np.arange(3)
    width = 0.35

    # Panel 1: Accuracy
    ax1.bar(x - width/2, tiers_base["Accuracy_Pct"], width, label="Baseline ([SEP])", color="#1e40af")
    ax1.bar(x + width/2, tiers_prom["Accuracy_Pct"], width, label="Prompt (Natural Lang)", color="#0284c7")
    ax1.set_ylabel("Defect Prediction Accuracy (%)", fontweight="bold")
    ax1.set_title("Accuracy Across All Evaluation Tiers", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Tier A\n(In-Distribution)", "Tier B\n(Held-Out Groups)", "Tier C\n(Held-Out Alloy)"], fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3, axis="y")

    for i in x:
        ax1.text(i - width/2, tiers_base["Accuracy_Pct"].iloc[i] + 1.5, f"{tiers_base['Accuracy_Pct'].iloc[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + width/2, tiers_prom["Accuracy_Pct"].iloc[i] + 1.5, f"{tiers_prom['Accuracy_Pct'].iloc[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Macro F1
    ax2.bar(x - width/2, tiers_base["Macro_F1"], width, label="Baseline ([SEP])", color="#059669")
    ax2.bar(x + width/2, tiers_prom["Macro_F1"], width, label="Prompt (Natural Lang)", color="#10b981")
    ax2.set_ylabel("Macro F1-Score", fontweight="bold")
    ax2.set_title("Macro F1-Score Across All Evaluation Tiers", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Tier A\n(In-Distribution)", "Tier B\n(Held-Out Groups)", "Tier C\n(Held-Out Alloy)"], fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3, axis="y")

    for i in x:
        ax2.text(i - width/2, tiers_base["Macro_F1"].iloc[i] + 0.02, f"{tiers_base['Macro_F1'].iloc[i]:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax2.text(i + width/2, tiers_prom["Macro_F1"].iloc[i] + 0.02, f"{tiers_prom['Macro_F1'].iloc[i]:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Format A: All-Tier Performance Comparison (Hybrid AdditiveLLM)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    comparison_png = RESULTS_DIR / "all_tiers_comparison.png"
    plt.savefig(comparison_png, bbox_inches="tight")
    plt.close()
    print(f"[Format A] Multi-tier comparison figure saved to {comparison_png}")

    # Regenerate final_comparison.csv & final_comparison.md with all three tiers side-by-side
    df_tiers.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)

    def df_to_md(df: pd.DataFrame) -> str:
        headers = list(df.columns)
        lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)

    md_content = "# Comprehensive All-Tier Benchmark: Hybrid AdditiveLLM\n\n"
    md_content += "### Format A: All Three Evaluation Tiers Side-by-Side\n\n"
    md_content += df_to_md(df_tiers)
    md_content += "\n\n### Format B: Genuine Pooled Overall Aggregate Metrics\n\n"
    md_content += df_to_md(df_overall)
    with open(RESULTS_DIR / "final_comparison.md", "w") as f:
        f.write(md_content)

    return df_tiers, df_overall


if __name__ == "__main__":
    run_all_tiers_evaluation(BASE_DIR)
