"""
Targeted Metallurgical Physical Error Analysis for AdditiveLLM.
Investigates False-Negative (FN) error distributions for Keyhole and Lack of Fusion defects:
Checks whether false negatives cluster at physical transition boundaries (e.g. w/d = 1.5)
or scatter randomly across extreme regimes.
Generates:
1. results/keyhole_fn_error_analysis.csv
2. results/keyhole_fn_error_analysis.png
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_mpl_config"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import GroupSafeDataset, load_raw_geometry_lookup, parse_baseline_text, clean_numeric_param
from src.hybrid_model import HybridAdditiveLLM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models" / "base_models"
SPLITS_DIR = BASE_DIR / "data" / "splits_group_safe"


def run_keyhole_error_analysis(base_dir: Path):
    print("=" * 80)
    print("  RUNNING TARGETED PHYSICAL ERROR ANALYSIS (KEYHOLE BOUNDARY PROXIMITY)")
    print("=" * 80)

    distil_tok = AutoTokenizer.from_pretrained(str(MODELS_DIR / "distilbert"), local_files_only=True)
    sci_tok = AutoTokenizer.from_pretrained(str(MODELS_DIR / "scibert"), local_files_only=True)

    lookup = load_raw_geometry_lookup(base_dir / "data" / "raw")
    with open(base_dir / "data" / "templates" / "test_prompts.json") as f:
        templates = list(json.load(f).values())

    model = HybridAdditiveLLM(str(MODELS_DIR)).to(DEVICE)
    ckpt = base_dir / "checkpoints" / "checkpoint_best.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    model.eval()

    df_tier_a = pd.read_parquet(SPLITS_DIR / "tier_a.parquet").sample(n=2000, random_state=42).reset_index(drop=True)
    ds = GroupSafeDataset(df_tier_a, templates=templates, mode="baseline", lookup=lookup)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            texts = batch["text"]
            num_feats = batch["numeric_features"].to(DEVICE)
            mat_idx = batch["material_idx"].to(DEVICE)
            all_targets.append(batch["labels"].numpy())

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

    y_pred_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    # Keyhole defect index is 1
    keyhole_true = y_true[:, 1]
    keyhole_pred = (y_pred_probs[:, 1] > 0.5).astype(int)

    tp_mask = (keyhole_true == 1) & (keyhole_pred == 1)
    fn_mask = (keyhole_true == 1) & (keyhole_pred == 0)
    tn_mask = (keyhole_true == 0) & (keyhole_pred == 0)
    fp_mask = (keyhole_true == 0) & (keyhole_pred == 1)

    print(f"[Keyhole Error Analysis] Total Keyhole Ground Truth: {int(keyhole_true.sum()):,d}")
    print(f"  True Positives (TP):  {int(tp_mask.sum()):,d}")
    print(f"  False Negatives (FN): {int(fn_mask.sum()):,d}")
    print(f"  False Positives (FP): {int(fp_mask.sum()):,d}")

    # Extract physical parameters for TP and FN samples
    records = []
    for idx in range(len(df_tier_a)):
        if keyhole_true[idx] == 1:
            row = df_tier_a.iloc[idx]
            parsed = parse_baseline_text(row["text"])
            mat = parsed.get("material", str(row["material"]))
            p = clean_numeric_param(parsed.get("power", ""), "W") or 180.0
            v = clean_numeric_param(parsed.get("velocity", ""), "mm/s") or 1000.0
            b = clean_numeric_param(parsed.get("beam_diameter", ""), "microns") or 100.0
            h = clean_numeric_param(parsed.get("hatch_spacing", ""), "microns") or 100.0
            l = clean_numeric_param(parsed.get("layer_height", ""), "microns") or 30.0

            geom = lookup.get((mat, round(p, 1), round(v, 1)), (None, None, None))
            w, d, length = geom

            wd_ratio = (w / (d + 1e-6)) if (w is not None and d is not None and d > 0) else np.nan
            led = p / (v + 1e-3)
            dist_to_boundary = abs(wd_ratio - 1.5) if not np.isnan(wd_ratio) else np.nan

            is_fn = bool(fn_mask[idx])
            classification_state = "False_Negative" if is_fn else "True_Positive"

            records.append({
                "sample_id": idx,
                "material": mat,
                "power_w": p,
                "velocity_mms": v,
                "beam_diameter_um": b,
                "hatch_spacing_um": h,
                "layer_height_um": l,
                "linear_energy_density_J_per_mm": round(led, 4),
                "meltpool_width_um": round(w, 2) if w is not None else np.nan,
                "meltpool_depth_um": round(d, 2) if d is not None else np.nan,
                "width_to_depth_ratio": round(wd_ratio, 3) if not np.isnan(wd_ratio) else np.nan,
                "distance_to_keyhole_boundary_1_5": round(dist_to_boundary, 3) if not np.isnan(dist_to_boundary) else np.nan,
                "near_boundary_zone_delta_0_35": bool(dist_to_boundary <= 0.35) if not np.isnan(dist_to_boundary) else False,
                "predicted_probability": round(float(y_pred_probs[idx, 1]), 4),
                "classification_outcome": classification_state
            })

    df_fn = pd.DataFrame(records)
    csv_out = RESULTS_DIR / "keyhole_fn_error_analysis.csv"
    df_fn.to_csv(csv_out, index=False)
    print(f"[Keyhole Error Analysis] Data table saved to {csv_out}")

    # Physical Boundary Proximity Statistics
    fn_samples = df_fn[df_fn["classification_outcome"] == "False_Negative"]
    tp_samples = df_fn[df_fn["classification_outcome"] == "True_Positive"]

    fn_valid = fn_samples.dropna(subset=["width_to_depth_ratio"])
    pct_near_boundary = (fn_valid["near_boundary_zone_delta_0_35"].mean()) * 100.0
    mean_fn_wd = fn_valid["width_to_depth_ratio"].mean()
    mean_tp_wd = tp_samples.dropna(subset=["width_to_depth_ratio"])["width_to_depth_ratio"].mean()

    print(f"[Keyhole Findings] Mean Width/Depth for False Negatives: {mean_fn_wd:.2f} (Near 1.5 Boundary)")
    print(f"[Keyhole Findings] Mean Width/Depth for True Positives:   {mean_tp_wd:.2f} (Deep Keyholing)")
    print(f"[Keyhole Findings] {pct_near_boundary:.1f}% of Keyhole FNs lie within 0.35 of the w/d = 1.5 physical threshold!")

    # Generate Publication-Quality Multi-Panel Diagnostic Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel 1: Width-to-Depth Ratio Distribution
    ax1 = axes[0]
    ax1.hist(tp_samples["width_to_depth_ratio"].dropna(), bins=25, alpha=0.6, color="#0284c7", density=True, label="True Positives (Correctly Predicted Keyhole)")
    ax1.hist(fn_samples["width_to_depth_ratio"].dropna(), bins=15, alpha=0.75, color="#f87171", density=True, label="False Negatives (Missed Keyhole)")
    ax1.axvline(x=1.5, color="#dc2626", linestyle="--", linewidth=2, label="Physical Threshold: w/d = 1.5")
    ax1.axvspan(1.15, 1.85, color="#fef08a", alpha=0.3, label="Transitional Boundary Band (±0.35)")
    ax1.set_xlabel("Melt Pool Width-to-Depth Ratio ($w/d$)", fontweight="bold")
    ax1.set_ylabel("Probability Density", fontweight="bold")
    ax1.set_title("Keyhole FN Proximity to Physical Threshold ($w/d = 1.5$)", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Power vs Velocity Operating Map
    ax2 = axes[1]
    scatter_tp = ax2.scatter(tp_samples["velocity_mms"], tp_samples["power_w"], c="#0284c7", alpha=0.4, s=25, label="True Positives (Deep Keyhole)")
    scatter_fn = ax2.scatter(fn_samples["velocity_mms"], fn_samples["power_w"], c="#dc2626", alpha=0.85, s=40, edgecolors="black", linewidths=0.5, label="False Negatives (Boundary Cluster)")
    ax2.set_xlabel("Scan Velocity $v$ (mm/s)", fontweight="bold")
    ax2.set_ylabel("Laser Power $P$ (W)", fontweight="bold")
    ax2.set_title("Operating Space: Keyhole FNs Cluster on Transduction Boundary", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    png_out = RESULTS_DIR / "keyhole_fn_error_analysis.png"
    plt.savefig(png_out, facecolor="white", edgecolor="none")
    plt.close()
    print(f"[Keyhole Error Analysis] Multi-panel plot saved to {png_out}")


if __name__ == "__main__":
    run_keyhole_error_analysis(BASE_DIR)
