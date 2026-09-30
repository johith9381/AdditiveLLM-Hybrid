"""
Visualizations for Class Imbalance Analysis and Data Preprocessing Pipeline.
Generates:
1. results/class_imbalance_analysis.png (Raw class distribution vs loss-weighted balanced gradient impact)
2. results/preprocessing_pipeline_flowchart.png (Technical pipeline flowchart of Group-Aware split & preprocessing)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_mpl_config"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SPLITS_DIR = BASE_DIR / "data" / "splits_group_safe"


def generate_class_imbalance_plot():
    print("[Visualization] Generating class imbalance analysis plot...", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    
    # Parse labels
    import ast
    all_labels = []
    for l in df_train["label"].sample(min(20000, len(df_train)), random_state=42):
        if isinstance(l, str):
            l = ast.literal_eval(l)
        all_labels.append(l)
    labels_arr = np.array(all_labels)

    class_names = ["None\n(Conduction)", "Keyhole\nPorosity", "Lack of Fusion\nPorosity", "Balling\nPorosity"]
    raw_counts = labels_arr.sum(axis=0)
    total = len(labels_arr)
    raw_pcts = (raw_counts / total) * 100.0

    # pos_weights used in training
    pos_weights = np.array([0.45, 2.10, 1.85, 4.30])
    effective_weights = raw_counts * pos_weights
    balanced_pcts = (effective_weights / effective_weights.sum()) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Raw Class Distribution
    colors_raw = ["#0284c7", "#ef4444", "#f59e0b", "#8b5cf6"]
    bars1 = axes[0].bar(class_names, raw_pcts, color=colors_raw, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[0].set_title("Raw Dataset Class Distribution (Imbalanced)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_ylabel("Occurrence Percentage (%)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 55)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count, pct in zip(bars1, raw_counts, raw_pcts):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{pct:.1f}%\n({int(count):,})",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Effective Gradient Impact After pos_weight Balancing
    colors_bal = ["#38bdf8", "#f87171", "#fbbf24", "#a78bfa"]
    bars2 = axes[1].bar(class_names, balanced_pcts, color=colors_bal, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    axes[1].set_title("Effective Gradient Impact (BCE pos_weight Balanced)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("Effective Loss Weight Share (%)", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 55)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    for bar, w, pct in zip(bars2, pos_weights, balanced_pcts):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{pct:.1f}%\n(x{w:.2f})",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Additive Manufacturing Defect Class Balancing in Training", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = RESULTS_DIR / "class_imbalance_analysis.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved class imbalance plot to {out_path}", flush=True)


def generate_preprocessing_pipeline_flowchart():
    print("[Visualization] Generating preprocessing pipeline flowchart...", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, w, h, title, details, bg_color="#1e293b", border_color="#38bdf8", text_color="#f8fafc"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center", fontsize=11, fontweight="bold", color=border_color)
        ax.text(x + w / 2, y + h / 2 - 0.25, details, ha="center", va="center", fontsize=8.5, color=text_color, linespacing=1.4)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2, mutation_scale=15))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.2, label, ha="center", fontsize=8, color="#cbd5e1", fontweight="bold")

    # Title
    ax.text(8, 9.5, "AdditiveLLM Preprocessing & Leakage-Free Data Flow", ha="center", fontsize=15, fontweight="bold", color="#0f172a")

    # Column 1: Raw Sources
    draw_box(0.5, 6.0, 3.2, 2.5, "1. Raw AM Sources",
             "• NIST AM-Bench 2018/2022\n• FLOW-3D Single-Track Sims\n• 13 Commercial Alloys\n• Experimental Melt Pools (w, d, l)",
             border_color="#0284c7")

    draw_box(0.5, 2.0, 3.2, 2.8, "2. Cleaning & Scaling",
             "• Power P in Watts [50 - 500 W]\n• Scan speed v [100 - 3000 mm/s]\n• Beam d [40 - 200 µm]\n• Hatch h [40 - 250 µm]\n• Layer t [20 - 100 µm]",
             border_color="#0284c7")

    draw_arrow(2.1, 6.0, 2.1, 4.8)

    # Column 2: Leakage-Free Group Partitioning
    draw_box(4.8, 4.0, 3.4, 4.5, "3. Group-Safe Split",
             "• Locked Group A Key:\n  (P, v, d, Material)\n• GroupShuffleSplit (70/15/15)\n  executed BEFORE augmentation\n• 0% Sibling Cross-Contamination\n• Verified 0.00% overlap:\n  results/leakage_check.txt",
             border_color="#10b981", bg_color="#064e3b")

    draw_arrow(3.7, 3.4, 4.8, 5.5, label="Raw Experiments")

    # Column 3: Split Tiers & Augmentation
    draw_box(9.2, 6.5, 3.2, 2.2, "Tier A: In-Distribution",
             "• Seen Alloys & Parameters\n• 83,074 Instances\n• Standard unseen test set",
             border_color="#3b82f6")

    draw_box(9.2, 3.8, 3.2, 2.2, "Tier B: Held-Out Groups",
             "• Unseen Physical Experiments\n• Sibling-Free Group-Aware\n• 130,135 Instances",
             border_color="#6366f1")

    draw_box(9.2, 1.1, 3.2, 2.2, "Tier C: Held-Out Alloy OOD",
             "• Complete Hold-Out: IN625\n• Unseen Material OOD\n• 38,897 Instances",
             border_color="#f59e0b", bg_color="#451a03")

    draw_arrow(8.2, 6.5, 9.2, 7.6)
    draw_arrow(8.2, 5.5, 9.2, 4.9)
    draw_arrow(8.2, 4.5, 9.2, 2.2)

    # Column 4: Multi-Modal Formatting
    draw_box(13.2, 5.2, 2.6, 3.5, "4. Input Syntax",
             "• Baseline Format:\n  Structured [SEP] tokens\n• Prompt Format:\n  Natural Language\n  Template Sampling\n• Continuous Physics:\n  [P, v, d, h, t] normalized",
             border_color="#8b5cf6")

    draw_box(13.2, 1.2, 2.6, 3.2, "5. Physics Targets",
             "• Defect Ground Truth:\n  [None, Keyhole, LoF, Balling]\n• Auxiliary Hydrodynamics:\n  w/d <= 1.5 -> Keyhole\n  l/w >= pi -> Balling",
             border_color="#ec4899")

    draw_arrow(12.4, 6.5, 13.2, 6.5)
    draw_arrow(12.4, 2.8, 13.2, 2.8)

    plt.tight_layout()
    out_flow = RESULTS_DIR / "preprocessing_pipeline_flowchart.png"
    plt.savefig(out_flow, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved preprocessing pipeline flowchart to {out_flow}", flush=True)


if __name__ == "__main__":
    generate_class_imbalance_plot()
    generate_preprocessing_pipeline_flowchart()
