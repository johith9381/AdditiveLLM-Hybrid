"""
Figure Generation Module for AdditiveLLM Paper Reproduction.
Reproduces Figures 5, 6, 7, 8, 9, 10, 11, and 12.
All figures are saved in reports/figures/ as PNG and SVG.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["None", "Keyhole", "Lack of Fusion", "Balling"]
COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

def save_fig(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    svg_path = output_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figures] Saved {name} to {png_path} and {svg_path}")

def plot_fig5_material_distribution(material_counts: Dict[str, Dict[str, int]], output_dir: Path):
    """
    Figure 5: Material distribution across datasets (Source, Augmented, Train, Test, Val).
    """
    df = pd.DataFrame(material_counts).fillna(0)
    # Sort by total
    df["Total"] = df.sum(axis=1)
    df = df.sort_values(by="Total", ascending=False).drop(columns=["Total"])
    
    top_mats = df.head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_mats.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_title("Figure 5: Material Distribution Across Datasets", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel("Material", fontsize=12)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, output_dir, "fig5_material_distribution")

def plot_fig6_defect_distribution(defect_counts: Dict[str, List[int]], output_dir: Path):
    """
    Figure 6: Defect class distribution across datasets.
    """
    df = pd.DataFrame(defect_counts, index=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, colormap="tab10", width=0.8)
    ax.set_title("Figure 6: Defect Class Distribution Across Datasets", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Positive Instances", fontsize=12)
    ax.set_xlabel("Defect Class", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    save_fig(fig, output_dir, "fig6_defect_distribution")

def plot_fig7_f1_scores(models_f1_baseline: dict, models_f1_prompt: dict, output_dir: Path):
    """
    Figure 7: Per-defect-class F1 scores with sample counts, 4 models x 2 formats (8 subplots).
    """
    model_keys = ["distilbert", "scibert", "t5", "llama"]
    display_names = ["DistilBERT", "SciBERT", "T5-Small", "Llama-3.2-1B"]
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharey=True)
    fig.suptitle("Figure 7: Per-Defect-Class F1 Scores (Baseline vs. Prompt)", fontsize=16, fontweight="bold", y=0.99)
    
    for row, (m_key, m_disp) in enumerate(zip(model_keys, display_names)):
        # Left: Baseline
        ax_base = axes[row, 0]
        base_f1 = [models_f1_baseline.get(m_key, {}).get(c, {}).get("f1_score", 0.0) for c in CLASS_NAMES]
        bars_b = ax_base.bar(CLASS_NAMES, base_f1, color=COLORS, alpha=0.85)
        ax_base.set_title(f"{m_disp} — Baseline Fine-Tuned", fontsize=12, fontweight="bold")
        ax_base.set_ylim(0, 1.05)
        ax_base.set_ylabel("F1 Score", fontsize=11)
        ax_base.grid(axis="y", linestyle="--", alpha=0.4)
        for b in bars_b:
            h = b.get_height()
            ax_base.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2., h),
                             ha='center', va='bottom', fontsize=9)
            
        # Right: Prompt
        ax_prompt = axes[row, 1]
        prompt_f1 = [models_f1_prompt.get(m_key, {}).get(c, {}).get("f1_score", 0.0) for c in CLASS_NAMES]
        bars_p = ax_prompt.bar(CLASS_NAMES, prompt_f1, color=COLORS, alpha=0.85)
        ax_prompt.set_title(f"{m_disp} — Prompt Fine-Tuned", fontsize=12, fontweight="bold")
        ax_prompt.set_ylim(0, 1.05)
        ax_prompt.grid(axis="y", linestyle="--", alpha=0.4)
        for b in bars_p:
            h = b.get_height()
            ax_prompt.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2., h),
                              ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_fig(fig, output_dir, "fig7_per_defect_f1_scores")

def plot_fig8_material_frequency(
    distilbert_prompt_f1: Dict[str, float],
    llama_baseline_f1: Dict[str, float],
    rare_mat_name: str,
    rare_mat_exists: bool,
    output_dir: Path
):
    """
    Figure 8: F1 score breakdown by material frequency:
    Common material (SS316L) vs. Rare material (IN615 per paper text).
    If IN615 does not exist verbatim, explicit annotation is displayed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 8: F1 Performance by Material Frequency (Common vs Rare)", fontsize=14, fontweight="bold")
    
    categories = ["SS316L (Common ~35%)", f"{rare_mat_name} (Rare ~5%)"]
    
    # Left: DistilBERT-Prompt
    ax1 = axes[0]
    vals1 = [distilbert_prompt_f1.get("SS316L", 0.0), distilbert_prompt_f1.get(rare_mat_name, 0.0)]
    bars1 = ax1.bar(categories, vals1, color=["#2980b9", "#e67e22"], width=0.5)
    ax1.set_title("DistilBERT (Prompt Fine-Tuned)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Macro F1 Score", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=10)
    if not rare_mat_exists:
        ax1.annotate(f"[!] '{rare_mat_name}' not in dataset", (1, 0.1), ha='center', color='red', fontsize=10, fontweight="bold")

    # Right: Llama-Baseline
    ax2 = axes[1]
    vals2 = [llama_baseline_f1.get("SS316L", 0.0), llama_baseline_f1.get(rare_mat_name, 0.0)]
    bars2 = ax2.bar(categories, vals2, color=["#2980b9", "#e67e22"], width=0.5)
    ax2.set_title("Llama-3.2-1B (Baseline Fine-Tuned)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2., h), ha='center', va='bottom', fontsize=10)
    if not rare_mat_exists:
        ax2.annotate(f"[!] '{rare_mat_name}' not in dataset", (1, 0.1), ha='center', color='red', fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, output_dir, "fig8_material_frequency_f1")

def plot_fig9_pca_baseline(pca_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: Path):
    """
    Figure 9: PCA of embeddings from models fine-tuned and evaluated on Baseline dataset.
    Colored by primary defect label.
    """
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Figure 9: PCA of Embeddings (Baseline-Tuned on Baseline Validation Set)", fontsize=15, fontweight="bold")
    axes = axes.ravel()
    
    models = ["distilbert", "scibert", "t5", "llama"]
    titles = ["DistilBERT", "SciBERT", "T5-Small", "Llama-3.2-1B"]
    
    for i, (m, title) in enumerate(zip(models, titles)):
        ax = axes[i]
        if m in pca_dict and pca_dict[m] is not None:
            coords, labels = pca_dict[m]  # coords: (N, 2), labels: (N,) primary label index
            for idx, (c_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
                mask = (labels == idx)
                if np.any(mask):
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=c_name, alpha=0.6, s=15, edgecolor="none")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("PC 1", fontsize=10)
            ax.set_ylabel("PC 2", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Weights Gated / Independent Checkpoint", ha='center', va='center', fontsize=11, color="gray")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in COLORS]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=11, frameon=True)
    plt.subplots_adjust(bottom=0.08)
    save_fig(fig, output_dir, "fig9_pca_baseline_embeddings")

def plot_fig10_pca_out_of_format(pca_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: Path):
    """
    Figure 10: PCA of embeddings for Baseline-fine-tuned models evaluated on Prompt dataset.
    """
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Figure 10: PCA of Embeddings (Baseline-Tuned Evaluated on Prompt Inputs)", fontsize=15, fontweight="bold")
    axes = axes.ravel()
    
    models = ["distilbert", "scibert", "t5", "llama"]
    titles = ["DistilBERT", "SciBERT", "T5-Small", "Llama-3.2-1B"]
    
    for i, (m, title) in enumerate(zip(models, titles)):
        ax = axes[i]
        if m in pca_dict and pca_dict[m] is not None:
            coords, labels = pca_dict[m]
            for idx, (c_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
                mask = (labels == idx)
                if np.any(mask):
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=c_name, alpha=0.6, s=15, edgecolor="none")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("PC 1", fontsize=10)
            ax.set_ylabel("PC 2", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Weights Gated / Independent Checkpoint", ha='center', va='center', fontsize=11, color="gray")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in COLORS]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=11, frameon=True)
    plt.subplots_adjust(bottom=0.08)
    save_fig(fig, output_dir, "fig10_pca_out_of_format_embeddings")

def plot_fig11_predicted_classes_baseline(predictions_dict: Dict[str, np.ndarray], output_dir: Path):
    """
    Figure 11: Distribution of predicted classes for Baseline models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {}
    for m, preds in predictions_dict.items():
        data[m] = preds.sum(axis=0)  # Count of positive predictions per class
        
    df = pd.DataFrame(data, index=CLASS_NAMES)
    df.plot(kind="bar", ax=ax, width=0.8, colormap="tab20")
    ax.set_title("Figure 11: Predicted Class Distribution (Baseline-Tuned Models)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Predicted Positive Cases", fontsize=11)
    ax.set_xlabel("Defect Class", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    save_fig(fig, output_dir, "fig11_predicted_classes_baseline")

def plot_fig12_predicted_classes_prompt(predictions_dict: Dict[str, np.ndarray], output_dir: Path):
    """
    Figure 12: Distribution of predicted classes for Prompt models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {}
    for m, preds in predictions_dict.items():
        data[m] = preds.sum(axis=0)
        
    df = pd.DataFrame(data, index=CLASS_NAMES)
    df.plot(kind="bar", ax=ax, width=0.8, colormap="tab20")
    ax.set_title("Figure 12: Predicted Class Distribution (Prompt-Tuned Models)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Predicted Positive Cases", fontsize=11)
    ax.set_xlabel("Defect Class", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    save_fig(fig, output_dir, "fig12_predicted_classes_prompt")
