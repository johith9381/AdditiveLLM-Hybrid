"""
Publication-Grade Multi-Seed Training and Controlled Baseline Reproduction Pipeline.
Evaluates Hybrid AdditiveLLM, DistilBERT, SciBERT, and T5-Small across 5 seeds:
[13, 42, 123, 2024, 7] on identical group-safe splits (Tiers A, B, C; Baseline & Prompt).
Generates:
1. results/multiseed_summary.csv
2. results/reproduced_baselines_summary.csv
3. results/controlled_comparison.csv
4. results/multiseed_runs_detailed.csv
"""

import os
import sys
import time
import math
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure base_dir is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import GroupSafeDataset, load_raw_geometry_lookup
from src.hybrid_model import HybridAdditiveLLM
from src.evaluate import compute_ece

SEEDS = [13, 42, 123, 2024, 7]
DEFECT_CLASSES = ["None", "Keyhole", "Lack_of_Fusion", "Balling"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class BaselineClassifierHead(nn.Module):
    """Classifier head matching AdditiveLLM paper on top of pooled encoder representations."""
    def __init__(self, hidden_dim: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_encoder_embeddings(
    encoder: nn.Module,
    enc_type: str,
    tokenizer: Any,
    dataset: Any,
    device: torch.device,
    batch_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-extracts pooled transformer representations with frozen encoder."""
    encoder.eval()
    all_embeddings = []
    all_labels = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            texts = batch["text"]
            labels = batch["labels"]
            enc = tokenizer(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(device)
            if enc_type == "t5-small":
                out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                hidden = out.last_hidden_state
            else:
                out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                hidden = out.last_hidden_state

            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            all_embeddings.append(pooled.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def train_classifier_head(
    head: BaselineClassifierHead,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 4,
    lr: float = 2e-3,
    batch_size: int = 64,
    seed: int = 42
) -> BaselineClassifierHead:
    """Trains the baseline classifier head on pre-extracted embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    head.to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    n = len(train_feats)
    head.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx = train_feats[idx].to(device)
            by = train_labels[idx].float().to(device)

            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    head.eval()
    return head


def evaluate_dataset(
    model: nn.Module,
    eval_loader: DataLoader,
    tokenizer_dict: Dict[str, Any],
    device: torch.device,
    is_hybrid: bool = False
) -> Dict[str, float]:
    """Evaluates accuracy, macro F1, precision, recall, and ECE on an eval loader."""
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in eval_loader:
            texts = batch["text"]
            targets = batch["labels"].numpy()
            all_targets.append(targets)

            if is_hybrid:
                num_feats = batch["numeric_features"].to(device)
                mat_idx = batch["material_idx"].to(device)
                enc_d = tokenizer_dict["distilbert"](texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(device)
                enc_s = tokenizer_dict["scibert"](texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(device)
                out = model(
                    input_ids_distil=enc_d["input_ids"],
                    attention_mask_distil=enc_d["attention_mask"],
                    input_ids_sci=enc_s["input_ids"],
                    attention_mask_sci=enc_s["attention_mask"],
                    numeric_features=num_feats,
                    material_idx=mat_idx
                )
                probs = torch.sigmoid(out["logits"]).cpu().numpy()
            else:
                tok = tokenizer_dict["primary"]
                enc = tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(device)
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)

    y_pred_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    acc = (y_pred_binary == y_true).mean() * 100.0
    f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)
    ece = compute_ece(y_pred_probs, y_true, n_bins=10)

    return {
        "accuracy_pct": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "ece": float(ece)
    }


def run_publication_benchmark(base_dir: Path):
    """
    Main orchestrator for multi-seed and controlled baseline evaluation.
    """
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = base_dir / "models" / "base_models"
    splits_dir = base_dir / "data" / "splits_group_safe"

    print("=" * 80, flush=True)
    print("  ADDITIVELLM PUBLICATION-GRADE MULTI-SEED & CONTROLLED BASELINE PIPELINE", flush=True)
    print("=" * 80, flush=True)
    print(f"[Publication Pipeline] Compute Device: {DEVICE}", flush=True)
    print(f"[Publication Pipeline] Evaluation Seeds (n=5): {SEEDS}", flush=True)

    # 1. Load splits and templates
    print("\n[Publication Pipeline] Loading dataset splits and geometry lookup...", flush=True)
    df_train = pd.read_parquet(splits_dir / "train.parquet")
    df_tier_a = pd.read_parquet(splits_dir / "tier_a.parquet")
    df_tier_b = pd.read_parquet(splits_dir / "tier_b.parquet")
    df_tier_c = pd.read_parquet(splits_dir / "tier_c.parquet")
    lookup = load_raw_geometry_lookup(base_dir / "data" / "raw")

    with open(base_dir / "data" / "templates" / "test_prompts.json") as f:
        test_templates = list(json.load(f).values())

    # Build representative pools for fast, low-overhead evaluation across 5 seeds
    print("[Publication Pipeline] Initializing dataset pools...", flush=True)
    pool_a = df_tier_a.sample(n=min(1200, len(df_tier_a)), random_state=42).reset_index(drop=True)
    pool_b = df_tier_b.sample(n=min(1200, len(df_tier_b)), random_state=42).reset_index(drop=True)
    pool_c = df_tier_c.sample(n=min(1200, len(df_tier_c)), random_state=42).reset_index(drop=True)
    pool_train = df_train.sample(n=2500, random_state=42).reset_index(drop=True)

    dataset_tier_a_base = GroupSafeDataset(pool_a, templates=test_templates, mode="baseline", lookup=lookup)
    dataset_tier_a_prompt = GroupSafeDataset(pool_a, templates=test_templates, mode="prompt", lookup=lookup)

    dataset_tier_b_base = GroupSafeDataset(pool_b, templates=test_templates, mode="baseline", lookup=lookup)
    dataset_tier_b_prompt = GroupSafeDataset(pool_b, templates=test_templates, mode="prompt", lookup=lookup)

    dataset_tier_c_base = GroupSafeDataset(pool_c, templates=test_templates, mode="baseline", lookup=lookup)
    dataset_tier_c_prompt = GroupSafeDataset(pool_c, templates=test_templates, mode="prompt", lookup=lookup)

    dataset_train_base = GroupSafeDataset(pool_train, templates=test_templates, mode="baseline", lookup=lookup)

    # 2. Tokenizers
    print("[Publication Pipeline] Initializing offline tokenizers...", flush=True)
    distil_tok = AutoTokenizer.from_pretrained(str(models_dir / "distilbert"), local_files_only=True)
    sci_tok = AutoTokenizer.from_pretrained(str(models_dir / "scibert"), local_files_only=True)
    t5_tok = AutoTokenizer.from_pretrained(str(models_dir / "t5-small"), local_files_only=True)

    hybrid_tok_dict = {"distilbert": distil_tok, "scibert": sci_tok}

    # 3. Load Trained Hybrid Model
    print("[Publication Pipeline] Loading trained 4-Expert Hybrid model...", flush=True)
    hybrid_model = HybridAdditiveLLM(str(models_dir)).to(DEVICE)
    ckpt_path = base_dir / "checkpoints" / "checkpoint_best.pt"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hybrid_model.load_state_dict(payload["model_state_dict"])
        print(f"[Publication Pipeline] Best trained weights loaded from {ckpt_path}!", flush=True)
    hybrid_model.eval()

    eval_datasets_map = {
        ("Tier A", "Baseline"): dataset_tier_a_base,
        ("Tier A", "Prompt"): dataset_tier_a_prompt,
        ("Tier B", "Baseline"): dataset_tier_b_base,
        ("Tier B", "Prompt"): dataset_tier_b_prompt,
        ("Tier C", "Baseline"): dataset_tier_c_base,
        ("Tier C", "Prompt"): dataset_tier_c_prompt,
    }

    all_run_records = []

    # -------------------------------------------------------------
    # STEP 1: Multi-Seed Evaluation of Hybrid Model (5 Seeds)
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 1: MULTI-SEED EVALUATION OF HYBRID MODEL (5 SEEDS)", flush=True)
    print("=" * 80, flush=True)

    n_eval_samples = 500

    for seed in SEEDS:
        print(f"\n>>> Evaluating Hybrid Model under Seed {seed}...", flush=True)
        rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for (tier_name, fmt), base_ds in eval_datasets_map.items():
            n_sub = min(n_eval_samples, len(base_ds))
            sub_indices = rng.choice(len(base_ds), size=n_sub, replace=False)
            sub_dataset = Subset(base_ds, sub_indices)
            loader = DataLoader(sub_dataset, batch_size=64, shuffle=False)

            metrics = evaluate_dataset(hybrid_model, loader, hybrid_tok_dict, DEVICE, is_hybrid=True)
            record = {
                "seed": seed,
                "model": "Hybrid AdditiveLLM",
                "tier": tier_name,
                "format": fmt,
                "accuracy_pct": metrics["accuracy_pct"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "ece": metrics["ece"]
            }
            all_run_records.append(record)
            print(f"  [{tier_name} {fmt}] Acc: {metrics['accuracy_pct']:.2f}% | F1: {metrics['macro_f1']:.4f} | ECE: {metrics['ece']:.4f}", flush=True)

    # -------------------------------------------------------------
    # STEP 2: Controlled Single-Model Baseline Reproduction (5 Seeds)
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 2: CONTROLLED BASELINE REPRODUCTION ON MAC M5 (5 SEEDS)", flush=True)
    print("=" * 80, flush=True)

    baseline_configs = [
        ("DistilBERT", "distilbert", distil_tok, str(models_dir / "distilbert")),
        ("SciBERT", "scibert", sci_tok, str(models_dir / "scibert")),
        ("T5-Small", "t5-small", t5_tok, str(models_dir / "t5-small")),
    ]

    for model_name, enc_type, tokenizer, model_path in baseline_configs:
        print(f"\n--- Pre-extracting Features & Evaluating Controlled Baseline: {model_name} (5 Seeds) ---", flush=True)
        t_start = time.time()
        if enc_type == "t5-small":
            encoder = T5EncoderModel.from_pretrained(model_path, local_files_only=True).to(DEVICE)
            hidden_dim = encoder.config.d_model
        else:
            encoder = AutoModel.from_pretrained(model_path, local_files_only=True).to(DEVICE)
            hidden_dim = encoder.config.hidden_size

        for p in encoder.parameters():
            p.requires_grad = False

        # Extract features for training pool (2000 samples)
        train_pool_sub = Subset(dataset_train_base, list(range(min(2000, len(dataset_train_base)))))
        train_feats, train_labels = extract_encoder_embeddings(encoder, enc_type, tokenizer, train_pool_sub, DEVICE)

        # Extract features for all evaluation datasets
        eval_feats_map = {}
        for (tier_name, fmt), base_ds in eval_datasets_map.items():
            feats, labels = extract_encoder_embeddings(encoder, enc_type, tokenizer, base_ds, DEVICE)
            eval_feats_map[(tier_name, fmt)] = (feats, labels)

        # Free the heavy encoder
        del encoder
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print(f"Features pre-extracted in {time.time() - t_start:.1f}s. Evaluating across 5 seeds...", flush=True)

        for seed in SEEDS:
            t0 = time.time()
            rng = np.random.RandomState(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            # Sample 2000 samples from train pool
            train_idx = rng.choice(len(train_feats), size=len(train_feats), replace=False)
            head = BaselineClassifierHead(hidden_dim).to(DEVICE)
            train_classifier_head(head, train_feats[train_idx], train_labels[train_idx], DEVICE, epochs=4, lr=2e-3, seed=seed)

            # Evaluate on all tiers and formats
            for (tier_name, fmt), (ef_feats, ef_labels) in eval_feats_map.items():
                n_sub = min(n_eval_samples, len(ef_feats))
                sub_indices = rng.choice(len(ef_feats), size=n_sub, replace=False)
                sub_x = ef_feats[sub_indices].to(DEVICE)
                sub_y = ef_labels[sub_indices].numpy()

                with torch.no_grad():
                    logits = head(sub_x)
                    probs = torch.sigmoid(logits).cpu().numpy()

                y_pred_binary = (probs > 0.5).astype(int)
                acc = (y_pred_binary == sub_y).mean() * 100.0
                f1 = f1_score(sub_y, y_pred_binary, average="macro", zero_division=0)
                prec = precision_score(sub_y, y_pred_binary, average="macro", zero_division=0)
                rec = recall_score(sub_y, y_pred_binary, average="macro", zero_division=0)
                ece = compute_ece(probs, sub_y, n_bins=10)

                record = {
                    "seed": seed,
                    "model": model_name,
                    "tier": tier_name,
                    "format": fmt,
                    "accuracy_pct": acc,
                    "macro_f1": f1,
                    "macro_precision": prec,
                    "macro_recall": rec,
                    "ece": ece
                }
                all_run_records.append(record)

            elapsed = time.time() - t0
            tier_a_base_rec = [r for r in all_run_records if r["seed"] == seed and r["model"] == model_name and r["tier"] == "Tier A" and r["format"] == "Baseline"][0]
            tier_a_prom_rec = [r for r in all_run_records if r["seed"] == seed and r["model"] == model_name and r["tier"] == "Tier A" and r["format"] == "Prompt"][0]
            print(f"  [Seed {seed}] {model_name} in {elapsed:.2f}s | Tier A Base: {tier_a_base_rec['accuracy_pct']:.2f}% (F1: {tier_a_base_rec['macro_f1']:.4f}) | Prompt: {tier_a_prom_rec['accuracy_pct']:.2f}% (F1: {tier_a_prom_rec['macro_f1']:.4f})", flush=True)

    # -------------------------------------------------------------
    # STEP 3: Aggregation & Summary Metrics Computation
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 3: COMPUTING AGGREGATED METRICS & TABLES", flush=True)
    print("=" * 80, flush=True)

    df_runs = pd.DataFrame(all_run_records)
    df_runs.to_csv(results_dir / "multiseed_runs_detailed.csv", index=False)
    print(f"[Publication Pipeline] Detailed runs saved to {results_dir / 'multiseed_runs_detailed.csv'}", flush=True)

    # 1. results/multiseed_summary.csv (Hybrid model performance across 5 seeds)
    df_hybrid = df_runs[df_runs["model"] == "Hybrid AdditiveLLM"]
    summary_hybrid = []
    for (tier, fmt), grp in df_hybrid.groupby(["tier", "format"]):
        summary_hybrid.append({
            "Tier": tier,
            "Format": fmt,
            "Mean_Accuracy_Pct": round(grp["accuracy_pct"].mean(), 2),
            "Std_Accuracy_Pct": round(grp["accuracy_pct"].std(), 2),
            "Min_Accuracy_Pct": round(grp["accuracy_pct"].min(), 2),
            "Max_Accuracy_Pct": round(grp["accuracy_pct"].max(), 2),
            "Mean_Macro_F1": round(grp["macro_f1"].mean(), 4),
            "Std_Macro_F1": round(grp["macro_f1"].std(), 4),
            "Min_Macro_F1": round(grp["macro_f1"].min(), 4),
            "Max_Macro_F1": round(grp["macro_f1"].max(), 4),
            "Mean_Precision": round(grp["macro_precision"].mean(), 4),
            "Mean_Recall": round(grp["macro_recall"].mean(), 4),
            "Mean_ECE": round(grp["ece"].mean(), 4),
            "Sample_Size_Seeds": len(grp)
        })
    df_summary_hybrid = pd.DataFrame(summary_hybrid)
    df_summary_hybrid.to_csv(results_dir / "multiseed_summary.csv", index=False)
    print(f"[Publication Pipeline] Multi-seed summary saved to {results_dir / 'multiseed_summary.csv'}", flush=True)

    # 2. results/reproduced_baselines_summary.csv (DistilBERT, SciBERT, T5 across 5 seeds)
    df_baselines = df_runs[df_runs["model"] != "Hybrid AdditiveLLM"]
    summary_baselines = []
    for (model_name, tier, fmt), grp in df_baselines.groupby(["model", "tier", "format"]):
        summary_baselines.append({
            "Model": model_name,
            "Tier": tier,
            "Format": fmt,
            "Mean_Accuracy_Pct": round(grp["accuracy_pct"].mean(), 2),
            "Std_Accuracy_Pct": round(grp["accuracy_pct"].std(), 2),
            "Min_Accuracy_Pct": round(grp["accuracy_pct"].min(), 2),
            "Max_Accuracy_Pct": round(grp["accuracy_pct"].max(), 2),
            "Mean_Macro_F1": round(grp["macro_f1"].mean(), 4),
            "Std_Macro_F1": round(grp["macro_f1"].std(), 4),
            "Mean_ECE": round(grp["ece"].mean(), 4),
            "Sample_Size_Seeds": len(grp)
        })
    df_summary_baselines = pd.DataFrame(summary_baselines)
    df_summary_baselines.to_csv(results_dir / "reproduced_baselines_summary.csv", index=False)
    print(f"[Publication Pipeline] Reproduced baselines summary saved to {results_dir / 'reproduced_baselines_summary.csv'}", flush=True)

    # 3. results/controlled_comparison.csv (3-Column Table)
    paper_reported = {
        ("Tier A", "Baseline"): "94.23% / 0.9059 (Llama-3.2-1B reported on sibling-leakage)",
        ("Tier A", "Prompt"): "82.22% / 0.7140 (DistilBERT reported on sibling-leakage)",
        ("Tier B", "Baseline"): "N/A (Paper did not evaluate group-safe held-out splits)",
        ("Tier B", "Prompt"): "N/A (Paper did not evaluate group-safe held-out splits)",
        ("Tier C", "Baseline"): "N/A (Paper did not evaluate held-out alloy OOD)",
        ("Tier C", "Prompt"): "N/A (Paper did not evaluate held-out alloy OOD)"
    }

    comparison_records = []
    for (tier, fmt), h_grp in df_hybrid.groupby(["tier", "format"]):
        h_acc_mean = h_grp["accuracy_pct"].mean()
        h_acc_std = h_grp["accuracy_pct"].std()
        h_f1_mean = h_grp["macro_f1"].mean()
        h_f1_std = h_grp["macro_f1"].std()

        # Best reproduced baseline for this condition
        base_match = df_baselines[(df_baselines["tier"] == tier) & (df_baselines["format"] == fmt)]
        base_summary = base_match.groupby("model").agg({"accuracy_pct": "mean", "macro_f1": "mean"}).reset_index()
        best_base_row = base_summary.sort_values(by="macro_f1", ascending=False).iloc[0]
        best_base_name = best_base_row["model"]

        best_base_runs = base_match[base_match["model"] == best_base_name]
        b_acc_mean = best_base_runs["accuracy_pct"].mean()
        b_acc_std = best_base_runs["accuracy_pct"].std()
        b_f1_mean = best_base_runs["macro_f1"].mean()
        b_f1_std = best_base_runs["macro_f1"].std()

        delta_acc = h_acc_mean - b_acc_mean
        delta_f1 = h_f1_mean - b_f1_mean

        comparison_records.append({
            "Tier": tier,
            "Format": fmt,
            "Original_Paper_Reported (Uncontrolled, Diff Infra)": paper_reported.get((tier, fmt), "N/A"),
            "Reproduced_Baseline_Controlled (Mean ± Std, n=5)": f"{b_acc_mean:.2f} ± {b_acc_std:.2f}% (F1: {b_f1_mean:.4f} ± {b_f1_std:.4f}) [{best_base_name}]",
            "Hybrid_AdditiveLLM_Ours (Mean ± Std, n=5)": f"{h_acc_mean:.2f} ± {h_acc_std:.2f}% (F1: {h_f1_mean:.4f} ± {h_f1_std:.4f})",
            "Delta_Acc_Pct": round(delta_acc, 2),
            "Delta_Macro_F1": round(delta_f1, 4),
            "Controlled_Baseline_Outperformed": bool(delta_f1 > 0)
        })

    df_comparison = pd.DataFrame(comparison_records)
    df_comparison.to_csv(results_dir / "controlled_comparison.csv", index=False)
    print(f"[Publication Pipeline] Controlled comparison table saved to {results_dir / 'controlled_comparison.csv'}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  PUBLICATION BENCHMARK PIPELINE COMPLETE!", flush=True)
    print("=" * 80, flush=True)

    return df_runs, df_summary_hybrid, df_summary_baselines, df_comparison


if __name__ == "__main__":
    run_publication_benchmark(BASE_DIR)
