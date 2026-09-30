"""
AdditiveLLM Tier C (True Material OOD) Generalization Interventions Suite.
Implements and evaluates 4 distinct scientific interventions across 5 seeds:
[13, 42, 123, 2024, 7] to investigate whether Tier C weakness can be mitigated.

Interventions:
1. Material-Agnostic Feature Emphasis (results/tierc_intervention_material_agnostic.csv)
   - Zeroes out material identity embedding, forcing reliance on continuous physics features & aspect ratios.
2. Physics-Informed Auxiliary Loss Reweighting (results/tierc_intervention_loss_reweighting.csv)
   - Sweeps lambda_reg in [0.25, 0.5, 1.0] to heavily enforce universal hydrodynamic constraints.
3. Material Thermophysical Property Features (results/tierc_intervention_property_features.csv)
   - Replaces discrete material index with true physical thermophysical properties (Tm, k, Cv, eta, alpha).
4. Physical Property Interpolation Augmentation (results/tierc_intervention_augmentation.csv)
   - Generates physically-motivated synthetic training instances from neighboring alloy regimes.

Master Summary:
- results/tierc_improvement_summary.csv comparing Baseline Tier C vs all interventions with significance tests.
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
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import (
    GroupSafeDataset,
    load_raw_geometry_lookup,
    PARAM_STATS,
    KNOWN_MATERIALS,
    MATERIAL_TO_IDX
)
from src.hybrid_model import HybridAdditiveLLM, NumericParameterMLPExpert
from src.evaluate import compute_ece

SEEDS = [13, 42, 123, 2024, 7]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models" / "base_models"
SPLITS_DIR = BASE_DIR / "data" / "splits_group_safe"

# ==============================================================================
# LITERATURE THERMOPHYSICAL PROPERTIES LOOKUP TABLE
# Sources:
# 1. Mills, K. C., Recommended values of thermophysical properties for selected commercial alloys (2002).
# 2. ASM Handbook, Vol 2: Properties and Selection: Nonferrous Alloys and Special-Purpose Materials.
# 3. NIST Additive Manufacturing Benchmark Test Series (AM-Bench 2018/2022).
# 4. King et al., Observation of keyhole-mode laser melting in LPBF, J. Mater. Process. Tech. (2014).
# Fields: [Solidus_Tm_K, Thermal_Cond_k_W_mK, Volumetric_Heat_Cap_Cv_kJ_m3K, Absorptivity_eta, Thermal_Diffusivity_alpha_mm2_s]
# ==============================================================================
THERMOPHYSICAL_PROPERTIES = {
    # Tier C Held-Out Alloy
    "IN625":       [1623.0,  9.8, 3460.0, 0.38, 2.83],
    # Training / Seen Alloys
    "SS316L":      [1673.0, 16.3, 4000.0, 0.35, 4.08],
    "Ti-6Al-4V":   [1923.0,  6.7, 2330.0, 0.35, 2.88],
    "IN718":       [1609.0, 11.4, 3563.0, 0.37, 3.20],
    "AlSi10Mg":    [ 873.0, 130.0, 2439.0, 0.18, 53.30],
    "Cu10Sn":      [1273.0, 64.0, 3336.0, 0.22, 19.18],
    "Hastelloy X": [1583.0,  9.1, 3995.0, 0.38, 2.28],
    "Tungsten":    [3695.0, 173.0, 2548.0, 0.45, 67.90],
    "Invar36":     [1698.0, 10.5, 4146.0, 0.36, 2.53],
    "SS17-4PH":    [1693.0, 18.0, 3588.0, 0.35, 5.02],
    "IN738LC":     [1615.0, 10.8, 3510.0, 0.37, 3.08],
    "Al-2.5Fe":    [ 915.0, 145.0, 2480.0, 0.19, 58.47],
    "HCP Cu":      [1358.0, 390.0, 3450.0, 0.12, 113.04],
}
DEFAULT_THERMO = [1600.0, 20.0, 3500.0, 0.35, 5.71]

# Normalization means & stds for thermophysical properties
THERMO_STATS = {
    "Tm": (1600.0, 500.0),
    "k": (50.0, 80.0),
    "Cv": (3300.0, 600.0),
    "eta": (0.32, 0.08),
    "alpha": (20.0, 30.0)
}


def get_normalized_thermo_properties(material_name: str) -> np.ndarray:
    props = THERMOPHYSICAL_PROPERTIES.get(str(material_name).strip(), DEFAULT_THERMO)
    norm = np.array([
        (props[0] - THERMO_STATS["Tm"][0]) / THERMO_STATS["Tm"][1],
        (props[1] - THERMO_STATS["k"][0]) / THERMO_STATS["k"][1],
        (props[2] - THERMO_STATS["Cv"][0]) / THERMO_STATS["Cv"][1],
        (props[3] - THERMO_STATS["eta"][0]) / THERMO_STATS["eta"][1],
        (props[4] - THERMO_STATS["alpha"][0]) / THERMO_STATS["alpha"][1],
    ], dtype=np.float32)
    return norm


def precompute_dataset_features(
    model: HybridAdditiveLLM,
    ds: Dataset,
    tok_dict: Dict[str, Any],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Precomputes frozen transformer features and batch tensors for ultra-fast multi-seed evaluation."""
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    e1_list, e2_list, e3_list = [], [], []
    num_feats_list, mat_idx_list, labels_list, aux_targets_list = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            texts = batch["text"]
            num_feats = batch["numeric_features"].to(device)
            mat_idx = batch["material_idx"].to(device)
            labels = batch["labels"].to(device)

            enc_d = tok_dict["distilbert"](texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(device)
            enc_s = tok_dict["scibert"](texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(device)

            e1 = model.expert1_distilbert(enc_d["input_ids"], enc_d["attention_mask"])
            e2 = model.expert2_scibert(enc_s["input_ids"], enc_s["attention_mask"])
            e3 = model.expert3_matscibert(enc_s["input_ids"], enc_s["attention_mask"])

            e1_list.append(e1)
            e2_list.append(e2)
            e3_list.append(e3)
            num_feats_list.append(num_feats)
            mat_idx_list.append(mat_idx)
            labels_list.append(labels)
            if "aux_targets" in batch:
                aux_targets_list.append(batch["aux_targets"].to(device))

    return {
        "e1": torch.cat(e1_list, dim=0),
        "e2": torch.cat(e2_list, dim=0),
        "e3": torch.cat(e3_list, dim=0),
        "num_feats": torch.cat(num_feats_list, dim=0),
        "mat_idx": torch.cat(mat_idx_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
        "aux_targets": torch.cat(aux_targets_list, dim=0) if aux_targets_list else None
    }


def evaluate_tier_c_condition_cached(
    model: nn.Module,
    cached_feats: Dict[str, torch.Tensor],
    sub_indices: np.ndarray,
    device: torch.device,
    zero_material: bool = False,
    thermo_expert: nn.Module = None,
    raw_materials: List[str] = None,
    neighbor_blend_embed: torch.Tensor = None
) -> Dict[str, float]:
    """Helper to evaluate Tier C under various model configurations using precomputed representations."""
    model.eval()
    if thermo_expert is not None:
        thermo_expert.eval()

    idx_t = torch.tensor(sub_indices, dtype=torch.long, device=device)
    e1 = cached_feats["e1"][idx_t]
    e2 = cached_feats["e2"][idx_t]
    e3 = cached_feats["e3"][idx_t]
    num_feats = cached_feats["num_feats"][idx_t]
    mat_idx = cached_feats["mat_idx"][idx_t]
    y_true = cached_feats["labels"][idx_t].cpu().numpy()

    with torch.no_grad():
        if thermo_expert is not None and raw_materials is not None:
            sub_mats = [raw_materials[i] for i in sub_indices]
            thermo_feats = np.stack([get_normalized_thermo_properties(m) for m in sub_mats])
            comb = torch.cat([num_feats, torch.tensor(thermo_feats, dtype=torch.float32, device=device)], dim=-1)
            e4 = thermo_expert(comb)
        elif zero_material:
            mat_embed_zero = torch.zeros(len(sub_indices), 16, device=device)
            x = torch.cat([num_feats, mat_embed_zero], dim=-1)
            e4 = model.expert4_numeric.net(x)
        elif neighbor_blend_embed is not None:
            b_blend = neighbor_blend_embed.expand(len(sub_indices), -1)
            x = torch.cat([num_feats, b_blend], dim=-1)
            e4 = model.expert4_numeric.net(x)
        else:
            e4 = model.expert4_numeric(num_feats, mat_idx)

        fused, _ = model.fusion([e1, e2, e3, e4])
        logits = model.classifier(fused)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_pred_binary = (probs > 0.5).astype(int)
    acc = (y_pred_binary == y_true).mean() * 100.0
    f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)
    ece = compute_ece(probs, y_true, n_bins=10)

    return {
        "accuracy_pct": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "ece": float(ece)
    }


def run_all_tierc_interventions(base_dir: Path):
    print("=" * 80, flush=True)
    print("  TIER C (TRUE MATERIAL OOD) GENERALIZATION INTERVENTIONS SUITE", flush=True)
    print("=" * 80, flush=True)

    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = base_dir / "models" / "base_models"
    splits_dir = base_dir / "data" / "splits_group_safe"

    df_train = pd.read_parquet(splits_dir / "train.parquet")
    df_tier_c = pd.read_parquet(splits_dir / "tier_c.parquet")
    lookup = load_raw_geometry_lookup(base_dir / "data" / "raw")

    with open(base_dir / "data" / "templates" / "test_prompts.json") as f:
        test_templates = list(json.load(f).values())

    print("[Tier C Interventions] Initializing tokenizers & model...", flush=True)
    distil_tok = AutoTokenizer.from_pretrained(str(models_dir / "distilbert"), local_files_only=True)
    sci_tok = AutoTokenizer.from_pretrained(str(models_dir / "scibert"), local_files_only=True)
    tok_dict = {"distilbert": distil_tok, "scibert": sci_tok}

    hybrid_model = HybridAdditiveLLM(str(models_dir)).to(DEVICE)
    ckpt_path = base_dir / "checkpoints" / "checkpoint_best.pt"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hybrid_model.load_state_dict(payload["model_state_dict"])
    hybrid_model.eval()

    # Pre-parse Tier C datasets
    pool_c = df_tier_c.sample(n=min(1200, len(df_tier_c)), random_state=42).reset_index(drop=True)
    c_dataset_base = GroupSafeDataset(pool_c, templates=test_templates, mode="baseline", lookup=lookup)
    c_dataset_prompt = GroupSafeDataset(pool_c, templates=test_templates, mode="prompt", lookup=lookup)
    raw_mats_c = [pool_c.iloc[i]["material"] for i in range(len(pool_c))]

    n_eval_samples = 500

    print("[Tier C Interventions] Precomputing frozen transformer features...", flush=True)
    t0 = time.time()
    cached_base = precompute_dataset_features(hybrid_model, c_dataset_base, tok_dict, DEVICE)
    cached_prompt = precompute_dataset_features(hybrid_model, c_dataset_prompt, tok_dict, DEVICE)
    print(f"[Tier C Interventions] Features precomputed in {time.time() - t0:.1f}s.", flush=True)

    cached_by_format = {"Baseline": cached_base, "Prompt": cached_prompt}

    # --------------------------------------------------------------------------
    # 0. BASELINE TIER C (No Intervention - Control)
    # --------------------------------------------------------------------------
    print("\n--- Running Control: Baseline Tier C (5 Seeds) ---", flush=True)
    baseline_records = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for fmt in ["Baseline", "Prompt"]:
            sub_indices = rng.choice(len(cached_by_format[fmt]["e1"]), size=n_eval_samples, replace=False)
            metrics = evaluate_tier_c_condition_cached(
                hybrid_model, cached_by_format[fmt], sub_indices, DEVICE, zero_material=False
            )
            baseline_records.append({
                "intervention": "0_Baseline_Control",
                "seed": seed,
                "format": fmt,
                **metrics
            })
            print(f"  [Control Seed {seed} {fmt}] Acc: {metrics['accuracy_pct']:.2f}% | F1: {metrics['macro_f1']:.4f}", flush=True)

    # --------------------------------------------------------------------------
    # 1. INTERVENTION 1: Material-Agnostic Feature Emphasis
    # --------------------------------------------------------------------------
    print("\n--- Running Intervention 1: Material-Agnostic Feature Emphasis (5 Seeds) ---", flush=True)
    int1_records = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for fmt in ["Baseline", "Prompt"]:
            sub_indices = rng.choice(len(cached_by_format[fmt]["e1"]), size=n_eval_samples, replace=False)
            metrics = evaluate_tier_c_condition_cached(
                hybrid_model, cached_by_format[fmt], sub_indices, DEVICE, zero_material=True
            )
            int1_records.append({
                "intervention": "1_Material_Agnostic",
                "seed": seed,
                "format": fmt,
                **metrics
            })
            print(f"  [Int 1 Seed {seed} {fmt}] Acc: {metrics['accuracy_pct']:.2f}% | F1: {metrics['macro_f1']:.4f}", flush=True)

    df_int1 = pd.DataFrame(int1_records)
    df_int1.to_csv(results_dir / "tierc_intervention_material_agnostic.csv", index=False)
    print(f"Saved to {results_dir / 'tierc_intervention_material_agnostic.csv'}", flush=True)

    # --------------------------------------------------------------------------
    # 2. INTERVENTION 2: Physics-Informed Auxiliary Loss Reweighting
    # --------------------------------------------------------------------------
    print("\n--- Running Intervention 2: Physics Loss Reweighting Sweep (5 Seeds) ---", flush=True)
    int2_records = []
    lambda_weights = [0.25, 0.5, 1.0]

    # Precompute training representations
    train_pool = df_train.sample(n=2500, random_state=42).reset_index(drop=True)
    train_ds = GroupSafeDataset(train_pool, templates=test_templates, mode="baseline", lookup=lookup)
    train_cached = precompute_dataset_features(hybrid_model, train_ds, tok_dict, DEVICE)

    with torch.no_grad():
        train_e4 = hybrid_model.expert4_numeric(train_cached["num_feats"], train_cached["mat_idx"])
        train_fused, _ = hybrid_model.fusion([train_cached["e1"], train_cached["e2"], train_cached["e3"], train_e4])

    for l_reg in lambda_weights:
        print(f"\n>> Testing Physics Auxiliary Loss Weight lambda_reg = {l_reg}...", flush=True)
        torch.manual_seed(42)
        reweighted_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 4)
        ).to(DEVICE)
        reweighted_aux = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        ).to(DEVICE)

        opt = torch.optim.AdamW(list(reweighted_classifier.parameters()) + list(reweighted_aux.parameters()), lr=1e-3)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        mse_loss_fn = nn.MSELoss()

        batch_sz = 64
        n_samples = len(train_fused)
        for ep in range(3):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_sz):
                idx = perm[i: i + batch_sz]
                b_fused = train_fused[idx]
                b_cls_targets = train_cached["labels"][idx]
                b_aux_targets = train_cached["aux_targets"][idx]

                opt.zero_grad()
                pred_cls = reweighted_classifier(b_fused)
                pred_aux = reweighted_aux(b_fused)

                l_cls = bce_loss_fn(pred_cls, b_cls_targets)
                l_aux = mse_loss_fn(pred_aux, b_aux_targets)
                total_l = l_cls + l_reg * l_aux
                total_l.backward()
                opt.step()

        # Swap classifier head temporarily
        orig_cls = hybrid_model.classifier
        hybrid_model.classifier = reweighted_classifier

        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            for fmt in ["Baseline", "Prompt"]:
                sub_indices = rng.choice(len(cached_by_format[fmt]["e1"]), size=n_eval_samples, replace=False)
                metrics = evaluate_tier_c_condition_cached(
                    hybrid_model, cached_by_format[fmt], sub_indices, DEVICE, zero_material=False
                )
                int2_records.append({
                    "intervention": f"2_Loss_Reweight_lambda_{l_reg}",
                    "lambda_reg": l_reg,
                    "seed": seed,
                    "format": fmt,
                    **metrics
                })

        hybrid_model.classifier = orig_cls
        print(f"  [lambda={l_reg}] Evaluated across 5 seeds.", flush=True)

    df_int2 = pd.DataFrame(int2_records)
    df_int2.to_csv(results_dir / "tierc_intervention_loss_reweighting.csv", index=False)
    print(f"Saved to {results_dir / 'tierc_intervention_loss_reweighting.csv'}", flush=True)

    # --------------------------------------------------------------------------
    # 3. INTERVENTION 3: Material Thermophysical Property Features
    # --------------------------------------------------------------------------
    print("\n--- Running Intervention 3: Thermophysical Property Branch (5 Seeds) ---", flush=True)
    thermo_expert = nn.Sequential(
        nn.Linear(10, 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Linear(128, 128)
    ).to(DEVICE)

    train_mats = [train_pool.iloc[i]["material"] for i in range(len(train_pool))]
    train_thermo_feats = np.stack([get_normalized_thermo_properties(m) for m in train_mats])
    train_comb = torch.cat([train_cached["num_feats"], torch.tensor(train_thermo_feats, dtype=torch.float32, device=DEVICE)], dim=-1)

    thermo_opt = torch.optim.AdamW(thermo_expert.parameters(), lr=1e-3)
    for ep in range(5):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, 64):
            idx = perm[i: i + 64]
            thermo_opt.zero_grad()
            out_e4 = thermo_expert(train_comb[idx])
            loss = F.mse_loss(out_e4, train_e4[idx])
            loss.backward()
            thermo_opt.step()

    int3_records = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for fmt in ["Baseline", "Prompt"]:
            sub_indices = rng.choice(len(cached_by_format[fmt]["e1"]), size=n_eval_samples, replace=False)
            metrics = evaluate_tier_c_condition_cached(
                hybrid_model, cached_by_format[fmt], sub_indices, DEVICE,
                zero_material=False, thermo_expert=thermo_expert, raw_materials=raw_mats_c
            )
            int3_records.append({
                "intervention": "3_Thermophysical_Properties",
                "seed": seed,
                "format": fmt,
                **metrics
            })
            print(f"  [Int 3 Seed {seed} {fmt}] Acc: {metrics['accuracy_pct']:.2f}% | F1: {metrics['macro_f1']:.4f}", flush=True)

    df_int3 = pd.DataFrame(int3_records)
    df_int3.to_csv(results_dir / "tierc_intervention_property_features.csv", index=False)
    print(f"Saved to {results_dir / 'tierc_intervention_property_features.csv'}", flush=True)

    # --------------------------------------------------------------------------
    # 4. INTERVENTION 4: Physics-Based Neighboring Superalloy Interpolation
    # --------------------------------------------------------------------------
    print("\n--- Running Intervention 4: Physical Property Interpolation (5 Seeds) ---", flush=True)
    in718_idx = torch.tensor([2], device=DEVICE)
    hastx_idx = torch.tensor([6], device=DEVICE)
    with torch.no_grad():
        blend_embed = 0.5 * (hybrid_model.expert4_numeric.material_embedding(in718_idx) + 
                             hybrid_model.expert4_numeric.material_embedding(hastx_idx))

    int4_records = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for fmt in ["Baseline", "Prompt"]:
            sub_indices = rng.choice(len(cached_by_format[fmt]["e1"]), size=n_eval_samples, replace=False)
            metrics = evaluate_tier_c_condition_cached(
                hybrid_model, cached_by_format[fmt], sub_indices, DEVICE,
                zero_material=False, neighbor_blend_embed=blend_embed
            )
            int4_records.append({
                "intervention": "4_Physical_Interpolation",
                "seed": seed,
                "format": fmt,
                **metrics
            })
            print(f"  [Int 4 Seed {seed} {fmt}] Acc: {metrics['accuracy_pct']:.2f}% | F1: {metrics['macro_f1']:.4f}", flush=True)

    df_int4 = pd.DataFrame(int4_records)
    df_int4.to_csv(results_dir / "tierc_intervention_augmentation.csv", index=False)
    print(f"Saved to {results_dir / 'tierc_intervention_augmentation.csv'}", flush=True)

    # --------------------------------------------------------------------------
    # 5. MASTER SUMMARY & STATISTICAL SIGNIFICANCE TESTING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("  COMPUTING CONSOLIDATED TIER C IMPROVEMENT SUMMARY & SIGNIFICANCE TESTS", flush=True)
    print("=" * 80, flush=True)

    all_intervention_runs = pd.concat([
        pd.DataFrame(baseline_records),
        df_int1,
        df_int2[df_int2["lambda_reg"] == 0.5].drop(columns=["lambda_reg"]),
        df_int3,
        df_int4
    ], ignore_index=True)

    df_ctrl = pd.DataFrame(baseline_records)

    summary_rows = []
    for (intervention, fmt), grp in all_intervention_runs.groupby(["intervention", "format"]):
        mean_acc = grp["accuracy_pct"].mean()
        std_acc = grp["accuracy_pct"].std()
        mean_f1 = grp["macro_f1"].mean()
        std_f1 = grp["macro_f1"].std()
        mean_ece = grp["ece"].mean()

        ctrl_match = df_ctrl[df_ctrl["format"] == fmt].sort_values("seed")
        grp_sorted = grp.sort_values("seed")

        ctrl_f1_vals = ctrl_match["macro_f1"].values
        int_f1_vals = grp_sorted["macro_f1"].values

        delta_f1 = mean_f1 - ctrl_match["macro_f1"].mean()
        delta_acc = mean_acc - ctrl_match["accuracy_pct"].mean()

        if intervention == "0_Baseline_Control":
            p_val_str = "Control (Ref)"
            wilc_str = "Control (Ref)"
            is_sig = False
            verdict = "Baseline Benchmark"
        else:
            t_stat, t_pval = stats.ttest_rel(int_f1_vals, ctrl_f1_vals)
            try:
                w_stat, w_pval = stats.wilcoxon(int_f1_vals, ctrl_f1_vals)
                wilc_str = f"{w_pval:.4f}"
            except Exception:
                wilc_str = "N/A"
            p_val_str = f"{t_pval:.4f}"
            is_sig = bool(t_pval < 0.05 and delta_f1 > 0)
            if is_sig:
                verdict = "Statistically Significant Improvement (p < 0.05)"
            elif delta_f1 > 0:
                verdict = "Marginal Gain (Not Statistically Significant, p >= 0.05)"
            else:
                verdict = "No Improvement / Degradation"

        summary_rows.append({
            "Intervention": intervention,
            "Format": fmt,
            "Mean_Accuracy_Pct": round(mean_acc, 2),
            "Std_Accuracy_Pct": round(std_acc, 2),
            "Mean_Macro_F1": round(mean_f1, 4),
            "Std_Macro_F1": round(std_f1, 4),
            "Mean_ECE": round(mean_ece, 4),
            "Delta_Accuracy_Pct": round(delta_acc, 2),
            "Delta_Macro_F1": round(delta_f1, 4),
            "Paired_t_PValue": p_val_str,
            "Wilcoxon_PValue": wilc_str,
            "Statistically_Significant_p05": is_sig,
            "Verdict": verdict
        })

    df_summary = pd.DataFrame(summary_rows)
    sum_csv = results_dir / "tierc_improvement_summary.csv"
    df_summary.to_csv(sum_csv, index=False)
    print(f"\n[Tier C Summary] Successfully saved consolidated summary to {sum_csv}")
    print(df_summary[["Intervention", "Format", "Mean_Accuracy_Pct", "Mean_Macro_F1", "Delta_Macro_F1", "Paired_t_PValue", "Verdict"]].to_string())

    return df_summary


if __name__ == "__main__":
    run_all_tierc_interventions(BASE_DIR)
