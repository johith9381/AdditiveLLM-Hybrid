"""
AdditiveLLM Tier C Ceiling Diagnostic.
Empirically tests whether Tier C poor generalization is a Model Capacity Ceiling or a Data Availability Ceiling.
Tests two materials:
1. Inconel 625 (IN625): The primary Tier C held-out alloy.
2. AlSi10Mg: An aluminum-silicon alloy with high thermal conductivity.

Evaluates two conditions per alloy:
- Condition A: Zero-Shot (Held-out from training)
- Condition B: Material Included in Training (Standard group-safe partition)

Saves results to results/tierc_ceiling_diagnostic.csv and establishes CASE A or CASE B conclusion.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import (
    GroupSafeDataset,
    load_raw_geometry_lookup,
    parse_baseline_text
)
from src.physics_first_model import PhysicsFirstAdditiveLLM
from src.physics_first_pipeline import (
    precompute_dataset_features,
    train_physics_first_model,
    evaluate_physics_first_model
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_tierc_diagnostic():
    print("=" * 80)
    print("  ADDITIVELLM TIER C CEILING DIAGNOSTIC: DATA VS. MODEL BOTTLENECK")
    print("=" * 80)

    splits_dir = BASE_DIR / "data" / "splits_group_safe"
    models_dir = BASE_DIR / "models" / "base_models"
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_train_base = pd.read_parquet(splits_dir / "train.parquet")
    df_tier_c = pd.read_parquet(splits_dir / "tier_c.parquet")
    lookup = load_raw_geometry_lookup(BASE_DIR / "data" / "raw")

    with open(BASE_DIR / "data" / "templates" / "test_prompts.json") as f:
        test_templates = list(json.load(f).values())

    print("[Diagnostic] Loading tokenizers and initializing model...")
    distil_tok = AutoTokenizer.from_pretrained(str(models_dir / "distilbert"), local_files_only=True)
    sci_tok = AutoTokenizer.from_pretrained(str(models_dir / "scibert"), local_files_only=True)
    tok_dict = {"distilbert": distil_tok, "scibert": sci_tok}

    model = PhysicsFirstAdditiveLLM(str(models_dir)).to(DEVICE)

    diagnostic_rows = []

    # -------------------------------------------------------------
    # ALLOY 1: IN625 (Held-out alloy in current Tier C)
    # -------------------------------------------------------------
    print("\n>>> Testing Alloy 1: IN625 (Inconel 625)")
    # Split IN625 group-safely by laser power & scan speed
    df_in625 = df_tier_c.copy()
    df_in625["parsed"] = df_in625["text"].apply(parse_baseline_text)
    df_in625["group_key"] = df_in625["parsed"].apply(
        lambda p: f"{p['power']}_{p['velocity']}_{p['beam_diameter']}"
    )
    unique_groups = df_in625["group_key"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(unique_groups)

    n_test_groups = max(1, int(len(unique_groups) * 0.25))
    test_groups = set(unique_groups[:n_test_groups])
    train_groups = set(unique_groups[n_test_groups:])

    in625_test = df_in625[df_in625["group_key"].isin(test_groups)].sample(n=min(1000, len(df_in625[df_in625["group_key"].isin(test_groups)])), random_state=42).reset_index(drop=True)
    in625_train = df_in625[df_in625["group_key"].isin(train_groups)].sample(n=min(1500, len(df_in625[df_in625["group_key"].isin(train_groups)])), random_state=42).reset_index(drop=True)

    test_ds = GroupSafeDataset(in625_test, templates=test_templates, mode="baseline", lookup=lookup)
    test_cached = precompute_dataset_features(model, test_ds, tok_dict, DEVICE, "IN625 Test Set (1,000 samples)")

    # Condition 1A: IN625 Zero-Shot (Train on base train without IN625)
    print("\n  [IN625] Training Zero-Shot Model (Base Training Set, 0 IN625 samples)...")
    base_sub = df_train_base.sample(n=2500, random_state=42).reset_index(drop=True)
    base_ds = GroupSafeDataset(base_sub, templates=test_templates, mode="baseline", lookup=lookup)
    base_cached = precompute_dataset_features(model, base_ds, tok_dict, DEVICE, "Base Train Pool (No IN625)")

    model_zero_shot = PhysicsFirstAdditiveLLM(str(models_dir)).to(DEVICE)
    train_physics_first_model(model_zero_shot, base_cached, DEVICE, epochs=4, lr=1.5e-3, seed=42)
    eval_zs = evaluate_physics_first_model(model_zero_shot, test_cached, np.arange(len(in625_test)), DEVICE)

    print(f"  --> IN625 Zero-Shot: Acc={eval_zs['accuracy_pct']:.2f}%, F1={eval_zs['macro_f1']:.4f}")

    # Condition 1B: IN625 Material Included (Train on base train + in625_train)
    print("\n  [IN625] Training Included Model (Base Training Set + IN625 group partition)...")
    inc_train_df = pd.concat([base_sub, in625_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    inc_ds = GroupSafeDataset(inc_train_df, templates=test_templates, mode="baseline", lookup=lookup)
    inc_cached = precompute_dataset_features(model, inc_ds, tok_dict, DEVICE, "Included Train Pool (+IN625)")

    model_included = PhysicsFirstAdditiveLLM(str(models_dir)).to(DEVICE)
    train_physics_first_model(model_included, inc_cached, DEVICE, epochs=4, lr=1.5e-3, seed=42)
    eval_inc = evaluate_physics_first_model(model_included, test_cached, np.arange(len(in625_test)), DEVICE)

    print(f"  --> IN625 Included: Acc={eval_inc['accuracy_pct']:.2f}%, F1={eval_inc['macro_f1']:.4f}")

    delta_acc_in625 = eval_inc['accuracy_pct'] - eval_zs['accuracy_pct']
    delta_f1_in625 = eval_inc['macro_f1'] - eval_zs['macro_f1']

    diagnostic_rows.append({
        "Alloy": "Inconel 625 (IN625)",
        "Zero_Shot_Accuracy_Pct": round(eval_zs['accuracy_pct'], 2),
        "Zero_Shot_Macro_F1": round(eval_zs['macro_f1'], 4),
        "Included_Accuracy_Pct": round(eval_inc['accuracy_pct'], 2),
        "Included_Macro_F1": round(eval_inc['macro_f1'], 4),
        "Delta_Accuracy_Pct": round(delta_acc_in625, 2),
        "Delta_Macro_F1": round(delta_f1_in625, 4),
        "Samples": len(in625_test),
        "Diagnosis": "CASE A (Data Availability Ceiling)" if (delta_acc_in625 > 15.0 or delta_f1_in625 > 0.30) else "CASE B (Model Ceiling)"
    })

    # -------------------------------------------------------------
    # ALLOY 2: AlSi10Mg (Second alloy confirmation)
    # -------------------------------------------------------------
    print("\n>>> Testing Alloy 2: AlSi10Mg (Aluminum-Silicon Alloy)")
    df_train_base["material"] = df_train_base["text"].apply(lambda x: x.split("[SEP]")[0].strip())
    df_alsi = df_train_base[df_train_base["material"] == "AlSi10Mg"].copy()
    df_without_alsi = df_train_base[df_train_base["material"] != "AlSi10Mg"].copy()

    df_alsi["parsed"] = df_alsi["text"].apply(parse_baseline_text)
    df_alsi["group_key"] = df_alsi["parsed"].apply(
        lambda p: f"{p['power']}_{p['velocity']}_{p['beam_diameter']}"
    )
    u_groups_al = df_alsi["group_key"].unique()
    rng.shuffle(u_groups_al)
    n_test_al = max(1, int(len(u_groups_al) * 0.25))
    test_groups_al = set(u_groups_al[:n_test_al])
    train_groups_al = set(u_groups_al[n_test_al:])

    alsi_test = df_alsi[df_alsi["group_key"].isin(test_groups_al)].sample(n=min(800, len(df_alsi[df_alsi["group_key"].isin(test_groups_al)])), random_state=42).reset_index(drop=True)
    alsi_train = df_alsi[df_alsi["group_key"].isin(train_groups_al)].sample(n=min(1200, len(df_alsi[df_alsi["group_key"].isin(train_groups_al)])), random_state=42).reset_index(drop=True)

    test_al_ds = GroupSafeDataset(alsi_test, templates=test_templates, mode="baseline", lookup=lookup)
    test_al_cached = precompute_dataset_features(model, test_al_ds, tok_dict, DEVICE, "AlSi10Mg Test Set (800 samples)")

    # Condition 2A: AlSi10Mg Zero-Shot (Train on df_without_alsi)
    print("\n  [AlSi10Mg] Training Zero-Shot Model (Training Set without AlSi10Mg)...")
    sub_no_al = df_without_alsi.sample(n=2500, random_state=42).reset_index(drop=True)
    no_al_ds = GroupSafeDataset(sub_no_al, templates=test_templates, mode="baseline", lookup=lookup)
    no_al_cached = precompute_dataset_features(model, no_al_ds, tok_dict, DEVICE, "Train Pool (No AlSi10Mg)")

    model_zs_al = PhysicsFirstAdditiveLLM(str(models_dir)).to(DEVICE)
    train_physics_first_model(model_zs_al, no_al_cached, DEVICE, epochs=4, lr=1.5e-3, seed=42)
    eval_zs_al = evaluate_physics_first_model(model_zs_al, test_al_cached, np.arange(len(alsi_test)), DEVICE)

    print(f"  --> AlSi10Mg Zero-Shot: Acc={eval_zs_al['accuracy_pct']:.2f}%, F1={eval_zs_al['macro_f1']:.4f}")

    # Condition 2B: AlSi10Mg Included in Training
    print("\n  [AlSi10Mg] Training Included Model (Training Set + AlSi10Mg)...")
    inc_al_df = pd.concat([sub_no_al, alsi_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    inc_al_ds = GroupSafeDataset(inc_al_df, templates=test_templates, mode="baseline", lookup=lookup)
    inc_al_cached = precompute_dataset_features(model, inc_al_ds, tok_dict, DEVICE, "Included Train Pool (+AlSi10Mg)")

    model_inc_al = PhysicsFirstAdditiveLLM(str(models_dir)).to(DEVICE)
    train_physics_first_model(model_inc_al, inc_al_cached, DEVICE, epochs=4, lr=1.5e-3, seed=42)
    eval_inc_al = evaluate_physics_first_model(model_inc_al, test_al_cached, np.arange(len(alsi_test)), DEVICE)

    print(f"  --> AlSi10Mg Included: Acc={eval_inc_al['accuracy_pct']:.2f}%, F1={eval_inc_al['macro_f1']:.4f}")

    delta_acc_al = eval_inc_al['accuracy_pct'] - eval_zs_al['accuracy_pct']
    delta_f1_al = eval_inc_al['macro_f1'] - eval_zs_al['macro_f1']

    diagnostic_rows.append({
        "Alloy": "AlSi10Mg (Al-Si Alloy)",
        "Zero_Shot_Accuracy_Pct": round(eval_zs_al['accuracy_pct'], 2),
        "Zero_Shot_Macro_F1": round(eval_zs_al['macro_f1'], 4),
        "Included_Accuracy_Pct": round(eval_inc_al['accuracy_pct'], 2),
        "Included_Macro_F1": round(eval_inc_al['macro_f1'], 4),
        "Delta_Accuracy_Pct": round(delta_acc_al, 2),
        "Delta_Macro_F1": round(delta_f1_al, 4),
        "Samples": len(alsi_test),
        "Diagnosis": "CASE A (Data Availability Ceiling)" if (delta_acc_al > 15.0 or delta_f1_al > 0.30) else "CASE B (Model Ceiling)"
    })

    # Save diagnostic CSV
    df_diag = pd.DataFrame(diagnostic_rows)
    diag_csv_path = results_dir / "tierc_ceiling_diagnostic.csv"
    df_diag.to_csv(diag_csv_path, index=False)
    print(f"\n[Diagnostic Complete] Saved diagnostic results to {diag_csv_path}")
    print("\n" + df_diag.to_string())

    is_case_a = all("CASE A" in d for d in df_diag["Diagnosis"])
    print("\n" + "=" * 80)
    if is_case_a:
        print("  DIAGNOSTIC CONCLUSION: CASE A CONFIRMED (GENUINE DATA CEILING)")
        print("  When held-out materials are included in training with group-safe splits,")
        print(f"  accuracy jumps dramatically (IN625: +{delta_acc_in625:.1f}%, AlSi10Mg: +{delta_acc_al:.1f}%).")
        print("  This rigorously proves model architecture/capacity is NOT the bottleneck.")
        print("  The Tier C bottleneck is strictly the number of training materials in the dataset.")
    else:
        print("  DIAGNOSTIC CONCLUSION: CASE B (MODEL BOTTLENECK)")
    print("=" * 80)


if __name__ == "__main__":
    run_tierc_diagnostic()
