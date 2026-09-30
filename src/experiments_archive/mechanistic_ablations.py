"""
Mechanistic and Material Dependency Ablation Suite for AdditiveLLM.
Generates:
1. results/material_dependency_ablation.csv:
   Evaluates impact of categorical material identity embeddings on in-distribution (Tier A/B)
   vs held-out alloy out-of-distribution (Tier C), testing whether Tier C weakness stems from
   reliance on categorical material signal vs physical parameter relationships.
2. results/matscibert_substitute_justification.csv:
   Compares domain-adapted SciBERT substitute branch against a plain duplicate SciBERT branch,
   empirically verifying whether the domain-adapted branch adds measurable value.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import GroupSafeDataset, load_raw_geometry_lookup
from src.hybrid_model import HybridAdditiveLLM
from src.evaluate import compute_ece

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models" / "base_models"
SPLITS_DIR = BASE_DIR / "data" / "splits_group_safe"


def run_material_dependency_ablation(base_dir: Path):
    """
    Evaluates model with full material categorical embedding vs zeroed-out material embedding
    across Tiers A, B, and C.
    """
    print("=" * 80, flush=True)
    print("  RUNNING MATERIAL DEPENDENCY ABLATION (TIER A/B VS TIER C OOD)", flush=True)
    print("=" * 80, flush=True)

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

    tiers = [
        ("Tier A (In-Distribution)", pd.read_parquet(SPLITS_DIR / "tier_a.parquet")),
        ("Tier B (Held-Out Groups)", pd.read_parquet(SPLITS_DIR / "tier_b.parquet")),
        ("Tier C (Held-Out Alloy IN625 OOD)", pd.read_parquet(SPLITS_DIR / "tier_c.parquet")),
    ]

    records = []

    for tier_name, df in tiers:
        for fmt in ["baseline", "prompt"]:
            df_sub = df.sample(n=min(600, len(df)), random_state=42).reset_index(drop=True)
            ds = GroupSafeDataset(df_sub, templates=templates, mode=fmt, lookup=lookup)
            loader = DataLoader(ds, batch_size=64, shuffle=False)

            y_pred_full, y_pred_no_mat, y_true = [], [], []
            with torch.no_grad():
                for batch in loader:
                    texts = batch["text"]
                    num_feats = batch["numeric_features"].to(DEVICE)
                    mat_idx = batch["material_idx"].to(DEVICE)
                    zero_mat_idx = torch.zeros_like(mat_idx).to(DEVICE)
                    targets = batch["labels"].numpy()

                    enc_d = distil_tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(DEVICE)
                    enc_s = sci_tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(DEVICE)

                    e1 = model.expert1_distilbert(enc_d["input_ids"], enc_d["attention_mask"])
                    e2 = model.expert2_scibert(enc_s["input_ids"], enc_s["attention_mask"])
                    e3 = model.expert3_matscibert(enc_s["input_ids"], enc_s["attention_mask"])

                    # 1. Condition: Full Model with Material Categorical Input
                    e4_full = model.expert4_numeric(num_feats, mat_idx)
                    fused_full, _ = model.fusion([e1, e2, e3, e4_full])
                    logits_full = model.classifier(fused_full)
                    y_pred_full.append((torch.sigmoid(logits_full).cpu().numpy() > 0.5).astype(int))

                    # 2. Condition: Zeroed Out Material Identity
                    e4_no_mat = model.expert4_numeric(num_feats, zero_mat_idx)
                    fused_no_mat, _ = model.fusion([e1, e2, e3, e4_no_mat])
                    logits_no_mat = model.classifier(fused_no_mat)
                    y_pred_no_mat.append((torch.sigmoid(logits_no_mat).cpu().numpy() > 0.5).astype(int))

                    y_true.append(targets)

            y_t = np.vstack(y_true)
            yp_f = np.vstack(y_pred_full)
            yp_nm = np.vstack(y_pred_no_mat)

            acc_full = (yp_f == y_t).mean() * 100.0
            f1_full = f1_score(y_t, yp_f, average="macro", zero_division=0)

            acc_no_mat = (yp_nm == y_t).mean() * 100.0
            f1_no_mat = f1_score(y_t, yp_nm, average="macro", zero_division=0)

            delta_acc = acc_full - acc_no_mat
            delta_f1 = f1_full - f1_no_mat

            records.append({
                "Evaluation_Tier": tier_name,
                "Format": fmt.capitalize(),
                "Full_Model_Accuracy_Pct": round(acc_full, 2),
                "Full_Model_Macro_F1": round(f1_full, 4),
                "No_Material_Token_Accuracy_Pct": round(acc_no_mat, 2),
                "No_Material_Token_Macro_F1": round(f1_no_mat, 4),
                "Delta_Accuracy_Pct": round(delta_acc, 2),
                "Delta_Macro_F1": round(delta_f1, 4),
                "Material_Identity_Sensitivity": "High" if delta_f1 > 0.05 else ("Moderate" if delta_f1 > 0.01 else "Low/Negligible")
            })
            print(f"  [{tier_name} {fmt.capitalize()}] Full Acc: {acc_full:.2f}%, F1: {f1_full:.4f} | No-Mat Acc: {acc_no_mat:.2f}%, F1: {f1_no_mat:.4f} | Delta F1: {delta_f1:+.4f}", flush=True)

    df_out = pd.DataFrame(records)
    out_file = RESULTS_DIR / "material_dependency_ablation.csv"
    df_out.to_csv(out_file, index=False)
    print(f"[Material Dependency] Results saved to {out_file}", flush=True)
    return df_out


def run_matscibert_substitute_justification(base_dir: Path):
    """
    Controlled ablation comparing the domain-adapted SciBERT substitute branch against
    a plain duplicate SciBERT branch without adaptation.
    """
    print("\n" + "=" * 80, flush=True)
    print("  RUNNING MATSCIBERT SUBSTITUTE JUSTIFICATION ABLATION", flush=True)
    print("=" * 80, flush=True)

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

    df_tier_a = pd.read_parquet(SPLITS_DIR / "tier_a.parquet")
    df_eval = df_tier_a.sample(n=600, random_state=42).reset_index(drop=True)

    records = []
    for fmt in ["baseline", "prompt"]:
        ds = GroupSafeDataset(df_eval, templates=templates, mode=fmt, lookup=lookup)
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        all_e1, all_e2, all_e3, all_e4 = [], [], [], []
        y_true = []

        with torch.no_grad():
            for batch in loader:
                texts = batch["text"]
                num_feats = batch["numeric_features"].to(DEVICE)
                mat_idx = batch["material_idx"].to(DEVICE)
                y_true.append(batch["labels"].numpy())

                enc_d = distil_tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(DEVICE)
                enc_s = sci_tok(texts, padding="max_length", truncation=True, max_length=80, return_tensors="pt").to(DEVICE)

                e1 = model.expert1_distilbert(enc_d["input_ids"], enc_d["attention_mask"])
                e2 = model.expert2_scibert(enc_s["input_ids"], enc_s["attention_mask"])
                e3 = model.expert3_matscibert(enc_s["input_ids"], enc_s["attention_mask"])
                e4 = model.expert4_numeric(num_feats, mat_idx)

                all_e1.append(e1)
                all_e2.append(e2)
                all_e3.append(e3)
                all_e4.append(e4)

        e1_all = torch.cat(all_e1, dim=0)
        e2_all = torch.cat(all_e2, dim=0)
        e3_all = torch.cat(all_e3, dim=0)
        e4_all = torch.cat(all_e4, dim=0)
        yt = np.vstack(y_true)

        # 1. Full architecture with domain-adapted MatSciBERT substitute
        with torch.no_grad():
            fused_adapted, _ = model.fusion([e1_all, e2_all, e3_all, e4_all])
            logits_adapted = model.classifier(fused_adapted)
            yp_adapted = (torch.sigmoid(logits_adapted).cpu().numpy() > 0.5).astype(int)

        # 2. Control architecture replacing adapted branch with duplicate unadapted SciBERT (e2_all)
        with torch.no_grad():
            fused_dup, _ = model.fusion([e1_all, e2_all, e2_all, e4_all])
            logits_dup = model.classifier(fused_dup)
            yp_dup = (torch.sigmoid(logits_dup).cpu().numpy() > 0.5).astype(int)

        acc_ad = (yp_adapted == yt).mean() * 100.0
        f1_ad = f1_score(yt, yp_adapted, average="macro", zero_division=0)

        acc_dup = (yp_dup == yt).mean() * 100.0
        f1_dup = f1_score(yt, yp_dup, average="macro", zero_division=0)

        records.append({
            "Format": fmt.capitalize(),
            "Domain_Adapted_Substitute_Accuracy_Pct": round(acc_ad, 2),
            "Domain_Adapted_Substitute_Macro_F1": round(f1_ad, 4),
            "Plain_Duplicate_Branch_Accuracy_Pct": round(acc_dup, 2),
            "Plain_Duplicate_Branch_Macro_F1": round(f1_dup, 4),
            "Accuracy_Advantage_Pct": round(acc_ad - acc_dup, 2),
            "Macro_F1_Advantage": round(f1_ad - f1_dup, 4),
            "Measurable_Value_Confirmed": bool((f1_ad - f1_dup) > 0)
        })
        print(f"  [{fmt.capitalize()}] Adapted F1: {f1_ad:.4f} vs Plain Duplicate F1: {f1_dup:.4f} | Adv: {f1_ad - f1_dup:+.4f}", flush=True)

    df_just = pd.DataFrame(records)
    out_file = RESULTS_DIR / "matscibert_substitute_justification.csv"
    df_just.to_csv(out_file, index=False)
    print(f"[MatSciBERT Justification] Results saved to {out_file}", flush=True)
    return df_just


if __name__ == "__main__":
    run_material_dependency_ablation(BASE_DIR)
    run_matscibert_substitute_justification(BASE_DIR)
