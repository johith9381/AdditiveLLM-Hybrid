"""
Controlled Benchmark: On-The-Fly Template Sampling vs. Static Expansion.
Measures:
1. Peak Memory (RAM in MB) using psutil.
2. Training throughput & time per epoch (seconds).
3. Resulting validation accuracy & Macro F1 on held-out templates.
Saves measured empirical benchmark to results/template_sampling_benchmark.csv.
"""

import sys
import time
import json
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_pipeline import parse_baseline_text, format_prompt_template


class StaticExpandedDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len=128):
        self.texts = texts
        self.labels = [np.array(l, dtype=np.float32) for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class OnTheFlyDataset(Dataset):
    def __init__(self, base_samples: list, templates: list, tokenizer, max_len=128):
        self.base_samples = base_samples
        self.templates = templates
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.base_samples)

    def __getitem__(self, idx):
        item = self.base_samples[idx]
        tmpl = np.random.choice(self.templates)
        prompt_text = format_prompt_template(tmpl, item["parsed"])
        enc = self.tokenizer(
            prompt_text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.float32)
        }


def run_benchmark():
    print("=" * 70)
    print("  TEMPLATE SAMPLING BENCHMARK: STATIC EXPANSION VS ON-THE-FLY")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load templates
    with open(data_dir / "templates" / "train_prompts.json") as f:
        train_templates = list(json.load(f).values())
    with open(data_dir / "templates" / "validation_prompts.json") as f:
        val_templates = list(json.load(f).values())

    # Controlled sample: 600 base rows (6,000 static rows) and 200 validation rows
    df_train = pd.read_parquet(data_dir / "splits_group_safe" / "train.parquet").sample(n=600, random_state=42)
    df_val = pd.read_parquet(data_dir / "splits_group_safe" / "tier_a.parquet").sample(n=200, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir / "models" / "base_models" / "distilbert"), local_files_only=True)
    process = psutil.Process()

    def parse_rows(df):
        parsed = []
        for _, r in df.iterrows():
            lbl = r["label"]
            if isinstance(lbl, str):
                import ast
                lbl = ast.literal_eval(lbl)
            p = parse_baseline_text(r["text"])
            parsed.append({"parsed": p, "label": list(lbl), "baseline": r["text"]})
        return parsed

    train_samples = parse_rows(df_train)
    val_samples = parse_rows(df_val)

    def build_test_model():
        base_model = AutoModel.from_pretrained(str(base_dir / "models" / "base_models" / "distilbert"), local_files_only=True)
        for p in base_model.parameters():
            p.requires_grad = False
        classifier = nn.Linear(768, 4)
        return nn.Sequential(base_model, classifier).to(device)

    # -------------------------------------------------------------
    # 1. Static Expansion Benchmark (Pre-materialize 10 templates per row)
    # -------------------------------------------------------------
    print("\n[Benchmark 1/2] Benchmarking Static Expansion (10x Materialization)...")
    ram_before_static = process.memory_info().rss / (1024 * 1024)
    t0_static_prep = time.time()

    static_texts = []
    static_labels = []
    expansion_factor = 10
    for s in train_samples:
        for i in range(expansion_factor):
            tmpl = train_templates[i % len(train_templates)]
            static_texts.append(format_prompt_template(tmpl, s["parsed"]))
            static_labels.append(s["label"])

    static_dataset = StaticExpandedDataset(static_texts, static_labels, tokenizer, max_len=128)
    static_loader = DataLoader(static_dataset, batch_size=64, shuffle=True)
    ram_after_static = process.memory_info().rss / (1024 * 1024)
    peak_ram_static = max(ram_before_static, ram_after_static)
    prep_time_static = time.time() - t0_static_prep

    print(f"  Pre-materialized {len(static_texts):,} rows in {prep_time_static:.3f}s | RAM: {peak_ram_static:.1f} MB")

    model_static = build_test_model()
    opt_static = torch.optim.AdamW(model_static[1].parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    t0_train = time.time()
    for batch in static_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        opt_static.zero_grad()
        with torch.no_grad():
            out = model_static[0](input_ids=input_ids, attention_mask=attn_mask)
            mask_exp = attn_mask.unsqueeze(-1).expand_as(out.last_hidden_state)
            pooled = (out.last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        logits = model_static[1](pooled)
        loss = loss_fn(logits, lbls)
        loss.backward()
        opt_static.step()
    time_static_epoch = time.time() - t0_train
    throughput_static = len(static_texts) / time_static_epoch
    print(f"  Static Epoch Time: {time_static_epoch:.2f}s | Throughput: {throughput_static:.1f} samples/s")

    # Batched evaluation on held-out validation templates
    def evaluate_model_batched(model, samples, templates, batch_size=64):
        model.eval()
        val_texts = [format_prompt_template(templates[0], s["parsed"]) for s in samples]
        val_labels = np.array([s["label"] for s in samples], dtype=int)
        
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                b_texts = val_texts[i:i + batch_size]
                enc = tokenizer(b_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
                out = model[0](input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                mask_exp = enc["attention_mask"].unsqueeze(-1).expand_as(out.last_hidden_state)
                pooled = (out.last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
                logits = model[1](pooled)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                all_preds.append(preds)
        
        preds_arr = np.concatenate(all_preds, axis=0)
        exact_matches = np.all(preds_arr == val_labels, axis=1)
        acc = np.mean(exact_matches) * 100.0
        
        f1_list = []
        for p, a in zip(preds_arr, val_labels):
            tp = np.sum((p == 1) & (a == 1))
            fp = np.sum((p == 1) & (a == 0))
            fn = np.sum((p == 0) & (a == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            f1_list.append(f1)
        return float(acc), float(np.mean(f1_list))

    acc_static, f1_static = evaluate_model_batched(model_static, val_samples, val_templates)
    print(f"  Static Accuracy on Held-Out Templates: {acc_static:.2f}% | F1: {f1_static:.4f}")

    # -------------------------------------------------------------
    # 2. On-The-Fly Dynamic Sampling Benchmark
    # -------------------------------------------------------------
    print("\n[Benchmark 2/2] Benchmarking On-The-Fly Template Sampling (Natural Base Size)...")
    ram_before_fly = process.memory_info().rss / (1024 * 1024)
    t0_fly_prep = time.time()

    fly_dataset = OnTheFlyDataset(train_samples, train_templates, tokenizer, max_len=128)
    fly_loader = DataLoader(fly_dataset, batch_size=64, shuffle=True)
    ram_after_fly = process.memory_info().rss / (1024 * 1024)
    peak_ram_fly = max(ram_before_fly, ram_after_fly)
    prep_time_fly = time.time() - t0_fly_prep

    print(f"  Base Dataset: {len(train_samples):,} rows (Zero pre-expansion) | RAM: {peak_ram_fly:.1f} MB")

    model_fly = build_test_model()
    opt_fly = torch.optim.AdamW(model_fly[1].parameters(), lr=1e-3)

    t0_train_fly = time.time()
    for batch in fly_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        opt_fly.zero_grad()
        with torch.no_grad():
            out = model_fly[0](input_ids=input_ids, attention_mask=attn_mask)
            mask_exp = attn_mask.unsqueeze(-1).expand_as(out.last_hidden_state)
            pooled = (out.last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        logits = model_fly[1](pooled)
        loss = loss_fn(logits, lbls)
        loss.backward()
        opt_fly.step()
    time_fly_epoch = time.time() - t0_train_fly
    throughput_fly = len(train_samples) / time_fly_epoch
    print(f"  On-The-Fly Epoch Time: {time_fly_epoch:.2f}s | Throughput: {throughput_fly:.1f} samples/s")

    acc_fly, f1_fly = evaluate_model_batched(model_fly, val_samples, val_templates)
    print(f"  On-The-Fly Accuracy on Held-Out Templates: {acc_fly:.2f}% | F1: {f1_fly:.4f}")

    benchmark_rows = [
        {
            "approach": "Static Expansion (Paper Method)",
            "dataset_rows": len(static_texts),
            "storage_expansion_ratio": f"{expansion_factor:.1f}x",
            "peak_ram_mb": round(peak_ram_static, 2),
            "prep_time_sec": round(prep_time_static, 3),
            "time_per_epoch_sec": round(time_static_epoch, 2),
            "throughput_samples_per_sec": round(throughput_static, 1),
            "val_accuracy_pct": round(acc_static, 2),
            "val_macro_f1": round(f1_static, 4),
            "relative_ram": "100.0% (Baseline)",
        },
        {
            "approach": "On-The-Fly Sampling (Proposed Hybrid)",
            "dataset_rows": len(train_samples),
            "storage_expansion_ratio": "1.0x (Natural Base)",
            "peak_ram_mb": round(peak_ram_fly, 2),
            "prep_time_sec": round(prep_time_fly, 3),
            "time_per_epoch_sec": round(time_fly_epoch, 2),
            "throughput_samples_per_sec": round(throughput_fly, 1),
            "val_accuracy_pct": round(acc_fly, 2),
            "val_macro_f1": round(f1_fly, 4),
            "relative_ram": f"{round((peak_ram_fly / peak_ram_static) * 100, 1)}%",
        }
    ]

    out_csv = results_dir / "template_sampling_benchmark.csv"
    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"  BENCHMARK COMPLETED SUCCESSFULLY! Saved to {out_csv}")
    print("=" * 70)
    print(bench_df[["approach", "dataset_rows", "peak_ram_mb", "time_per_epoch_sec", "val_accuracy_pct", "val_macro_f1"]])


if __name__ == "__main__":
    run_benchmark()
