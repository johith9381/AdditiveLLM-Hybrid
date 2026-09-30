"""
Unified, rock-solid, disk-memmapped evaluation engine for AdditiveLLM models.
Features:
- Memmapped embedding storage: uses near-zero RAM, immune to OOM/SIGKILL.
- Resumable: automatically resumes from the last completed batch if interrupted.
- MPS cache management: calls torch.mps.empty_cache() periodically.
- Evaluates full 72,475 validation samples across 5, 10, 15, 20, 25 epoch checkpoints.
"""

import sys
import time
import json
import ast
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.evaluate import compute_multilabel_metrics

CLASS_NAMES = ["None", "Keyhole", "Lack of Fusion", "Balling"]

MODEL_CONFIGS = {
    "distilbert": {
        "hidden_dim": 768,
        "base_model_path": "models/base_models/distilbert",
        "model_cls": AutoModel,
        "tokenizer_cls": AutoTokenizer,
        "is_t5": False,
    },
    "t5": {
        "hidden_dim": 512,
        "base_model_path": "models/base_models/t5-small",
        "model_cls": T5EncoderModel,
        "tokenizer_cls": AutoTokenizer,
        "is_t5": True,
    },
    "scibert": {
        "hidden_dim": 768,
        "base_model_path": "models/base_models/scibert",
        "model_cls": AutoModel,
        "tokenizer_cls": AutoTokenizer,
        "is_t5": False,
    },
}

def parse_labels_array(raw_labels):
    parsed = []
    for l in raw_labels:
        if isinstance(l, str):
            l = ast.literal_eval(l)
        parsed.append(l)
    return np.array(parsed, dtype=np.int32)

def extract_embeddings_memmap(model_name: str, texts: list, device: torch.device) -> np.ndarray:
    cfg = MODEL_CONFIGS[model_name]
    hidden_dim = cfg["hidden_dim"]
    n_samples = len(texts)
    
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    mmap_file = cache_dir / f"{model_name}_val_baseline_embeddings.npy"
    meta_file = cache_dir / f"{model_name}_val_baseline_meta.json"
    
    # Check if already fully computed
    if mmap_file.exists() and meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        if meta.get("completed", False) and meta.get("count", 0) == n_samples:
            print(f"[{model_name.upper()}] Loading fully cached embeddings ({n_samples:,}, {hidden_dim})...")
            return np.load(mmap_file, mmap_mode="r")
            
    # Open or create memmap array
    if mmap_file.exists() and meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        start_idx = meta.get("count", 0)
        embeddings = np.lib.format.open_memmap(mmap_file, mode="r+", dtype="float32", shape=(n_samples, hidden_dim))
        print(f"[{model_name.upper()}] Resuming embedding computation from sample {start_idx:,}/{n_samples:,}...")
    else:
        start_idx = 0
        embeddings = np.lib.format.open_memmap(mmap_file, mode="w+", dtype="float32", shape=(n_samples, hidden_dim))
        print(f"[{model_name.upper()}] Starting embedding computation for {n_samples:,} samples ({hidden_dim}-dim)...")

    # Load tokenizer and base model
    tok = cfg["tokenizer_cls"].from_pretrained(cfg["base_model_path"], local_files_only=True)
    model = cfg["model_cls"].from_pretrained(cfg["base_model_path"], local_files_only=True).to(device)
    model.eval()
    
    batch_size = 128
    t0 = time.time()
    last_print = t0
    
    with torch.no_grad():
        for i in range(start_idx, n_samples, batch_size):
            b_texts = texts[i:i + batch_size]
            enc = tok(b_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            summed = out.last_hidden_state.sum(dim=1)
            embeddings[i:i + len(b_texts)] = summed.cpu().numpy()
            
            # Periodically free MPS cache and update meta
            if device.type == "mps":
                torch.mps.empty_cache()
                
            now = time.time()
            if now - last_print >= 5.0 or (i + len(b_texts)) == n_samples:
                done = i + len(b_texts)
                rate = (done - start_idx) / max(0.001, now - t0)
                pct = done / n_samples * 100.0
                print(f"  [{model_name.upper()}] {done:,}/{n_samples:,} ({pct:.1f}%) @ {rate:.1f} samples/s", flush=True)
                embeddings.flush()
                with open(meta_file, "w") as f:
                    json.dump({"count": done, "completed": done == n_samples}, f)
                last_print = now
                
    embeddings.flush()
    with open(meta_file, "w") as f:
        json.dump({"count": n_samples, "completed": True}, f)
    print(f"[{model_name.upper()}] Embedding extraction complete in {time.time() - t0:.1f}s!\n")
    return embeddings

def evaluate_checkpoints(model_name: str, embeddings: np.ndarray, labels: np.ndarray, device: torch.device):
    cfg = MODEL_CONFIGS[model_name]
    hidden_dim = cfg["hidden_dim"]
    epochs = [5, 10, 15, 20, 25]
    results = {}
    
    print("=" * 75)
    print(f"  {model_name.upper()} BASELINE EVALUATION (5 -> 25 EPOCHS) ON {len(labels):,} SAMPLES")
    print("=" * 75)
    print(f"{'Epoch':<8} {'Binary Acc':<14} {'Exact Match':<14} {'Macro F1':<12} {'Micro F1':<12}")
    print("-" * 75)
    
    # Process classifier in chunks to keep memory low
    chunk_size = 8192
    
    for ep in epochs:
        head_path = Path(f"models/checkpoints/{model_name}_baseline_{ep}/classification_head.pt")
        if not head_path.exists():
            print(f"Checkpoint not found for epoch {ep}: {head_path}")
            continue
            
        head_state = torch.load(head_path, map_location=device)
        classifier = nn.Linear(hidden_dim, 4).to(device)
        classifier.load_state_dict(head_state)
        classifier.eval()
        
        all_logits = []
        with torch.no_grad():
            for c_start in range(0, len(embeddings), chunk_size):
                chunk_emb = torch.tensor(embeddings[c_start:c_start + chunk_size], dtype=torch.float32).to(device)
                logits = classifier(chunk_emb)
                all_logits.append(logits.cpu().numpy())
                
        logits_arr = np.concatenate(all_logits, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits_arr))
        preds = (probs > 0.5).astype(np.int32)
        
        metrics = compute_multilabel_metrics(labels, preds, probs)
        results[ep] = {
            "epoch": ep,
            "overall_accuracy": float(metrics["mean_binary_accuracy"]),
            "exact_match_ratio": float(metrics["exact_match_ratio"]),
            "macro_f1": float(metrics["macro_f1"]),
            "micro_f1": float(metrics["micro_f1"]),
            "per_class": metrics["per_class"],
            "confusion_matrices": metrics["confusion_matrices"],
        }
        print(f"{ep:<8} {metrics['mean_binary_accuracy']*100:<13.2f}% {metrics['exact_match_ratio']*100:<13.2f}% {metrics['macro_f1']:<12.4f} {metrics['micro_f1']:<12.4f}")
        
    print("=" * 75)
    
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_baseline_progression.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_name.upper()}] Metrics saved to {out_file}\n")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilbert", choices=["distilbert", "t5", "scibert", "all"])
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Evaluation Engine] Active compute device: {device}")
    
    # Load dataset
    df = pd.read_parquet("data/splits/validation_baseline.parquet")
    texts = df["text"].tolist()
    labels = parse_labels_array(df["label"].tolist())
    print(f"[Evaluation Engine] Loaded {len(texts):,} validation baseline samples.")
    
    models_to_run = ["distilbert", "t5", "scibert"] if args.model == "all" else [args.model]
    
    for m in models_to_run:
        print(f"\n>>> Running Evaluation for {m.upper()} <<<")
        embs = extract_embeddings_memmap(m, texts, device)
        evaluate_checkpoints(m, embs, labels, device)

if __name__ == "__main__":
    main()
