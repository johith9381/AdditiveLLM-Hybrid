"""
Evaluate T5-Small on full validation baseline set (72,475 samples).
Caches 512-d transformer embeddings once, then computes exact metrics for all 5 checkpoints (5, 10, 15, 20, 25 epochs).
"""

import sys
import time
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.evaluate import compute_multilabel_metrics

CLASS_NAMES = ["None", "Keyhole", "Lack of Fusion", "Balling"]

def parse_labels_array(raw_labels):
    parsed = []
    for l in raw_labels:
        if isinstance(l, str):
            l = ast.literal_eval(l)
        parsed.append(l)
    return np.array(parsed, dtype=np.int32)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[T5 Eval] Compute device: {device}")
    
    # 1. Load validation baseline
    df = pd.read_parquet("data/splits/validation_baseline.parquet")
    print(f"[T5 Eval] Loaded {len(df):,} validation samples.")
    texts = df["text"].tolist()
    labels = parse_labels_array(df["label"].tolist())
    
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_cache_file = cache_dir / "t5_val_baseline_embeddings.npy"
    
    # 2. Extract or load cached embeddings
    if emb_cache_file.exists():
        print(f"[T5 Eval] Loading cached embeddings from {emb_cache_file}...")
        embeddings = np.load(emb_cache_file)
        print(f"[T5 Eval] Loaded embeddings shape: {embeddings.shape}")
    else:
        print(f"[T5 Eval] Computing 512-d embeddings for {len(texts):,} samples (max_length=256)...")
        tokenizer = AutoTokenizer.from_pretrained("models/base_models/t5-small", local_files_only=True)
        base_model = T5EncoderModel.from_pretrained("models/base_models/t5-small", local_files_only=True).to(device)
        base_model.eval()
        
        batch_size = 256
        all_embeddings = []
        t0 = time.time()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b_texts = texts[i:i + batch_size]
                enc = tokenizer(b_texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(device)
                out = base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                summed = out.last_hidden_state.sum(dim=1)
                all_embeddings.append(summed.cpu().numpy().astype(np.float32))
                
                if (i // batch_size) % 20 == 0:
                    done = min(i + batch_size, len(texts))
                    elapsed = time.time() - t0
                    print(f"  Processed {done:,}/{len(texts):,} samples ({done/elapsed:.1f} samples/s)...", flush=True)
                    
        embeddings = np.concatenate(all_embeddings, axis=0)
        np.save(emb_cache_file, embeddings)
        print(f"[T5 Eval] Saved embeddings {embeddings.shape} in {time.time() - t0:.1f}s to {emb_cache_file}!")
        
    # 3. Evaluate each epoch checkpoint
    epochs = [5, 10, 15, 20, 25]
    results_by_epoch = {}
    X_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    print("\n" + "=" * 70)
    print(f"  T5-SMALL BASELINE PROGRESSION (5 -> 25 EPOCHS) ON {len(texts):,} SAMPLES")
    print("=" * 70)
    print(f"{'Epoch':<8} {'Binary Acc':<14} {'Exact Match':<14} {'Macro F1':<12} {'Micro F1':<12}")
    print("-" * 70)
    
    for ep in epochs:
        head_path = Path(f"models/checkpoints/t5_baseline_{ep}/classification_head.pt")
        if not head_path.exists():
            print(f"Checkpoint not found for epoch {ep}: {head_path}")
            continue
            
        head_state = torch.load(head_path, map_location=device)
        classifier = nn.Linear(512, 4).to(device)
        classifier.load_state_dict(head_state)
        classifier.eval()
        
        with torch.no_grad():
            logits = classifier(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(np.int32)
            
        metrics = compute_multilabel_metrics(labels, preds, probs)
        results_by_epoch[ep] = {
            "epoch": ep,
            "overall_accuracy": float(metrics["mean_binary_accuracy"]),
            "exact_match_ratio": float(metrics["exact_match_ratio"]),
            "macro_f1": float(metrics["macro_f1"]),
            "micro_f1": float(metrics["micro_f1"]),
            "per_class": metrics["per_class"],
            "confusion_matrices": metrics["confusion_matrices"],
        }
        
        print(f"{ep:<8} {metrics['mean_binary_accuracy']*100:<13.2f}% {metrics['exact_match_ratio']*100:<13.2f}% {metrics['macro_f1']:<12.4f} {metrics['micro_f1']:<12.4f}")
        
    print("=" * 70)
    
    # 4. Save results to reports/
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "t5_baseline_progression.json"
    with open(out_file, "w") as f:
        json.dump(results_by_epoch, f, indent=2)
    print(f"\n[T5 Eval] Results successfully saved to {out_file}")

if __name__ == "__main__":
    main()
