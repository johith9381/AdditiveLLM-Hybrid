import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pipeline.evaluate import compute_multilabel_metrics

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load validation data
    df = pd.read_parquet("data/splits/validation_baseline.parquet")
    print(f"Loaded {len(df):,} validation samples.")

    texts = df["text"].tolist()
    labels = []
    for l in df["label"]:
        if isinstance(l, str):
            import ast
            l = ast.literal_eval(l)
        labels.append(l)
    y_true = np.array(labels, dtype=np.int8)

    # Load tokenizer and base model
    model_id = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModel.from_pretrained(model_id).to(device)
    base_model.eval()

    # Load head weights for epoch 25
    head_path = "models/checkpoints/distilbert_baseline_25/classification_head.pt"
    head_state = torch.load(head_path, map_location=device)
    classifier = torch.nn.Linear(768, 4).to(device)
    classifier.load_state_dict(head_state)
    classifier.eval()

    # Run inference in batches
    batch_size = 512
    all_logits = []
    print(f"Evaluating {len(texts)} samples in batches of {batch_size}...")
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            out = base_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            summed = out.last_hidden_state.sum(dim=1)
            logits = classifier(summed)
            all_logits.append(logits.cpu().numpy())

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f} seconds ({len(texts)/elapsed:.1f} samples/sec)!")

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    preds_arr = (probs_arr >= 0.5).astype(np.int8)

    metrics = compute_multilabel_metrics(y_true, preds_arr, probs_arr)
    print("\n--- DistilBERT Baseline 25-Epoch Results ---")
    print(f"Mean Binary Accuracy: {metrics['mean_binary_accuracy']*100:.2f}%")
    print(f"Macro F1 Score:       {metrics['macro_f1']:.4f}")
    print(f"Exact Match Ratio:    {metrics['exact_match_ratio']*100:.2f}%")
    for name, c_data in metrics["per_class"].items():
        print(f"  {name:15s}: F1={c_data['f1_score']:.4f}, Acc={c_data['binary_accuracy']*100:.2f}%, Support={c_data['support']}")

if __name__ == "__main__":
    main()
