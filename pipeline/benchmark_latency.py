"""
Benchmark CPU inference latency and memory footprint (Table 4 Reproduction).
Measures single-sample and batched inference latency on CPU across models.
"""

import time
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

def benchmark_model_cpu(model_name: str, model_path: str, model_cls, is_t5=False):
    print(f"\n[Table 4 Benchmark] Benchmarking {model_name} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = model_cls.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    baseline_text = "Stainless Steel 316L[SEP]195.0 W[SEP]800.0 mm/s[SEP]100.0 microns[SEP]100.0 microns[SEP]30.0 microns"
    prompt_text = "Predict the defect regimes for SS316L processed at laser power 195.0 W, scan speed 800.0 mm/s, beam diameter 100.0 microns, hatch spacing 100.0 microns, and layer height 30.0 microns."
    
    # Measure baseline latency
    enc_base = tokenizer([baseline_text], padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(**enc_base)
            
    n_runs = 50
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            out = model(**enc_base)
            _ = out.last_hidden_state.sum(dim=1)
    base_latency_ms = (time.time() - t0) / n_runs * 1000.0
    
    # Measure prompt latency
    enc_prompt = tokenizer([prompt_text], padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            out = model(**enc_prompt)
            _ = out.last_hidden_state.sum(dim=1)
    prompt_latency_ms = (time.time() - t0) / n_runs * 1000.0
    
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters())
    param_millions = param_count / 1e6
    
    print(f"  Parameters:          {param_millions:.1f}M")
    print(f"  Baseline CPU Latency: {base_latency_ms:.2f} ms/sample")
    print(f"  Prompt CPU Latency:   {prompt_latency_ms:.2f} ms/sample")
    
    return {
        "model": model_name,
        "parameters_m": round(param_millions, 1),
        "baseline_latency_ms": round(base_latency_ms, 2),
        "prompt_latency_ms": round(prompt_latency_ms, 2),
        "samples_per_sec_baseline": round(1000.0 / base_latency_ms, 1),
        "samples_per_sec_prompt": round(1000.0 / prompt_latency_ms, 1),
    }

def main():
    benchmarks = {}
    
    # 1. DistilBERT
    distil_res = benchmark_model_cpu("DistilBERT", "models/base_models/distilbert", AutoModel)
    benchmarks["distilbert"] = distil_res
    
    # 2. T5-Small
    t5_res = benchmark_model_cpu("T5-Small", "models/base_models/t5-small", T5EncoderModel, is_t5=True)
    benchmarks["t5"] = t5_res
    
    # Check if SciBERT exists
    scibert_bin = Path("models/base_models/scibert/pytorch_model.bin")
    if scibert_bin.exists():
        scibert_res = benchmark_model_cpu("SciBERT", "models/base_models/scibert", AutoModel)
        benchmarks["scibert"] = scibert_res
        
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "table4_cpu_latency.json"
    with open(out_file, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"\n[Table 4 Benchmark] Successfully saved to {out_file}")

if __name__ == "__main__":
    main()
