"""
Direct streaming downloader for base model weights using requests.
Avoids huggingface_hub socket hang on macOS LibreSSL.
"""

import os
import sys
import time
from pathlib import Path
import requests

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    print(f"\n[Downloader] Starting download: {url}")
    print(f"[Downloader] Target: {dest_path}")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()
    
    total_bytes = int(response.headers.get("content-length", 0))
    print(f"[Downloader] Total size: {total_bytes / (1024 * 1024):.2f} MB")
    
    downloaded = 0
    t0 = time.time()
    last_print = t0
    
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 512):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2.0 or downloaded == total_bytes:
                    pct = (downloaded / total_bytes * 100) if total_bytes else 0
                    speed = (downloaded / (1024 * 1024)) / max(0.001, now - t0)
                    print(f"  Downloaded {downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                    last_print = now
                    
    temp_path.rename(dest_path)
    print(f"[Downloader] Finished downloading {dest_path.name} in {time.time() - t0:.1f}s!\n")

def main():
    base_dir = Path("models/base_models")
    
    # 1. T5-Small
    t5_dir = base_dir / "t5-small"
    t5_files = {
        "config.json": "https://huggingface.co/google-t5/t5-small/resolve/main/config.json",
        "tokenizer_config.json": "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer_config.json",
        "spiece.model": "https://huggingface.co/google-t5/t5-small/resolve/main/spiece.model",
        "tokenizer.json": "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer.json",
        "model.safetensors": "https://huggingface.co/google-t5/t5-small/resolve/main/model.safetensors",
    }
    print("=== Checking T5-Small ===")
    for fname, url in t5_files.items():
        target = t5_dir / fname
        if not target.exists():
            download_file(url, target)
        else:
            print(f"  {fname} already exists ({target.stat().st_size / (1024*1024):.2f} MB)")

    # 2. SciBERT
    scibert_dir = base_dir / "scibert"
    scibert_files = {
        "config.json": "https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/config.json",
        "vocab.txt": "https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/vocab.txt",
        "pytorch_model.bin": "https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/pytorch_model.bin",
    }
    print("\n=== Checking SciBERT ===")
    for fname, url in scibert_files.items():
        target = scibert_dir / fname
        if not target.exists():
            download_file(url, target)
        else:
            print(f"  {fname} already exists ({target.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
