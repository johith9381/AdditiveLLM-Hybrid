"""
Dataset Downloader for AdditiveLLM reproduction.
Fetches official splits, raw data, and prompt templates from `ppak10/melt-pool-classification`.
"""

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
TEMPLATES_DIR = DATA_DIR / "templates"

BASE_URL = "https://huggingface.co/datasets/ppak10/melt-pool-classification/resolve/main"

FILES_TO_DOWNLOAD = [
    # Source / processed files
    ("v0/data/source/processed/meltpoolclassification.csv", RAW_DIR / "meltpoolclassification.csv"),
    ("v0/data/source/processed/meltpoolgeometry.csv", RAW_DIR / "meltpoolgeometry.csv"),
    ("v0/data/source/processed/dimensions_ss316l_10_micron.csv", RAW_DIR / "dimensions_ss316l_10_micron.csv"),
    ("v0/data/source/processed/dimensions_ti64.pkl", RAW_DIR / "dimensions_ti64.pkl"),
    # Prompt templates
    ("v0/dataset/train_prompts.json", TEMPLATES_DIR / "train_prompts.json"),
    ("v0/dataset/test_prompts.json", TEMPLATES_DIR / "test_prompts.json"),
    ("v0/dataset/validation_prompts.json", TEMPLATES_DIR / "validation_prompts.json"),
    # Splits (Parquet format)
    ("v0/dataset/validation_baseline.parquet", SPLITS_DIR / "validation_baseline.parquet"),
    ("v0/dataset/test_baseline.parquet", SPLITS_DIR / "test_baseline.parquet"),
    ("v0/dataset/train_baseline.parquet", SPLITS_DIR / "train_baseline.parquet"),
    ("v0/dataset/validation_prompt.parquet", SPLITS_DIR / "validation_prompt.parquet"),
    ("v0/dataset/test_prompt.parquet", SPLITS_DIR / "test_prompt.parquet"),
]

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Dataset] Already downloaded: {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return
    print(f"[Dataset] Downloading {url} -> {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "AdditiveLLM-Downloader/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"[Dataset] Finished: {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")

def main():
    print("=== Downloading AdditiveLLM Dataset Artifacts ===")
    for rel_path, dest_path in FILES_TO_DOWNLOAD:
        url = f"{BASE_URL}/{rel_path}"
        try:
            download_file(url, dest_path)
        except Exception as e:
            print(f"[Dataset Error] Failed downloading {rel_path}: {e}")

if __name__ == "__main__":
    main()
