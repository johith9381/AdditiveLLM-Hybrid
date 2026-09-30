"""
Checkpoint Head Downloader for AdditiveLLM models.
Downloads `classification_head.pt`, `config.json`, and tokenizer configurations
from Hugging Face into `models/checkpoints/`.
"""

import json
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"

with open(BASE_DIR / "reports" / "checkpoint_availability_report.json") as f:
    audit_data = json.load(f)

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "AdditiveLLM-CheckpointDownloader/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())

def main():
    print("=== Downloading Model Classification Heads ===")
    total = 0
    for split_type in ["baseline", "prompt"]:
        for model_name, epochs in audit_data[split_type].items():
            for epoch, info in epochs.items():
                if not info["available"]:
                    continue
                repo_id = info["repo_id"]
                dest_dir = CHECKPOINTS_DIR / f"{model_name}_{split_type}_{epoch}"
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Download classification_head.pt
                head_url = f"https://huggingface.co/{repo_id}/resolve/main/classification_head.pt"
                head_dest = dest_dir / "classification_head.pt"
                try:
                    download_file(head_url, head_dest)
                    total += 1
                    print(f"[Downloaded] {model_name} {split_type} epoch {epoch}: classification_head.pt ({head_dest.stat().st_size / 1024:.1f} KB)")
                except Exception as e:
                    print(f"[Error] {repo_id} classification_head.pt: {e}")

                # Download config.json if present
                config_url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
                config_dest = dest_dir / "config.json"
                try:
                    download_file(config_url, config_dest)
                except Exception:
                    pass

    print(f"\n[OK] Successfully downloaded {total} classification heads!")

if __name__ == "__main__":
    main()
