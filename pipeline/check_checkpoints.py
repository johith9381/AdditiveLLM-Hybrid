"""
Verification of Hugging Face Checkpoint Availability.
Probes all 28 expected checkpoint repositories under the `ppak10` namespace
and generates a comprehensive availability report for Table 2 and Table 1.
"""

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

CHECKPOINT_TARGETS = {
    "baseline": {
        "distilbert": [
            ("5", "ppak10/defect-classification-distilbert-baseline-05-epochs"),
            ("10", "ppak10/defect-classification-distilbert-baseline-10-epochs"),
            ("15", "ppak10/defect-classification-distilbert-baseline-15-epochs"),
            ("20", "ppak10/defect-classification-distilbert-baseline-20-epochs"),
            ("25", "ppak10/defect-classification-distilbert-baseline-25-epochs"),
        ],
        "scibert": [
            ("5", "ppak10/defect-classification-scibert-baseline-05-epochs"),
            ("10", "ppak10/defect-classification-scibert-baseline-10-epochs"),
            ("15", "ppak10/defect-classification-scibert-baseline-15-epochs"),
            ("20", "ppak10/defect-classification-scibert-baseline-20-epochs"),
            ("25", "ppak10/defect-classification-scibert-baseline-25-epochs"),
        ],
        "t5": [
            ("5", "ppak10/defect-classification-t5-baseline-05-epochs"),
            ("10", "ppak10/defect-classification-t5-baseline-10-epochs"),
            ("15", "ppak10/defect-classification-t5-baseline-15-epochs"),
            ("20", "ppak10/defect-classification-t5-baseline-20-epochs"),
            ("25", "ppak10/defect-classification-t5-baseline-25-epochs"),
        ],
        "llama": [
            ("5", "ppak10/defect-classification-llama-baseline-05-epochs"),
            ("10", "ppak10/defect-classification-llama-baseline-10-epochs"),
            ("15", "ppak10/defect-classification-llama-baseline-15-epochs"),
            ("20", "ppak10/defect-classification-llama-baseline-20-epochs"),
            ("25", "ppak10/defect-classification-llama-baseline-25-epochs"),
        ],
    },
    "prompt": {
        "distilbert": [
            ("2", "ppak10/defect-classification-distilbert-prompt-02-epochs"),
            ("5", "ppak10/defect-classification-distilbert-prompt-05-epochs"),
        ],
        "scibert": [
            ("2", "ppak10/defect-classification-scibert-prompt-02-epochs"),
            ("5", "ppak10/defect-classification-scibert-prompt-05-epochs"),
        ],
        "t5": [
            ("2", "ppak10/defect-classification-t5-prompt-02-epochs"),
            ("5", "ppak10/defect-classification-t5-prompt-05-epochs"),
        ],
        "llama": [
            ("0.5", "ppak10/defect-classification-llama-prompt-0.5-epochs"),
            ("2", "ppak10/defect-classification-llama-prompt-02-epochs"),
        ],
    }
}

def probe_hf_repo(repo_id: str) -> Tuple[bool, int, List[str]]:
    """Checks if repo exists on HF API and returns list of files."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "AdditiveLLM-Reproduction-Auditor/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                siblings = [s.get("rfilename") for s in data.get("siblings", [])]
                return True, 200, siblings
            return False, resp.status, []
    except urllib.error.HTTPError as e:
        return False, e.code, []
    except Exception as e:
        return False, -1, [str(e)]

def audit_all_checkpoints(output_dir: str = "reports") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "baseline": {},
        "prompt": {},
        "summary": {
            "total_targeted": 28,
            "available_count": 0,
            "missing_count": 0,
            "missing_repos": []
        }
    }

    print("[Checkpoint Audit] Probing Hugging Face for 28 candidate checkpoints...")
    for split_type in ["baseline", "prompt"]:
        for model_name, checkpoints in CHECKPOINT_TARGETS[split_type].items():
            if model_name not in results[split_type]:
                results[split_type][model_name] = {}
            for epoch, repo_id in checkpoints:
                exists, code, files = probe_hf_repo(repo_id)
                has_head = "classification_head.pt" in files
                entry = {
                    "repo_id": repo_id,
                    "available": exists,
                    "status_code": code,
                    "has_classification_head": has_head,
                    "file_count": len(files),
                }
                results[split_type][model_name][epoch] = entry
                if exists:
                    results["summary"]["available_count"] += 1
                    print(f"  [OK] {split_type.upper():8s} {model_name:10s} epoch {epoch:4s} -> {repo_id} (head: {has_head})")
                else:
                    results["summary"]["missing_count"] += 1
                    results["summary"]["missing_repos"].append({
                        "split": split_type,
                        "model": model_name,
                        "epoch": epoch,
                        "repo_id": repo_id,
                        "status_code": code
                    })
                    print(f"  [FAIL] {split_type.upper():8s} {model_name:10s} epoch {epoch:4s} -> {repo_id} (HTTP {code})")

    out_file = Path(output_dir) / "checkpoint_availability_report.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Checkpoint Audit] Report saved to {out_file}")
    return results

if __name__ == "__main__":
    audit_all_checkpoints()
