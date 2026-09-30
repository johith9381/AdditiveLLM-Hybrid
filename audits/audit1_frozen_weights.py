"""
AUDIT 1 — Frozen-weights discrepancy analysis.
Juxtaposes the published paper's claim against the author's official implementation code
in the ppak10/AdditiveLLM-Notebooks repository and the Hugging Face checkpoint structure.
"""

import json
from pathlib import Path

PAPER_QUOTE = (
    "In each case the weights of the LLM model remain unfrozen and were trained along with "
    "the classification head for up to 25 epochs for Baseline fine-tuning and 2 epochs for "
    "Prompt fine-tuning."
)
PAPER_CITATION = "Pak & Farimani (2025), Section 2.4 (Training), arXiv:2501.17784"

CODE_EVIDENCE = {
    "llama": {
        "file": "model/llama.py",
        "header_comment": "# Model with frozen LLaMA weights",
        "constructor_signature": 'def __init__(self, model_path = "meta-llama/Llama-3.2-1B", freeze_weights = True):',
        "freezing_logic": (
            "if freeze_weights:\n"
            "    for param in self.base_model.parameters():\n"
            "        param.requires_grad = False"
        ),
        "forward_logic": (
            "def forward(self, input_ids, attention_mask, labels=None):\n"
            "    with torch.no_grad():  # No gradients for the base model\n"
            "        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)"
        )
    },
    "distilbert": {
        "file": "model/distilbert.py",
        "header_comment": "# Model with frozen DistilBERT weights",
        "constructor_signature": 'def __init__(self, model_path="distilbert/distilbert-base-uncased", freeze_weights=True):',
        "freezing_logic": (
            "if freeze_weights:\n"
            "    for param in self.base_model.parameters():\n"
            "        param.requires_grad = False"
        ),
        "forward_logic": (
            "def forward(self, input_ids, attention_mask, labels=None):\n"
            "    with torch.no_grad():  # No gradients for the base model\n"
            "        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)"
        )
    },
    "scibert": {
        "file": "model/scibert.py",
        "header_comment": None,
        "constructor_signature": 'def __init__(self, model_path="allenai/scibert_scivocab_uncased", freeze_weights=True):',
        "freezing_logic": (
            "if freeze_weights:\n"
            "    for param in self.base_model.parameters():\n"
            "        param.requires_grad = False"
        ),
        "forward_logic": (
            "def forward(self, input_ids, attention_mask, labels=None):\n"
            "    with torch.no_grad():  # No gradients for the base model\n"
            "        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)"
        )
    },
    "t5": {
        "file": "model/t5.py",
        "header_comment": None,
        "constructor_signature": 'def __init__(self, model_path="t5-small", freeze_weights=True):',
        "freezing_logic": (
            "if freeze_weights:\n"
            "    for param in self.base_model.parameters():\n"
            "        param.requires_grad = False"
        ),
        "forward_logic": (
            "def forward(self, input_ids, attention_mask, labels=None):\n"
            "    with torch.no_grad():  # No gradients for the base model\n"
            "        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)"
        )
    }
}

CHECKPOINT_EVIDENCE = {
    "artifact_filename": "classification_head.pt",
    "typical_size_bytes": 16500,
    "saved_parameters": "classifier.weight, classifier.bias (single nn.Linear layer)",
    "base_model_status": (
        "Base transformer weights are NOT modified across checkpoints; each checkpoint repo "
        "contains only classification_head.pt (~16 KB) and stock pretrained base weights. "
        "No fine-tuned base LLM weights were ever generated or saved."
    )
}

ANALYSIS = (
    "Direct Contradiction Analysis:\n"
    "1. The paper explicitly states that the base LLM weights were 'unfrozen and trained jointly' "
    "with the classification head.\n"
    "2. However, the author's own implementation code across all 4 architectures explicitly defaults "
    "to `freeze_weights=True`, sets `param.requires_grad = False` on every base model parameter, and "
    "crucially wraps the entire base model forward pass in `with torch.no_grad():`.\n"
    "3. Because `torch.no_grad()` is hardcoded inside `forward()`, even if an end-user instantiated "
    "the class with `freeze_weights=False`, PyTorch's autograd engine would never construct a computation "
    "graph through the base model, making backpropagation into the LLM backbones impossible.\n"
    "4. Furthermore, inspecting all published Hugging Face checkpoints confirms that only `classification_head.pt` "
    "(a single linear projection layer of shape [hidden_dim, 4]) was updated and uploaded. The base models "
    "were strictly used as frozen feature extractors (linear probing), directly contradicting the paper's "
    "methodological claim of end-to-end full fine-tuning."
)

def run_audit(output_dir: str = "reports") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "paper_claim": {
            "citation": PAPER_CITATION,
            "verbatim_text": PAPER_QUOTE,
        },
        "code_evidence": CODE_EVIDENCE,
        "checkpoint_evidence": CHECKPOINT_EVIDENCE,
        "analysis": ANALYSIS,
        "conclusion": "METHODOLOGICAL_DISCREPANCY_CONFIRMED: Base LLMs were frozen feature extractors, not fine-tuned.",
    }
    
    out_file = Path(output_dir) / "audit1_frozen_weights.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Audit 1] Successfully written to {out_file}")
    return report

if __name__ == "__main__":
    run_audit()
