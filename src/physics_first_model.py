"""
AdditiveLLM Physics-First Hydrodynamic Dual-Pathway Architecture.
Replaces brittle categorical material embeddings with a physics-first formulation:
1. Multi-modal process encoder (DistilBERT + SciBERT domain adapter + continuous physics MLP).
2. Auxiliary Hydrodynamic Geometry Engine predicting:
   - r1 = Width / Depth (w/d)
   - r2 = Length / Width (l/w)
   - Volumetric / Linear Energy Densities
3. Physically-Constrained Residual Defect Classification:
   - Keyhole driven by physical violation: ReLU(1.5 - r1)
   - Balling driven by Rayleigh instability: ReLU(r2 - pi)
   - Conduction (None) driven by stable bounds: ReLU(r1 - 1.5) * ReLU(pi - r2)
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContinuousProcessPhysicsEncoder(nn.Module):
    """
    Encodes continuous process parameters into physical dimensionless spaces
    without learning any discrete material-identity lookup table.
    """
    def __init__(self, in_features: int = 5, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        # Input features: [power, velocity, beam_diameter, hatch_spacing, layer_height]
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, num_feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(num_feats)


class PhysicsFirstAdditiveLLM(nn.Module):
    """
    Physics-First Hybrid Architecture where defect classification is driven
    by continuous physical representations and explicit hydrodynamic geometry constraints.
    """
    def __init__(
        self,
        base_models_dir: str,
        hidden_dim: int = 256,
        num_classes: int = 4,
        freeze_backbones: bool = True
    ):
        super().__init__()
        base_path = Path(base_models_dir)

        # Expert 1: DistilBERT text encoder (frozen / linear probe per paper findings)
        self.distilbert = AutoModel.from_pretrained(str(base_path / "distilbert"), local_files_only=True)
        # Expert 2: SciBERT scientific text encoder
        self.scibert = AutoModel.from_pretrained(str(base_path / "scibert"), local_files_only=True)

        if freeze_backbones:
            for p in self.distilbert.parameters():
                p.requires_grad = False
            for p in self.scibert.parameters():
                p.requires_grad = False

        # Continuous physics encoder (material-identity free)
        self.physics_encoder = ContinuousProcessPhysicsEncoder(in_features=5, hidden_dim=64, out_dim=64)

        # Cross-Modal Fusion Projection (DistilBERT 768 + SciBERT 768 + Physics 64 = 1600 -> hidden_dim)
        self.fusion_projector = nn.Sequential(
            nn.Linear(768 + 768 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Auxiliary Hydrodynamic Geometry Engine:
        # Predicts continuous melt pool aspect ratios: [r1 = w/d, r2 = l/w]
        self.geometry_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

        # Standard Multi-Label Defect Head
        self.latent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes)
        )

        # Physics-Coupled Residual Weights (learnable physical scaling factors)
        # gamma[0] -> None, gamma[1] -> Keyhole, gamma[2] -> LoF, gamma[3] -> Balling
        self.physics_gamma = nn.Parameter(torch.tensor([1.2, 1.8, 1.2, 1.8], dtype=torch.float32))

    def extract_text_pooled(self, encoder: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_embeddings = torch.sum(hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids_distil: torch.Tensor,
        attention_mask_distil: torch.Tensor,
        input_ids_sci: torch.Tensor,
        attention_mask_sci: torch.Tensor,
        numeric_features: torch.Tensor,
        cached_distil: torch.Tensor = None,
        cached_sci: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Text representations (use cached if available for ultra-fast iteration)
        if cached_distil is not None and cached_sci is not None:
            e_distil = cached_distil
            e_sci = cached_sci
        else:
            e_distil = self.extract_text_pooled(self.distilbert, input_ids_distil, attention_mask_distil)
            e_sci = self.extract_text_pooled(self.scibert, input_ids_sci, attention_mask_sci)

        # 2. Continuous Physics Representation (no material lookup table)
        e_phys = self.physics_encoder(numeric_features)

        # 3. Fused Cross-Modal Representation
        fused_raw = torch.cat([e_distil, e_sci, e_phys], dim=-1)
        fused = self.fusion_projector(fused_raw)

        # 4. Predict Auxiliary Melt Pool Hydrodynamic Aspect Ratios:
        # r1 = w/d, r2 = l/w
        pred_geometry = self.geometry_regressor(fused)
        r1_pred = pred_geometry[:, 0]
        r2_pred = pred_geometry[:, 1]

        # 5. Physics Violation Residuals:
        # Keyhole threshold: w/d <= 1.5
        keyhole_violation = F.relu(1.5 - r1_pred)
        # Balling threshold: l/w >= pi
        balling_violation = F.relu(r2_pred - math.pi)
        # Stable conduction threshold: w/d > 1.5 and l/w < pi
        conduction_stability = F.relu(r1_pred - 1.5) * F.relu(math.pi - r2_pred)
        # Lack of fusion proxy: shallow depth / low energy
        lof_indicator = F.relu(r1_pred - 2.5)

        physics_residuals = torch.stack([
            conduction_stability,
            keyhole_violation,
            lof_indicator,
            balling_violation
        ], dim=-1)

        # 6. Combined Physics-First Classification Logits:
        # Logits = Latent_Classifier(Fused) + gamma * Physics_Residuals
        latent_logits = self.latent_classifier(fused)
        final_logits = latent_logits + self.physics_gamma * physics_residuals

        return {
            "logits": final_logits,
            "latent_logits": latent_logits,
            "pred_geometry": pred_geometry,
            "physics_residuals": physics_residuals,
            "fused_features": fused
        }
