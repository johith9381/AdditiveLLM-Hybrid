"""
AdditiveLLM v3: Physics-Informed Multi-Expert Hybrid Architecture.
Implements:
1. Thermophysical Property Injection (k, A, Tm, rho, Cp, alpha)
2. Hydrodynamic & Dimensionless Feature Engineering (VED, LED, AED, w/d, LoF ratio, l/w, beta, Pe)
3. Stage 1 Closed-Form Analytical Physics Baseline
4. Supplementary Experts: CatBoost and TabPFN
5. Cluster-Aware Focal Loss (4-cluster distribution-aware vs 3-cluster thermophysical control)
6. Stage 2 Residual Stacking Meta-Learner
"""

import os
import sys
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_pipeline import (
    parse_baseline_text,
    extract_numeric_features,
    KNOWN_MATERIALS,
    MATERIAL_TO_IDX,
    PARAM_STATS
)
from src.evaluate import compute_ece

DEFECT_CLASSES = ["None", "Keyhole", "Lack_of_Fusion", "Balling"]
SEEDS = [13, 42, 123, 2024, 7]

# Ambient temperature in Kelvin
T_0 = 293.15

# Thermophysical Properties Lookup Table
# Sourced from peer-reviewed databases (see results/physics_properties_sources.csv)
THERMOPHYSICAL_PROPERTIES = {
    "SS316L": {
        "k": 16.3, "A": 0.42, "Ts": 1648, "Tl": 1673, "Tm": 1660.0,
        "rho": 7950.0, "Cp": 500.0, "cluster_4": 0, "cluster_3": 0
    },
    "Ti-6Al-4V": {
        "k": 6.7, "A": 0.38, "Ts": 1878, "Tl": 1923, "Tm": 1900.0,
        "rho": 4430.0, "Cp": 526.0, "cluster_4": 1, "cluster_3": 1
    },
    "IN718": {
        "k": 11.4, "A": 0.40, "Ts": 1533, "Tl": 1609, "Tm": 1571.0,
        "rho": 8190.0, "Cp": 435.0, "cluster_4": 0, "cluster_3": 0
    },
    "IN625": {
        "k": 9.8, "A": 0.65, "Ts": 1563, "Tl": 1623, "Tm": 1593.0,
        "rho": 8440.0, "Cp": 410.0, "cluster_4": 2, "cluster_3": 0
    },
    "AlSi10Mg": {
        "k": 150.0, "A": 0.18, "Ts": 830, "Tl": 873, "Tm": 851.0,
        "rho": 2670.0, "Cp": 910.0, "cluster_4": 3, "cluster_3": 2
    },
    "Hastelloy X": {
        "k": 9.1, "A": 0.41, "Ts": 1533, "Tl": 1628, "Tm": 1580.0,
        "rho": 8220.0, "Cp": 486.0, "cluster_4": 0, "cluster_3": 0
    },
    "SS17-4PH": {
        "k": 17.9, "A": 0.43, "Ts": 1677, "Tl": 1713, "Tm": 1695.0,
        "rho": 7800.0, "Cp": 460.0, "cluster_4": 0, "cluster_3": 0
    },
    "IN738LC": {
        "k": 10.5, "A": 0.39, "Ts": 1503, "Tl": 1583, "Tm": 1543.0,
        "rho": 8110.0, "Cp": 420.0, "cluster_4": 0, "cluster_3": 0
    },
    "Cu10Sn": {
        "k": 64.0, "A": 0.22, "Ts": 1103, "Tl": 1273, "Tm": 1188.0,
        "rho": 8780.0, "Cp": 380.0, "cluster_4": 3, "cluster_3": 2
    },
    "Invar36": {
        "k": 10.1, "A": 0.44, "Ts": 1698, "Tl": 1703, "Tm": 1700.0,
        "rho": 8050.0, "Cp": 515.0, "cluster_4": 0, "cluster_3": 0
    },
    "SS304": {
        "k": 16.2, "A": 0.42, "Ts": 1673, "Tl": 1727, "Tm": 1700.0,
        "rho": 7930.0, "Cp": 500.0, "cluster_4": 0, "cluster_3": 0
    },
    "SS304L": {
        "k": 16.2, "A": 0.42, "Ts": 1673, "Tl": 1727, "Tm": 1700.0,
        "rho": 7930.0, "Cp": 500.0, "cluster_4": 0, "cluster_3": 0
    },
    "4140 steel": {
        "k": 42.6, "A": 0.45, "Ts": 1689, "Tl": 1789, "Tm": 1739.0,
        "rho": 7850.0, "Cp": 473.0, "cluster_4": 0, "cluster_3": 0
    },
    "Co-Cr-Fe-Mn-Ni": {
        "k": 14.8, "A": 0.39, "Ts": 1610, "Tl": 1670, "Tm": 1640.0,
        "rho": 8020.0, "Cp": 450.0, "cluster_4": 0, "cluster_3": 0
    },
    "DEFAULT": {
        "k": 15.0, "A": 0.40, "Ts": 1600, "Tl": 1650, "Tm": 1625.0,
        "rho": 7800.0, "Cp": 500.0, "cluster_4": 0, "cluster_3": 0
    }
}


def get_material_props(material_name: str) -> Dict[str, float]:
    """Retrieve thermophysical properties with standard fallback."""
    mat_clean = material_name.strip()
    if mat_clean in THERMOPHYSICAL_PROPERTIES:
        return THERMOPHYSICAL_PROPERTIES[mat_clean]
    # Check partial matching
    for k in THERMOPHYSICAL_PROPERTIES:
        if k in mat_clean or mat_clean in k:
            return THERMOPHYSICAL_PROPERTIES[k]
    return THERMOPHYSICAL_PROPERTIES["DEFAULT"]


def compute_stage1_analytical_physics(
    power: float,
    velocity: float,
    beam_diameter: float,
    hatch_spacing: float,
    layer_height: float,
    material_name: str
) -> Dict[str, float]:
    """
    Computes Stage 1 closed-form analytical physics baseline.
    Evaluates first-principles thermal and hydrodynamic geometry equations:
    - Width/Depth ratio (w/d) -> Keyhole criterion (Eq. 1)
    - (h/w)^2 + (t/d)^2 -> Lack of Fusion criterion (Eq. 2)
    - Length/Width ratio (l/w) -> Balling criterion (Eq. 3)
    Outputs rule-based class probabilities with ZERO learned training bias.
    """
    props = get_material_props(material_name)
    k = props["k"]
    A = props["A"]
    Tm = props["Tm"]
    rho = props["rho"]
    Cp = props["Cp"]
    alpha = k / (rho * Cp + 1e-4)  # m^2/s

    v_mps = max(velocity * 1e-3, 1e-4)  # mm/s -> m/s
    sigma = max(beam_diameter * 1e-6 / 2.0, 1e-6)  # beam radius in m
    h_m = max(hatch_spacing * 1e-6, 1e-6)
    t_m = max(layer_height * 1e-6, 1e-6)

    # Rosenthal / Normalized Enthalpy melt pool geometry estimation
    delta_T = max(Tm - T_0, 100.0)
    vol_heat_cap = rho * Cp  # J/(m^3 K)

    # 1. Estimated width (m -> um) (Rosenthal 3D conduction width + beam spot lower bound)
    w_m_sq = (8.0 * A * power) / (math.pi * math.e * vol_heat_cap * delta_T * v_mps + 1e-6) + (sigma * 2.0)**2
    w_m = math.sqrt(max(w_m_sq, 1e-12))
    w_um = w_m * 1e6

    # Normalized Enthalpy beta (Rubenchik & King, 2018)
    beta = (A * power) / (math.pi * vol_heat_cap * delta_T * math.sqrt(alpha * v_mps * (sigma ** 3) + 1e-18) + 1e-6)

    # 2. Estimated depth (m -> um)
    # Conduction depth: semi-ellipsoidal pool with d ~ w / 2.1
    # Keyhole transition: vapor recoil depression expands depth when beta > 6.0
    keyhole_factor = 1.0 + 1.2 * max(0.0, beta - 6.0)
    d_m = (w_m / 2.1) * keyhole_factor
    d_um = d_m * 1e6

    # 3. Estimated length (m -> um)
    # Elongation ratio L/W governed by scan velocity and thermal relaxation
    # Balling instability occurs when Rayleigh-Plateau limit L/W > pi (~3.14) is exceeded
    elongation = 1.8 + 0.6 * (v_mps / 1.0) * math.sqrt(max(0.1, 100.0 / (power + 1e-4)))
    l_m = w_m * elongation
    l_um = l_m * 1e6

    # Ratios
    w_over_d = w_um / (d_um + 1e-3)
    lof_geom = (hatch_spacing / (w_um + 1e-3)) ** 2 + (layer_height / (d_um + 1e-3)) ** 2
    l_over_w = l_um / (w_um + 1e-3)

    # Peclet Number
    Pe = (v_mps * sigma) / (alpha + 1e-8)

    # Energy densities
    led = power / (velocity + 1e-4)  # J/mm
    ved = power / (velocity * (hatch_spacing * 1e-3) * (layer_height * 1e-3) + 1e-4)  # J/mm^3
    aed = power / (velocity * (hatch_spacing * 1e-3) + 1e-4)

    # First-Principles Probabilities (calibrated to LPBF physical thresholds)
    # Keyhole: beta > 6.0 or w/d < 1.0 (vapor cavity deep penetration)
    p_kh_raw = 1.0 / (1.0 + math.exp(-3.0 * (beta - 6.5))) + 1.0 / (1.0 + math.exp(5.0 * (w_over_d - 1.1)))
    p_keyhole = min(0.98, max(0.01, p_kh_raw / 2.0))

    # Lack of Fusion: lof_geom > 1.0 (insufficient overlap between adjacent tracks and layers)
    p_lof = min(0.98, max(0.01, 1.0 / (1.0 + math.exp(-4.0 * (lof_geom - 1.15)))))

    # Balling: l/w > pi (Rayleigh-Plateau cylinder breakup at high elongation)
    p_balling = min(0.98, max(0.01, 1.0 / (1.0 + math.exp(-4.0 * (l_over_w - math.pi)))))

    # Conduction (None): joint absence of defect modes
    p_defect = max(p_keyhole, p_lof, p_balling)
    p_none = max(0.02, 1.0 - p_defect)

    # Normalize probabilities
    p_total = p_none + p_keyhole + p_lof + p_balling
    p_none /= p_total
    p_keyhole /= p_total
    p_lof /= p_total
    p_balling /= p_total

    return {
        "w_um": w_um,
        "d_um": d_um,
        "l_um": l_um,
        "w_over_d": w_over_d,
        "lof_geom": lof_geom,
        "l_over_w": l_over_w,
        "beta": beta,
        "Pe": Pe,
        "led": led,
        "ved": ved,
        "aed": aed,
        "k": k,
        "A": A,
        "Tm": Tm,
        "rho": rho,
        "Cp": Cp,
        "alpha": alpha,
        "p_none": p_none,
        "p_keyhole": p_keyhole,
        "p_lof": p_lof,
        "p_balling": p_balling,
        "cluster_4": props["cluster_4"],
        "cluster_3": props["cluster_3"]
    }


def extract_v3_feature_vectors(
    df: pd.DataFrame,
    include_beta_pe: bool = True,
    include_physics_props: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts complete v3 feature vectors:
    - X_features: continuous + physics features
    - y_labels: binary multi-label ground truth (None, Keyhole, LoF, Balling)
    - stage1_probs: [p_none, p_keyhole, p_lof, p_balling] from Stage 1 analytical physics
    - cluster_ids: material cluster IDs for cluster-aware focal loss
    """
    features = []
    labels = []
    stage1_probs_list = []
    cluster_ids = []

    for _, r in df.iterrows():
        text = r["text"]
        p = parse_baseline_text(text)
        nf, mat_idx = extract_numeric_features(p)

        power = nf[0] * PARAM_STATS["power"][1] + PARAM_STATS["power"][0]
        vel = nf[1] * PARAM_STATS["velocity"][1] + PARAM_STATS["velocity"][0]
        beam = nf[2] * PARAM_STATS["beam_diameter"][1] + PARAM_STATS["beam_diameter"][0]
        hatch = nf[3] * PARAM_STATS["hatch_spacing"][1] + PARAM_STATS["hatch_spacing"][0]
        layer = nf[4] * PARAM_STATS["layer_height"][1] + PARAM_STATS["layer_height"][0]
        mat_name = p.get("material", "DEFAULT")

        # Stage 1 analytical physics computation
        s1 = compute_stage1_analytical_physics(power, vel, beam, hatch, layer, mat_name)

        # Base 11 continuous features + material index (v2 baseline)
        p_over_b = power / (beam + 1e-4)
        aspect_h = hatch / (beam + 1e-4)
        aspect_t = layer / (beam + 1e-4)

        row_feat = [
            nf[0], nf[1], nf[2], nf[3], nf[4],
            float(mat_idx),
            s1["led"] / 2.0,
            s1["ved"] / 100.0,
            s1["aed"] / 10.0,
            p_over_b / 5.0,
            aspect_h,
            aspect_t
        ]

        if include_physics_props:
            # Add thermophysical properties and geometric ratios (Part 1 original spec)
            row_feat.extend([
                s1["k"] / 50.0,
                s1["A"],
                s1["Tm"] / 1800.0,
                s1["rho"] / 8000.0,
                s1["Cp"] / 600.0,
                s1["w_over_d"] / 2.0,
                s1["lof_geom"],
                s1["l_over_w"] / 3.0
            ])

            if include_beta_pe:
                # Add normalized enthalpy and Peclet number (Row 2 extension)
                row_feat.extend([
                    min(s1["beta"] / 10.0, 5.0),
                    min(s1["Pe"] / 50.0, 5.0)
                ])

        features.append(row_feat)
        stage1_probs_list.append([s1["p_none"], s1["p_keyhole"], s1["p_lof"], s1["p_balling"]])
        cluster_ids.append(s1["cluster_4"])

        lbl = r["label"]
        if isinstance(lbl, str):
            import ast
            lbl = ast.literal_eval(lbl)
        labels.append(lbl)

    return (
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(stage1_probs_list, dtype=np.float32),
        np.array(cluster_ids, dtype=np.int64)
    )
