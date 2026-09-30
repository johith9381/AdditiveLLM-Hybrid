"""
AdditiveLLM Group-Safe Data Pipeline.
Implements:
1. Group A Physical Key extraction: (material, power, velocity, beam_diameter).
2. Group B Augmentation Axes: (hatch_spacing, layer_height).
3. Zero-leakage GroupShuffleSplit on Group A keys BEFORE augmentation.
4. Three distinct evaluation tiers:
   - Tier A: In-Distribution (held-out sweep within seen groups)
   - Tier B: Held-Out Physical Groups (unseen physical parameters)
   - Tier C: Held-Out Materials (unseen material, true OOD)
5. Physics-consistency auxiliary targets (w/d, l/w) from FLOW-3D and MeltpoolNet.
6. On-the-fly Prompt template sampling in PyTorch Dataset / DataLoader.
"""

import os
import math
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

# Physical constants and defect thresholds per paper
KEYHOLE_THRESHOLD = 1.5      # w/d <= 1.5 -> Keyhole defect
BALLING_THRESHOLD = math.pi  # l/w >= pi -> Balling defect
LOF_THRESHOLD = 1.0          # (h/w)^2 + (t/d)^2 > 1.0 -> Lack of Fusion defect

DEFECT_NAMES = ["None", "Keyhole", "Lack of Fusion", "Balling"]

# Material catalog mapping
KNOWN_MATERIALS = [
    "SS316L", "Ti-6Al-4V", "IN718", "SS17-4PH", "IN625", "IN738LC",
    "Hastelloy X", "Cu10Sn", "AlSi10Mg", "Al-2.5Fe", "Tungsten",
    "Al-C-Co-Fe-Mn-Ni", "Ti-49Al-2Cr-2Nb", "HCP Cu", "Invar36",
    "SS304", "WE43", "MS1-", "CMSX-4", "TiC/Inconel 718", "SS304L",
    "Ti6242", "K403 superalloy", "Ti-45Al", "4140 steel", "AA7075",
    "Ni-5Nb", "Co-Cr-Fe-Mn-Ni", "Zn-2Al", "S31603"
]
MATERIAL_TO_IDX = {mat: i for i, mat in enumerate(KNOWN_MATERIALS)}

# Typical parameter normalizations (mean, std)
PARAM_STATS = {
    "power": (180.0, 100.0),          # W
    "velocity": (1000.0, 600.0),       # mm/s
    "beam_diameter": (90.0, 30.0),     # microns
    "hatch_spacing": (120.0, 60.0),    # microns
    "layer_height": (40.0, 20.0),      # microns
}


def parse_baseline_text(text: str) -> Dict[str, str]:
    """Parse baseline [SEP] text into constituent process parameter strings."""
    parts = text.split("[SEP]")
    return {
        "material": parts[0].strip() if len(parts) > 0 else "",
        "power": parts[1].strip() if len(parts) > 1 else "",
        "velocity": parts[2].strip() if len(parts) > 2 else "",
        "beam_diameter": parts[3].strip() if len(parts) > 3 else "",
        "hatch_spacing": parts[4].strip() if len(parts) > 4 else "",
        "layer_height": parts[5].strip() if len(parts) > 5 else "",
    }


def clean_numeric_param(val_str: str, unit: str) -> Optional[float]:
    """Strip unit string and parse float value."""
    if not val_str:
        return None
    cleaned = val_str.replace(unit, "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numeric_features(parsed: Dict[str, str]) -> Tuple[np.ndarray, int]:
    """
    Extract normalized continuous features [P, v, d, h, t] and material index.
    """
    p = clean_numeric_param(parsed["power"], "W")
    v = clean_numeric_param(parsed["velocity"], "mm/s")
    b = clean_numeric_param(parsed["beam_diameter"], "microns")
    h = clean_numeric_param(parsed["hatch_spacing"], "microns")
    l = clean_numeric_param(parsed["layer_height"], "microns")

    # Impute defaults if missing
    p_val = p if p is not None else PARAM_STATS["power"][0]
    v_val = v if v is not None else PARAM_STATS["velocity"][0]
    b_val = b if b is not None else PARAM_STATS["beam_diameter"][0]
    h_val = h if h is not None else PARAM_STATS["hatch_spacing"][0]
    l_val = l if l is not None else PARAM_STATS["layer_height"][0]

    # Z-score normalization
    norm_features = np.array([
        (p_val - PARAM_STATS["power"][0]) / PARAM_STATS["power"][1],
        (v_val - PARAM_STATS["velocity"][0]) / PARAM_STATS["velocity"][1],
        (b_val - PARAM_STATS["beam_diameter"][0]) / PARAM_STATS["beam_diameter"][1],
        (h_val - PARAM_STATS["hatch_spacing"][0]) / PARAM_STATS["hatch_spacing"][1],
        (l_val - PARAM_STATS["layer_height"][0]) / PARAM_STATS["layer_height"][1],
    ], dtype=np.float32)

    mat_str = parsed["material"]
    mat_idx = MATERIAL_TO_IDX.get(mat_str, len(MATERIAL_TO_IDX))  # OOD index if unknown

    return norm_features, mat_idx


def load_raw_geometry_lookup(raw_dir: Path) -> Dict[Tuple[str, float, float], Tuple[float, float, float]]:
    """
    Build unified lookup table of (material, power, velocity) -> (width, depth, length)
    from MeltpoolNet experimental data and FLOW-3D simulations.
    """
    lookup = {}

    # 1. MeltpoolNet geometry
    geom_file = raw_dir / "meltpoolgeometry.csv"
    if geom_file.exists():
        df_geom = pd.read_csv(geom_file)
        for _, r in df_geom.iterrows():
            mat = str(r["Material"]).strip()
            p = float(r["Power"]) if pd.notna(r["Power"]) else None
            v = float(r["Velocity"]) if pd.notna(r["Velocity"]) else None
            w = float(r["width of melt pool"]) if pd.notna(r["width of melt pool"]) else None
            d = float(r["depth of meltpool"]) if pd.notna(r["depth of meltpool"]) else None
            l = float(r["length of melt pool"]) if pd.notna(r["length of melt pool"]) else None
            if p is not None and v is not None:
                lookup[(mat, round(p, 1), round(v, 1))] = (w, d, l)

    # 2. Ti64 FLOW-3D simulations
    ti_file = raw_dir / "dimensions_ti64.pkl"
    if ti_file.exists():
        with open(ti_file, "rb") as f:
            df_ti = pickle.load(f)
        for _, r in df_ti.iterrows():
            p = float(r["power"])
            v = float(r["velocity"]) * 1000.0  # m/s to mm/s
            w = float(r["widths_avg"]) if "widths_avg" in r and pd.notna(r["widths_avg"]) else None
            d = float(r["depths_avg"]) if "depths_avg" in r and pd.notna(r["depths_avg"]) else None
            l = float(r["lengths_avg"]) if "lengths_avg" in r and pd.notna(r["lengths_avg"]) else None
            lookup[("Ti-6Al-4V", round(p, 1), round(v, 1))] = (w, d, l)

    # 3. SS316L simulations
    ss_file = raw_dir / "dimensions_ss316l_10_micron.csv"
    if ss_file.exists():
        df_ss = pd.read_csv(ss_file)
        for _, r in df_ss.iterrows():
            p = float(r["power"])
            v = float(r["velocity"]) * 1000.0
            w = float(r["avg_width"]) * 1e6 if pd.notna(r["avg_width"]) else None
            d = float(r["avg_depth"]) * 1e6 if pd.notna(r["avg_depth"]) else None
            l = float(r["avg_length"]) * 1e6 if pd.notna(r["avg_length"]) else None
            lookup[("SS316L", round(p, 1), round(v, 1))] = (w, d, l)

    return lookup


def compute_physics_auxiliary_targets(
    material: str,
    power: float,
    velocity: float,
    labels: List[int],
    lookup: Dict[Tuple[str, float, float], Tuple[float, float, float]]
) -> Tuple[float, float]:
    """
    Compute auxiliary regression targets:
    r1 = Width / Depth (w/d)
    r2 = Length / Width (l/w)
    Guarantees physics consistency with defect ground truth:
    - Keyhole (label[1] == 1) => w/d <= 1.5; Conduction mode => w/d > 1.5
    - Balling (label[3] == 1) => l/w >= pi; Stable mode => l/w < pi
    """
    key = (material, round(power, 1), round(velocity, 1))
    w, d, l = lookup.get(key, (None, None, None))

    is_keyhole = bool(labels[1] == 1)
    is_balling = bool(labels[3] == 1)

    # Calculate or estimate r1 (w/d)
    if w is not None and d is not None and d > 0:
        r1 = w / d
    else:
        # Calibrated analytical estimate based on linear energy density P / v
        linear_energy = power / (velocity + 1e-3)
        if is_keyhole:
            # Deep keyhole: w/d <= 1.5, scales inversely with energy density
            r1 = max(0.5, min(1.45, 1.45 - (linear_energy - 0.25) * 0.8))
        else:
            # Conduction regime: w/d > 1.5
            r1 = min(3.5, max(1.55, 1.55 + 0.5 / (linear_energy + 0.1)))

    # Ensure r1 strictly respects keyhole defect label
    if is_keyhole and r1 > 1.5:
        r1 = 1.45
    elif not is_keyhole and r1 <= 1.5:
        r1 = 1.55

    # Calculate or estimate r2 (l/w)
    if l is not None and w is not None and w > 0:
        r2 = l / w
    else:
        # Plateau-Rayleigh capillary elongation ratio
        speed_factor = velocity / 1000.0
        if is_balling:
            # Elongated cylinder: l/w >= pi
            r2 = max(math.pi + 0.05, math.pi + 0.5 * speed_factor)
        else:
            # Stable melt pool: l/w < pi
            r2 = min(math.pi - 0.05, max(1.2, 1.5 + 0.8 * speed_factor))

    # Ensure r2 strictly respects balling defect label
    if is_balling and r2 < math.pi:
        r2 = math.pi + 0.05
    elif not is_balling and r2 >= math.pi:
        r2 = math.pi - 0.05

    return float(r1), float(r2)


def format_baseline_row(mat: str, p: float, v: float, b: float, h: float, l: float) -> str:
    """Format structured [SEP] text string."""
    m_s = f"{mat}"
    p_s = f"{p:.1f} W" if p is not None else ""
    v_s = f"{v:.1f} mm/s" if v is not None else ""
    b_s = f"{b:.1f} microns" if b is not None else ""
    h_s = f"{h:.1f} microns" if h is not None else ""
    l_s = f"{l:.1f} microns" if l is not None else ""
    return f"{m_s}[SEP]{p_s}[SEP]{v_s}[SEP]{b_s}[SEP]{h_s}[SEP]{l_s}"


def format_prompt_template(template: str, params: Dict[str, str]) -> str:
    """Embed parameter dictionary into natural language template."""
    res = template
    for key in ["material", "power", "velocity", "beam_diameter", "hatch_spacing", "layer_height"]:
        val = params.get(key, "")
        res = res.replace(f"<{key}>", val)
    return res


class GroupSafeDataset(Dataset):
    """
    PyTorch Dataset with On-The-Fly Prompt Template Sampling.
    Maintains base un-expanded dataset in memory (~724k or split size).
    Supports dynamic random template sampling per step during training,
    or held-out evaluation templates during inference.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        templates: Optional[List[str]] = None,
        mode: str = "baseline",  # "baseline", "prompt_dynamic", "prompt_fixed"
        fixed_template_idx: int = 0,
        lookup: Optional[Dict] = None
    ):
        self.df = df.reset_index(drop=True)
        self.templates = templates if templates is not None else []
        self.mode = mode
        self.fixed_template_idx = fixed_template_idx
        self.lookup = lookup if lookup is not None else {}

        # Pre-parse parameters and labels for speed
        self.samples = []
        for idx, row in self.df.iterrows():
            text = row["text"]
            parsed = parse_baseline_text(text)
            lbl = row["label"]
            if isinstance(lbl, str):
                import ast
                lbl = ast.literal_eval(lbl)
            lbl = list(lbl)

            norm_feat, mat_idx = extract_numeric_features(parsed)
            p_val = clean_numeric_param(parsed["power"], "W") or 180.0
            v_val = clean_numeric_param(parsed["velocity"], "mm/s") or 1000.0
            r1, r2 = compute_physics_auxiliary_targets(parsed["material"], p_val, v_val, lbl, self.lookup)

            self.samples.append({
                "baseline_text": text,
                "parsed": parsed,
                "numeric_features": norm_feat,
                "material_idx": mat_idx,
                "labels": np.array(lbl, dtype=np.float32),
                "aux_targets": np.array([r1, r2], dtype=np.float32),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        if self.mode == "baseline":
            text = item["baseline_text"]
        elif self.mode in ("prompt_dynamic", "prompt"):
            # On-the-fly random template sampling from available pool
            tmpl = random.choice(self.templates) if self.templates else item["baseline_text"]
            text = format_prompt_template(tmpl, item["parsed"])
        elif self.mode == "prompt_fixed":
            idx_tmpl = self.fixed_template_idx % len(self.templates) if self.templates else 0
            tmpl = self.templates[idx_tmpl] if self.templates else item["baseline_text"]
            text = format_prompt_template(tmpl, item["parsed"])
        elif self.mode == "mixed":
            # 50% baseline structured text, 50% dynamic prompt templates
            if random.random() < 0.5:
                text = item["baseline_text"]
            else:
                tmpl = random.choice(self.templates) if self.templates else item["baseline_text"]
                text = format_prompt_template(tmpl, item["parsed"])
        else:
            text = item["baseline_text"]

        return {
            "text": text,
            "numeric_features": torch.tensor(item["numeric_features"], dtype=torch.float32),
            "material_idx": torch.tensor(item["material_idx"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.float32),
            "aux_targets": torch.tensor(item["aux_targets"], dtype=torch.float32),
        }


def partition_group_safe_tiers(
    data_dir: str = "data",
    output_dir: str = "data/splits_group_safe",
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Constructs the zero-leakage Group-Aware dataset split:
    1. Loads all 724,761 samples across official parquet splits.
    2. Identifies Group A physical keys (material, power, velocity, beam_diameter).
    3. Tier C: Completely holds out 'IN625' (unseen material, true OOD).
    4. Applies GroupShuffleSplit on remaining Group A keys:
       - 80% Train Groups
       - 20% Tier B Groups (unseen physical parameter combinations).
    5. Within Train Groups:
       - 85% Train samples
       - 15% Tier A samples (In-Distribution held-out hatch/layer sweep within seen groups).
    6. Saves clean parquets for Train, Tier A, Tier B, Tier C.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(data_dir) / "splits"
    print("[Data Pipeline] Loading published baseline parquets...")
    train_df = pd.read_parquet(splits_dir / "train_baseline.parquet")
    val_df = pd.read_parquet(splits_dir / "validation_baseline.parquet")
    test_df = pd.read_parquet(splits_dir / "test_baseline.parquet")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total_rows = len(all_df)
    print(f"[Data Pipeline] Total pooled samples: {total_rows:,}")

    # Extract Group A physical key
    def extract_group_a(text: str) -> Tuple[str, str, str, str]:
        p = parse_baseline_text(text)
        return (p["material"], p["power"], p["velocity"], p["beam_diameter"])

    all_df["group_a_key"] = all_df["text"].apply(extract_group_a)
    all_df["material"] = all_df["group_a_key"].apply(lambda k: k[0])

    # Tier C: Held-out material (IN625)
    print("[Data Pipeline] Partitioning Tier C (Held-Out Material: IN625)...")
    tier_c_df = all_df[all_df["material"] == "IN625"].copy().reset_index(drop=True)
    rem_df = all_df[all_df["material"] != "IN625"].copy().reset_index(drop=True)

    # GroupShuffleSplit on remaining Group A keys
    print("[Data Pipeline] Executing zero-leakage GroupShuffleSplit on Group A physical keys...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_group_idx, tier_b_group_idx = next(gss.split(rem_df, groups=rem_df["group_a_key"]))

    train_groups_df = rem_df.iloc[train_group_idx].copy().reset_index(drop=True)
    tier_b_df = rem_df.iloc[tier_b_group_idx].copy().reset_index(drop=True)

    # Within train groups, reserve 15% for Tier A (In-Distribution held-out sweep)
    # Guarantee that every Group A key in Tier A is also represented in Train
    train_rows = []
    tier_a_rows = []
    
    # Group by group_a_key to ensure clean within-group splitting
    for key, group in train_groups_df.groupby("group_a_key"):
        n_samples = len(group)
        if n_samples == 1:
            train_rows.append(group)
        else:
            # At least 1 row to Tier A, at least 1 row to Train
            n_val = max(1, int(round(n_samples * 0.15)))
            n_val = min(n_val, n_samples - 1)
            shuffled = group.sample(frac=1.0, random_state=seed)
            tier_a_rows.append(shuffled.iloc[:n_val])
            train_rows.append(shuffled.iloc[n_val:])

    train_final_df = pd.concat(train_rows, ignore_index=True)
    tier_a_df = pd.concat(tier_a_rows, ignore_index=True)

    print(f"[Data Pipeline] Partition complete:")
    print(f"  Training Split : {len(train_final_df):,} rows ({train_final_df['group_a_key'].nunique()} unique Group A keys)")
    print(f"  Tier A (In-Dist): {len(tier_a_df):,} rows ({tier_a_df['group_a_key'].nunique()} unique Group A keys)")
    print(f"  Tier B (Held-Gp): {len(tier_b_df):,} rows ({tier_b_df['group_a_key'].nunique()} unique Group A keys)")
    print(f"  Tier C (Held-Mat): {len(tier_c_df):,} rows ({tier_c_df['group_a_key'].nunique()} unique Group A keys)")

    # Save to disk
    train_final_df.to_parquet(out_path / "train.parquet")
    tier_a_df.to_parquet(out_path / "tier_a.parquet")
    tier_b_df.to_parquet(out_path / "tier_b.parquet")
    tier_c_df.to_parquet(out_path / "tier_c.parquet")

    splits = {
        "train": train_final_df,
        "tier_a": tier_a_df,
        "tier_b": tier_b_df,
        "tier_c": tier_c_df,
    }
    print(f"[Data Pipeline] All splits successfully saved to {out_path}")
    return splits


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    out_dir = data_dir / "splits_group_safe"
    partition_group_safe_tiers(str(data_dir), str(out_dir))
