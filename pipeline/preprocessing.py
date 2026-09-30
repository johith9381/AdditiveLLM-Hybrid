"""
Physics-based Defect Labeling and Data Augmentation Pipeline.
Implements the exact ground-truth equations and Lack-of-Fusion sweep from
AdditiveLLM (Pak & Farimani, 2025).
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Threshold constants per paper Section 2 & generate_data.ipynb
KEYHOLE_THRESHOLD = 1.5
BALLING_THRESHOLD = math.pi
LAYER_HEIGHT_RANGE = list(range(0, 1000, 50))  # 0 to 950 microns (20 values)
HATCH_SPACING_RANGE = list(range(0, 1000, 50))  # 0 to 950 microns (20 values)

DEFECT_COLUMNS = [
    "desirable",             # Index 0: None / Defect-free
    "defect_keyhole",         # Index 1: Keyhole porosity
    "defect_lack_of_fusion",  # Index 2: Lack of Fusion porosity
    "defect_balling",         # Index 3: Balling porosity
]

def check_keyhole(width: Optional[float], depth: Optional[float]) -> Optional[bool]:
    """
    Eq. 1 (Keyhole): Width / Depth > 1.5 -> AVOIDS keyhole.
    Returns True if sample violates criterion (i.e. has keyhole defect).
    """
    if width is None or depth is None or pd.isna(width) or pd.isna(depth) or depth <= 0:
        return None
    return bool((width / depth) <= KEYHOLE_THRESHOLD)

def check_lack_of_fusion(
    hatch_spacing: Optional[float],
    layer_height: Optional[float],
    width: Optional[float],
    depth: Optional[float]
) -> Optional[bool]:
    """
    Eq. 2 (Lack of Fusion): (hatch_spacing / width)^2 + (layer_height / depth)^2 <= 1 -> AVOIDS LoF.
    Returns True if sample violates criterion (i.e. has lack of fusion defect).
    """
    if (hatch_spacing is None or layer_height is None or width is None or depth is None or
        pd.isna(hatch_spacing) or pd.isna(layer_height) or pd.isna(width) or pd.isna(depth) or
        width <= 0 or depth <= 0):
        return None
    metric = (hatch_spacing / width) ** 2 + (layer_height / depth) ** 2
    return bool(metric > 1.0)

def check_balling(length: Optional[float], width: Optional[float]) -> Optional[bool]:
    """
    Eq. 3 (Balling): Length / Width < pi -> AVOIDS balling.
    Returns True if sample violates criterion (i.e. has balling defect).
    """
    if length is None or width is None or pd.isna(length) or pd.isna(width) or width <= 0:
        return None
    return bool((length / width) >= BALLING_THRESHOLD)

def derive_labels_from_geometry(
    width: Optional[float],
    depth: Optional[float],
    length: Optional[float],
    hatch_spacing: Optional[float],
    layer_height: Optional[float]
) -> List[int]:
    """
    Compute binary 4-label vector [None, Keyhole, Lack of Fusion, Balling]
    given melt pool dimensions and process parameters.
    """
    kh = 1 if check_keyhole(width, depth) else 0
    lof = 1 if check_lack_of_fusion(hatch_spacing, layer_height, width, depth) else 0
    bal = 1 if check_balling(length, width) else 0
    des = 1 if (kh == 0 and lof == 0 and bal == 0) else 0
    return [des, kh, lof, bal]

def format_baseline_text(
    material: str,
    power: Optional[float],
    velocity: Optional[float],
    beam_diameter: Optional[float],
    hatch_spacing: Optional[float],
    layer_height: Optional[float]
) -> str:
    """
    Exact baseline format:
    <material>[SEP]<power>[SEP]<velocity>[SEP]<beam_diameter>[SEP]<hatch_spacing>[SEP]<layer_height>
    Unknown/missing replaced with empty string, units included.
    """
    m = str(material) if pd.notna(material) else ""
    p = f"{power} W" if pd.notna(power) else ""
    v = f"{velocity} mm/s" if pd.notna(velocity) else ""
    b = f"{beam_diameter} microns" if pd.notna(beam_diameter) else ""
    h = f"{hatch_spacing} microns" if pd.notna(hatch_spacing) else ""
    l = f"{layer_height} microns" if pd.notna(layer_height) else ""
    return f"{m}[SEP]{p}[SEP]{v}[SEP]{b}[SEP]{h}[SEP]{l}"

def parse_baseline_text(text: str) -> Dict[str, str]:
    """
    Parse [SEP]-separated baseline text string into constituent parameter fields.
    """
    parts = text.split("[SEP]")
    return {
        "material": parts[0] if len(parts) > 0 else "",
        "power": parts[1] if len(parts) > 1 else "",
        "velocity": parts[2] if len(parts) > 2 else "",
        "beam_diameter": parts[3] if len(parts) > 3 else "",
        "hatch_spacing": parts[4] if len(parts) > 4 else "",
        "layer_height": parts[5] if len(parts) > 5 else "",
    }

def format_prompt_text(template: str, params: Dict[str, str]) -> str:
    """
    Embed parameters into natural language prompt template.
    """
    res = template
    for key, val in params.items():
        placeholder = f"<{key}>"
        res = res.replace(placeholder, val)
    return res
