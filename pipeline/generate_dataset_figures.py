"""
Generate Figure 5 (Material Distribution) and Figure 6 (Defect Distribution)
from official dataset parquets and raw data files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import ast
import pandas as pd
import numpy as np
from pipeline.figures import (
    plot_fig5_material_distribution,
    plot_fig6_defect_distribution,
    CLASS_NAMES
)

def parse_material_from_text(text: str) -> str:
    # Text format: "<Material>[SEP]<Power>..."
    return text.split("[SEP]")[0].strip()

def main():
    print("[Figures 5 & 6] Extracting material and defect statistics...")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Validation split
    val_df = pd.read_parquet("data/splits/validation_baseline.parquet")
    val_mats = [parse_material_from_text(t) for t in val_df["text"]]
    val_labels = [ast.literal_eval(l) if isinstance(l, str) else l for l in val_df["label"]]
    val_labels_arr = np.array(val_labels, dtype=np.int32)
    
    # 2. Test split
    test_df = pd.read_parquet("data/splits/test_baseline.parquet")
    test_mats = [parse_material_from_text(t) for t in test_df["text"]]
    test_labels = [ast.literal_eval(l) if isinstance(l, str) else l for l in test_df["label"]]
    test_labels_arr = np.array(test_labels, dtype=np.int32)
    
    # 3. Build material counts dictionary for Fig 5
    mat_counts = {
        "Validation": pd.Series(val_mats).value_counts().to_dict(),
        "Test": pd.Series(test_mats).value_counts().to_dict(),
    }
    plot_fig5_material_distribution(mat_counts, output_dir)
    
    # 4. Build defect counts dictionary for Fig 6
    defect_counts = {
        "Validation (72,475)": val_labels_arr.sum(axis=0).tolist(),
        "Test (108,714)": test_labels_arr.sum(axis=0).tolist(),
    }
    plot_fig6_defect_distribution(defect_counts, output_dir)
    print("[Figures 5 & 6] Generation complete!")

if __name__ == "__main__":
    main()
