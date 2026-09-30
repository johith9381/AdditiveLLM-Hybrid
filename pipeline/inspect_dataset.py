"""
Inspection script to verify split row counts, dataset schema, and material values (Correction 1 & 2).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"
RAW_DIR = DATA_DIR / "raw"

def main():
    print("=== DATASET VERIFICATION & SCHEMA AUDIT ===")
    
    # 1. Row counts of splits
    val_b = pd.read_parquet(SPLITS_DIR / "validation_baseline.parquet")
    test_b = pd.read_parquet(SPLITS_DIR / "test_baseline.parquet")
    train_b = pd.read_parquet(SPLITS_DIR / "train_baseline.parquet")
    val_p = pd.read_parquet(SPLITS_DIR / "validation_prompt.parquet")
    test_p = pd.read_parquet(SPLITS_DIR / "test_prompt.parquet")
    
    print(f"Validation Baseline rows: {len(val_b):,} (Paper: 72,476)")
    print(f"Test Baseline rows:       {len(test_b):,} (Paper: 108,715)")
    print(f"Train Baseline rows:      {len(train_b):,} (Paper: 543,573)")
    print(f"Validation Prompt rows:   {len(val_p):,} (Paper: ~724,760)")
    print(f"Test Prompt rows:         {len(test_p):,}")

    print(f"Note on row counts: Published parquets have exactly 72,475 / 108,714 / 543,572 rows.")
    print(f"This is exactly 1 row less per split than the paper text (72,476 / 108,715 / 543,573) due to `data_df.iloc[1:]` in generate_dataset.py.")
    assert len(val_b) in [72475, 72476], f"Validation baseline mismatch: {len(val_b)}"
    assert len(test_b) in [108714, 108715], f"Test baseline mismatch: {len(test_b)}"
    assert len(train_b) in [543572, 543573], f"Train baseline mismatch: {len(train_b)}"
    print("[OK] Split sizes match author's published parquet files!")

    # 2. Check schema
    print("\n--- Schema Check (Correction 2) ---")
    print(f"Validation Baseline columns: {list(val_b.columns)}")
    print(f"Sample row 0:\n{val_b.iloc[0].to_dict()}")

    # 3. Check materials (Correction 1: IN615 vs IN625)
    print("\n--- Material Name Audit (Correction 1) ---")
    all_materials = set()
    for df in [val_b, test_b, train_b]:
        for text in df["text"]:
            parts = text.split("[SEP]")
            mat = parts[0].strip()
            all_materials.add(mat)

    print(f"Distinct materials found in Baseline dataset ({len(all_materials)} total):")
    sorted_mats = sorted(list(all_materials))
    for m in sorted_mats:
        print(f"  - '{m}'")

    has_in615 = "IN615" in all_materials
    has_in625 = "IN625" in all_materials
    print(f"\nVerbatim 'IN615' in dataset: {has_in615}")
    print(f"Verbatim 'IN625' in dataset: {has_in625}")

if __name__ == "__main__":
    main()
