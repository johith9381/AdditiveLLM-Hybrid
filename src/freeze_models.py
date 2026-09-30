"""
Freeze and Serialize Model A (PGHP) and Model B (v2 Hybrid).
Saves all model binaries, weights, thresholds, and configurations into:
- checkpoints/pghp_ensemble/ (Model A: Primary Deployment)
- checkpoints/v2_hybrid_tierc/ (Model B: Best Zero-Shot Tier C)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SPLITS_DIR = BASE_DIR / "data" / "splits_group_safe"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

PGHP_CKPT_DIR = CHECKPOINTS_DIR / "pghp_ensemble"
V2_CKPT_DIR = CHECKPOINTS_DIR / "v2_hybrid_tierc"

PGHP_CKPT_DIR.mkdir(parents=True, exist_ok=True)
V2_CKPT_DIR.mkdir(parents=True, exist_ok=True)

from src.pghp_pipeline import extract_pghp_features_and_labels
from src.push_accuracy_suite import extract_features_and_labels as extract_v2_features_and_labels

SEEDS = [13, 42, 123, 2024, 7]
DEFECT_CLASSES = ["None", "Keyhole", "Lack_of_Fusion", "Balling"]

def freeze_model_a_pghp(clean_train_df: pd.DataFrame):
    print("=" * 60)
    print("FREEZING MODEL A: PGHP (Primary Deployment Stack)")
    print("=" * 60, flush=True)

    best_params_lgb = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    optimal_thresholds = [0.50, 0.62, 0.50, 0.65]

    for s_idx, s in enumerate(SEEDS):
        print(f"  Training & Serializing PGHP Seed {s} ({s_idx + 1}/5)...", flush=True)
        sub_s = clean_train_df.sample(n=min(150000, len(clean_train_df)), random_state=s).reset_index(drop=True)
        X_s, y_s, _ = extract_pghp_features_and_labels(sub_s)

        seed_dir = PGHP_CKPT_DIR / f"seed_{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for c_idx, c_name in enumerate(DEFECT_CLASSES):
            cl = lgb.LGBMClassifier(**best_params_lgb, random_state=s + c_idx, verbosity=-1, n_jobs=4)
            cl.fit(X_s, y_s[:, c_idx])
            joblib.dump(cl, seed_dir / f"lgb_class_{c_name}.joblib")

            cx = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.85,
                                   colsample_bytree=0.80, random_state=s + c_idx, n_jobs=4, verbosity=0)
            cx.fit(X_s, y_s[:, c_idx])
            joblib.dump(cx, seed_dir / f"xgb_class_{c_name}.joblib")

    # Save PGHP config
    pghp_config = {
        "model_name": "Physics-Grounded Hybrid Predictor (PGHP)",
        "version": "1.0-frozen-deployment",
        "seeds": SEEDS,
        "num_features": 20,
        "features": [
            "norm_power", "norm_velocity", "norm_beam_diameter", "norm_hatch_spacing", "norm_layer_height",
            "material_idx", "linear_energy_density_LED", "volumetric_energy_density_VED", "areal_energy_density_AED",
            "power_to_beam_ratio", "hatch_to_beam_aspect_ratio", "layer_to_beam_aspect_ratio",
            "rosenthal_w_um_norm", "rosenthal_d_um_norm", "rosenthal_l_um_norm", "normalized_enthalpy_beta",
            "peclet_number_Pe", "physics_w_over_d_ratio", "physics_l_over_w_ratio", "physics_lof_geom_metric"
        ],
        "ensemble_weights": {"lightgbm": 0.55, "xgboost": 0.45},
        "optimal_thresholds": optimal_thresholds,
        "defect_classes": DEFECT_CLASSES,
        "training_samples_per_seed": 150000,
        "frozen_checkpoint_dir": str(PGHP_CKPT_DIR)
    }
    with open(PGHP_CKPT_DIR / "model_config.json", "w") as f:
        json.dump(pghp_config, f, indent=2)
    print("  -> PGHP Model Config and 5-seed weights saved successfully!")


def freeze_model_b_v2_tierc(clean_train_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FREEZING MODEL B: v2 Hybrid (Best Zero-Shot Tier C Stack)")
    print("=" * 60, flush=True)

    best_params_lgb = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    optimal_thresholds = [0.50, 0.62, 0.50, 0.65]

    for s_idx, s in enumerate(SEEDS):
        print(f"  Training & Serializing v2 Hybrid Seed {s} ({s_idx + 1}/5)...", flush=True)
        sub_s = clean_train_df.sample(n=min(150000, len(clean_train_df)), random_state=s).reset_index(drop=True)
        X_s, y_s = extract_v2_features_and_labels(sub_s)

        seed_dir = V2_CKPT_DIR / f"seed_{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for c_idx, c_name in enumerate(DEFECT_CLASSES):
            cl = lgb.LGBMClassifier(**best_params_lgb, random_state=s + c_idx, verbosity=-1, n_jobs=4)
            cl.fit(X_s, y_s[:, c_idx])
            joblib.dump(cl, seed_dir / f"lgb_class_{c_name}.joblib")

            cx = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.85,
                                   colsample_bytree=0.80, random_state=s + c_idx, n_jobs=4, verbosity=0)
            cx.fit(X_s, y_s[:, c_idx])
            joblib.dump(cx, seed_dir / f"xgb_class_{c_name}.joblib")

    # Save v2 Hybrid config
    v2_config = {
        "model_name": "v2 5-Branch Black-Box Hybrid Ensemble",
        "version": "2.0-frozen-tierc",
        "seeds": SEEDS,
        "num_features": 12,
        "features": [
            "norm_power", "norm_velocity", "norm_beam_diameter", "norm_hatch_spacing", "norm_layer_height",
            "material_idx", "linear_energy_density_LED", "volumetric_energy_density_VED", "areal_energy_density_AED",
            "power_to_beam_ratio", "hatch_to_beam_aspect_ratio", "layer_to_beam_aspect_ratio"
        ],
        "ensemble_weights": {"lightgbm": 0.55, "xgboost": 0.45},
        "optimal_thresholds": optimal_thresholds,
        "defect_classes": DEFECT_CLASSES,
        "training_samples_per_seed": 150000,
        "frozen_checkpoint_dir": str(V2_CKPT_DIR)
    }
    with open(V2_CKPT_DIR / "model_config.json", "w") as f:
        json.dump(v2_config, f, indent=2)
    print("  -> v2 Hybrid Model Config and 5-seed weights saved successfully!")


def main():
    print("Loading clean training data...")
    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    clean_train_df = df_train.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Cleaned training samples: {len(clean_train_df):,}")

    freeze_model_a_pghp(clean_train_df)
    freeze_model_b_v2_tierc(clean_train_df)
    print("\n" + "=" * 60)
    print("BOTH MODELS SERIALIZED AND FROZEN SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    main()
