"""
Paired Statistical Significance Testing for AdditiveLLM Hybrid vs Reproduced Baselines.
Performs paired t-test and Wilcoxon signed-rank test across 5 seeds for all tiers and formats.
Saves to results/significance_test.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def run_significance_tests():
    runs_file = RESULTS_DIR / "multiseed_runs_detailed.csv"
    if not runs_file.exists():
        print(f"[Significance Test] Detailed runs file not found at {runs_file}")
        return

    df_runs = pd.read_csv(runs_file)
    print("=" * 80)
    print("  PAIRED STATISTICAL SIGNIFICANCE TESTING (HYBRID VS CONTROLLED BASELINES)")
    print("=" * 80)

    hybrid_runs = df_runs[df_runs["model"] == "Hybrid AdditiveLLM"]
    baseline_models = ["DistilBERT", "SciBERT", "T5-Small"]
    tiers = ["Tier A", "Tier B", "Tier C"]
    formats = ["Baseline", "Prompt"]
    metrics = ["accuracy_pct", "macro_f1"]

    test_results = []

    for tier in tiers:
        for fmt in formats:
            h_sub = hybrid_runs[(hybrid_runs["tier"] == tier) & (hybrid_runs["format"] == fmt)].sort_values("seed")
            
            for b_model in baseline_models:
                b_sub = df_runs[(df_runs["model"] == b_model) & (df_runs["tier"] == tier) & (df_runs["format"] == fmt)].sort_values("seed")
                if len(h_sub) != 5 or len(b_sub) != 5:
                    continue

                for metric in metrics:
                    h_vals = h_sub[metric].values
                    b_vals = b_sub[metric].values
                    diffs = h_vals - b_vals

                    # Paired t-test
                    t_stat, t_pval = stats.ttest_rel(h_vals, b_vals)

                    # Wilcoxon signed-rank test
                    try:
                        w_stat, w_pval = stats.wilcoxon(h_vals, b_vals)
                    except Exception:
                        w_stat, w_pval = np.nan, np.nan

                    mean_diff = float(np.mean(diffs))
                    h_mean = float(np.mean(h_vals))
                    b_mean = float(np.mean(b_vals))
                    is_sig = bool(t_pval < 0.05 and mean_diff > 0)

                    test_results.append({
                        "Tier": tier,
                        "Format": fmt,
                        "Baseline_Model": b_model,
                        "Metric": "Accuracy (%)" if metric == "accuracy_pct" else "Macro F1",
                        "Hybrid_Mean": round(h_mean, 4 if "f1" in metric else 2),
                        "Baseline_Mean": round(b_mean, 4 if "f1" in metric else 2),
                        "Mean_Difference": round(mean_diff, 4 if "f1" in metric else 2),
                        "Paired_t_Statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else "N/A",
                        "Paired_t_PValue": f"{t_pval:.4e}" if not np.isnan(t_pval) else "N/A",
                        "Wilcoxon_PValue": f"{w_pval:.4e}" if not np.isnan(w_pval) else "N/A",
                        "Statistically_Significant_p05": is_sig,
                        "Conclusion": "Hybrid Statistically Superior (p < 0.05)" if is_sig else (
                            "No Significant Difference (p >= 0.05)" if mean_diff >= 0 else "Baseline Superior"
                        )
                    })

    df_sig = pd.DataFrame(test_results)
    out_csv = RESULTS_DIR / "significance_test.csv"
    df_sig.to_csv(out_csv, index=False)
    print(f"\n[Significance Test] Evaluated {len(df_sig)} paired test conditions.")
    print(f"Results successfully saved to {out_csv}")
    return df_sig


if __name__ == "__main__":
    run_significance_tests()
