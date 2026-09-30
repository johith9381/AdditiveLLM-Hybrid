# Comprehensive All-Tier Benchmark: Physics-First AdditiveLLM

### Format A: All Three Evaluation Tiers Side-by-Side (5-Seed Mean)

| Tier | Format | Accuracy_Pct | Macro_F1 | Macro_Precision | Macro_Recall | ECE | Sample_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tier A | Baseline | 83.53 | 0.6824 | 0.7351 | 0.7121 | 0.0604 | 1200 |
| Tier A | Prompt | 67.12 | 0.4207 | 0.5937 | 0.5151 | 0.2141 | 1200 |
| Tier B | Baseline | 83.92 | 0.6726 | 0.7183 | 0.7039 | 0.0606 | 1200 |
| Tier B | Prompt | 66.5 | 0.4286 | 0.5812 | 0.5423 | 0.2225 | 1200 |
| Tier C | Baseline | 66.06 | 0.2819 | 0.3333 | 0.3645 | 0.2045 | 1200 |
| Tier C | Prompt | 58.25 | 0.2267 | 0.2754 | 0.4215 | 0.3161 | 1200 |

### Format B: Genuine Pooled Overall Aggregate Metrics

| Evaluation_Scope | Format | Pooled_Accuracy_Pct | Pooled_Macro_F1 | Pooled_Macro_Precision | Pooled_Macro_Recall | Pooled_ECE | Total_Test_Samples_Pooled | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall (All Tiers Pooled: Tier A + B + C) | Baseline | 77.83 | 0.5497 | 0.5874 | 0.5496 | 0.0808 | 18000 | Genuine micro-pooled sample aggregation across seen, held-out groups, and held-out alloy IN625. |
| Overall (All Tiers Pooled: Tier A + B + C) | Prompt | 63.96 | 0.3881 | 0.5285 | 0.4673 | 0.2378 | 18000 | Genuine micro-pooled sample aggregation across seen, held-out groups, and held-out alloy IN625. |
