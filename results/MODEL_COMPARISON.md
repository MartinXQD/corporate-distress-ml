# Historical model comparison

All rows below fit through 2014; five-seed prediction means are used except the deterministic one-seed logistic reference. The frozen blend averages annual ranks. Other rows use raw score means. Annual AP and recall remain comparable; pooled scores have different conventions. Historical records were previously inspected.

| Model | Annual AP | Pooled AP | Pooled ROC-AUC | Annual R@5% | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen rank blend | 0.373320 | 0.382691 | 0.921602 | 0.619824 | 5 + 5 |
| Selected MLP | 0.392916 | 0.391599 | 0.912470 | 0.621462 | 5 |
| Selected XGBoost | 0.312815 | 0.324517 | 0.903908 | 0.540400 | 5 |
| Fixed lag-feature XGBoost | 0.302835 | 0.296057 | 0.852542 | 0.513465 | 5 |
| Fixed Random Forest | 0.286383 | 0.286210 | 0.890616 | 0.503744 | 5 |
| Fixed raw-input CatBoost | 0.279037 | 0.270969 | 0.852025 | 0.487553 | 5 |
| Fixed legacy residual / train-local L1 | 0.262098 | 0.259424 | 0.862103 | 0.524208 | 5 |
| Fixed residual MLP / corrected steps | 0.228299 | 0.214204 | 0.873665 | 0.462027 | 5 |
| Logistic reference | 0.169428 | 0.160701 | 0.846505 | 0.385656 | 1 |

[Every evaluated pipeline and score channel](model_comparison.csv) · [Unchanged source table](historical_2015_2018.csv) · [Matching annual-rank reference rows](baseline_bridge.csv)

The selected MLP has higher historical annual AP than the blend. The official model remains the blend chosen on development evidence; see [selection record](../docs/MODEL_SELECTION.md). Earlier source observations, different training cutoffs, post-selection group stress, and seed-level rows are preserved separately and are not interchangeable improvement denominators.
