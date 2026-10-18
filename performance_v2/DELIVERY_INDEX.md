# Performance V2 delivery

**Qidian (Martin) Xing.** Owner-directed and Codex-assisted; original team credit is preserved in [contributions](CONTRIBUTIONS.md). Work is confined to this extension. No paid compute was used.

The frozen two-model annual-rank blend achieves historical annual AP **0.373320**, pooled rank AP **0.382691**, pooled ROC **0.921602**, and annual Recall@5% **0.619824**. The strongest fixed same-cutoff baseline is V1 lag XGBoost at annual AP **0.302835**: a **+0.070485** point difference. Its conditional paired company-bootstrap interval is approximately **[0.002, 0.144]**; all four strong-baseline annual-AP intervals have positive lower bounds. These intervals condition on fitted models/current companies and do not cover complete model-selection or macro uncertainty.

| Five-seed prediction ensemble, fitted through 2014 | Annual AP | Pooled AP | Pooled ROC | Annual R@5% |
| --- | --- | --- | --- | --- |
| Fixed legacy train-local L1 residual | 0.262098 | 0.259424 | 0.862103 | 0.524208 |
| Fixed V1 lag XGBoost | 0.302835 | 0.296057 | 0.852542 | 0.513465 |
| Fixed V1 raw CatBoost | 0.279037 | 0.270969 | 0.852025 | 0.487553 |
| Fixed V1 selected Random Forest | 0.286383 | 0.286210 | 0.890616 | 0.503744 |
| Development-selected XGBoost recipe | 0.312815 | 0.324517 | 0.903908 | 0.540400 |
| Development-selected MLP representative | 0.392916 | 0.391599 | 0.912470 | 0.621462 |
| Development-frozen annual-rank blend | 0.373320 | 0.382691 | 0.921602 | 0.619824 |

Pooled AP/ROC of the rank blend use a different score convention from raw-score rows; matching rank references are retained in the [full comparison](results/baseline_bridge.csv). Annual AP and annual R@5% are the primary comparisons. The DL representative performs better than the blend on historical annual AP; it is not substituted after viewing history. The 2015–2018 period was previously inspected and is not a fresh holdout.

## Read and reproduce

- [English report, 7 pages](reports/Research_Report_V2_EN.pdf) and [Chinese performance report, 8 pages](reports/PERFORMANCE_DELTA_CN.pdf), both rendered and manually checked on all pages.
- [Chinese interview addendum](reports/Interview_Addendum_V2_CN.md), including evidence, uncertainty and attribution.
- [Freshly executed results notebook](notebooks/Performance_V2_Results.ipynb), [execution receipt](results/notebook_verification.json), [numerical verification](results/verification.json), and [all-pipeline source-order reloads](results/pipeline_row_alignment_reloads.json).
- [All logical trials](results/trials.csv), [every recorded attempt](results/trial_attempts.csv), [actual coverage and shortfalls](results/search_budget_summary.csv), [five-year selection trace](results/selection_trace.csv), [2014 confirmation](results/confirmation_2014.csv), [historical results](results/historical_2015_2018.csv), [five-seed results](results/seeds.csv), [paired intervals](results/paired_company_intervals.csv), and [costs](results/costs.csv).
- [Frozen model inventory](configs/final_inventory.json) and [local artifact verification](results/model_artifacts_status.json). All 47 fitted pipelines, preprocessing, metadata and neural states are saved locally under the ignored artifacts directory. Raw data, company-level predictions and models are excluded from public Git. The two English and two Chinese resume bullets remain in the private local file `reports/Resume_Bullets_V2_EN_CN.md` for owner review.
- [Protocol](V2_PROTOCOL.yaml), [execution notes](PROTOCOL_EXECUTION_NOTES.md), [sources and versions](REFERENCES.md), and [runtime/reproduction instructions](README.md).

## What the new experiments establish

The old and V1 networks differed in capacity, inputs, weighting, stopping and refit roles. Their AP implementations agree on real scores and tied-score fixtures. A fixed-2012 capacity control does not support a general bigger-is-better explanation: RAW18 residual AP is 0.249 at 192×4 and 0.259 at 64×2. Removing its duplicate raw column gives 0.325; removing full class weighting gives 0.364. These are local mechanism controls, not a causal decomposition of the entire historical version gap.

The clearest development-only single-factor evidence is three-year time decay: XGBoost AP 0.458→0.548 and MLP 0.382→0.493, on the same three scoring years and seed. Compact ratios and three-year history did not automatically help the same tree recipe; PLE helps some input/backbone pairs and hurts another. [Input controls](results/matched_input_controls.csv), [optimization controls](results/optimization_controls.csv), [embedding controls](results/numeric_matched_controls.csv), and [equal-update sampling evidence](results/training_budget_ablation.csv) retain scope and exact recipes.

The overall tree's raw pooled AP 0.324517 exceeds the archived legacy L1 observation 0.295332 descriptively. Cutoff, selection and seed roles differ, so that archive difference is not an algorithm-only gain. Actual fixed-13 original-loop retraining gives 0.224962; the included epoch31 base checkpoint gives 0.139890, and is a different artifact from the archived epoch23 model. Both archive-only cached-feature permutations were diagnosed, corrected without refitting, and independently reloaded; [receipts](results/archive_alignment_correction.json) and [base receipt](results/archive_base_alignment_correction.json) preserve the evidence.

## Limits and complete accounting

108 completed three-year screening configurations represent 107 effective recipes; 30 bounded local-refinement configurations also completed. CatBoost and XGBoost each completed 36 screens. RF, ExtraTrees, legacy residual, MLP, numeric MLP and TabM each completed 6; their larger target counts were not reached. Official FT and XGB ranking completed no screens. Two NN ranking-loss controls did run and their sampling sequences were reconstructed separately. Unfinished logical trials, invalidated ROS migration and failed attempts remain visible; they are not counted as completed models or silently removed.

The first eight-hour registration expired partial. A separate explicitly authorized continuation produced final results, preserving both clocks and the original partial snapshot. The [summary](results/experiment_summary.json) and [fit accounting](results/fit_accounting.csv) distinguish the windows; all 1,312 recorded fit starts fall inside authorized windows. No further model fitting is needed or running.

1,344 protected V0/V1 files matched the original manifest. macOS denied access to one private original image even for the read-only escalated check; that file's current equality is unverified, with exact details retained privately. No private originals are published.

Company novelty is sparse: 468 naturally unseen firms include only five failed firms, and single-class annual metrics remain NA. Company-group stress uses parameters selected on broader companies; it is post-selection stress, with mean pooled AP about 0.143 for ML and 0.138 for DL. It does not support a claim of robust prospective unseen-company performance. Company-constant retrospective labels and missing event/publication dates preclude verified annual PD or early-warning claims.
