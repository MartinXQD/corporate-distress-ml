# Performance V2 - partial budget closeout

Qidian (Martin) Xing | NOT a completed performance upgrade. The original UTC budget expired; no final model or new historical score is claimed.

| Recipe/version | Fit to | Input/scale | Sel | Seeds | Year AP | Pool AP | ROC | R@5 | Min | Identity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1/P2 RF | 08/10/12 | V1_S1/v1 | N/A | 1 | 0.418 | 0.411 | 0.871 | 0.397 | 0.753 | DEV 09/11/13 |
| V0/P2 inner L1 | 08/10/12 | INNER_L1/std01 | N/A | 1 | 0.225 | 0.209 | 0.785 | 0.260 | 3.195 | DEV 09/11/13 |
| V1/P2 lag XGB | 08/10/12 | V1_LAGS/v1 | N/A | 1 | 0.232 | 0.220 | 0.786 | 0.270 | 0.311 | DEV 09/11/13 |
| V1/P2 Cat raw | 08/10/12 | V1_S0/v1 | N/A | 1 | 0.207 | 0.088 | 0.660 | 0.252 | 0.296 | DEV 09/11/13 |
| V1/P2 DL | 08/10/12 | V1_S1/v1 | N/A | 1 | 0.252 | 0.250 | 0.821 | 0.279 | 0.763 | DEV 09/11/13 |
| V2 MLP raw | 08/10/12 | RAW17/std01 | N/A | 1 | 0.382 | 0.357 | 0.830 | 0.380 | 3.948 | DEV 09/11/13 |
| V2 old Res raw | 08/10/12 | RAW17/std01 | N/A | 1 | 0.370 | 0.347 | 0.827 | 0.358 | 19.336 | DEV 09/11/13 |
| V2 catboost | 08/10/12 | SLOG_RATIO/native | N/A | 1 | 0.318 | 0.309 | 0.828 | 0.319 | 1.295 | DEV 09/11/13 |
| V2 xgboost | 08/10/12 | LAGS_RAW3/native | N/A | 1 | 0.329 | 0.320 | 0.836 | 0.326 | 0.277 | DEV 09/11/13 |
| V2 extra_trees | 08/10/12 | RATIO_CORE/none | N/A | 1 | 0.343 | 0.337 | 0.845 | 0.344 | 0.650 | DEV 09/11/13 |
| V2 tabm | 08/10/12 | RAW17/std01 | N/A | 1 | 0.323 | 0.303 | 0.822 | 0.328 | 36.234 | DEV 09/11/13 |

The strongest completed development baseline remains V1/P2 RF (0.418 mean annual AP), versus 0.382 for the plain raw-input MLP. These are seed42, three-year development results. They do not establish an improvement on 2015-2018.

All rows use scoring years 2009/2011/2013 and fit cutoffs 2008/2010/2012. Pooled scores combine different fold-trained models. R@5 averages annual recall. No final representative was selected. This is the same-protocol comparison currently available, not the requested final cutoff-2014 benchmark.

# 2. What the legacy bridge actually established

| Old archive identity | Historical AP | Historical ROC |
| --- | --- | --- |
| base_scaled | 0.145 | 0.835 |
| lasso_selected_scaled | 0.295 | 0.858 |

| Controlled 2012 recipe | AP | Seconds |
| --- | --- | --- |
| xgb_raw_keepbest | 0.239 | 0.654 |
| legacy192_s1 | 0.206 | 22.082 |
| legacy192_raw18 | 0.249 | 59.504 |
| xgb_LAGS_RAW3 | 0.333 | 6.668 |
| legacy192_inner_l1 | 0.214 | 170.366 |
| xgb_RATIO_CORE | 0.291 | 6.149 |
| xgb_RAW17 | 0.285 | 1.123 |
| legacy192_epochrefit | 0.256 | 62.872 |
| legacy192_raw17 | 0.325 | 196.214 |
| legacy192_clip005 | 0.314 | 157.399 |
| legacy192_quantile | 0.243 | 26.896 |
| legacy64_raw18 | 0.259 | 18.227 |
| legacy192_keepbest | 0.263 | 37.591 |
| legacy192_unweighted | 0.364 | 117.674 |
| legacy192_ros_steps | 0.392 | 141.263 |
| legacy192_ros_epochs | 0.417 | 234.856 |

Archive scores are old observations, not newly reproduced results. Legacy L1 AP 0.295332 was not the old validation-ROC-selected representative. The legacy residual has width192/four blocks and 302,209 parameters at 18 inputs; selected V1 DL has 36,801 parameters and 18 refit epochs. Capacity, inputs, clipping, loss weights and refit roles all changed. AP implementations agree on saved scores and ties.

Single-year/seed controls suggest that preprocessing, weighting and sampling can change results, but do not prove the cause of the historical gap. More capacity was not uniformly better. The included epoch31 checkpoint is not the archived epoch23 run.

# 3. Real coverage and resource limits

| Family | Complete | Target | Shortfall |
| --- | --- | --- | --- |
| catboost | 7 | 36 | 29 |
| xgboost | 7 | 36 | 29 |
| random_forest | 2 | 12 | 10 |
| extra_trees | 2 | 12 | 10 |
| legacy_residual | 1 | 18 | 17 |
| mlp | 1 | 18 | 17 |
| num_mlp | 1 | 14 | 13 |
| tabm | 1 | 14 | 13 |
| official_ft | 0 | 8 | 8 |
| xgb_rank | 0 | 6 | 6 |

22 complete screening trials represent 21 effective distinct recipes; the no-embedding numerical MLP duplicates the plain MLP. There are 3 complete local-refinement trials, 19 single-year bridge controls and 267 recorded fit calls. No-embedding TabM completed one full 200-epoch-ceiling configuration (development AP 0.323). PLE, full TabM tuning, FT and ranking performance searches were not completed.

ROS/SMOTE controls received 2,000 updates each; exposures and learning curves are saved. The incorrect migrated V1 ROS refit was invalidated and replaced. An RF timeout inference based on UTC gaps was retracted. Monotonic completed-fold timings omit gaps and interrupted work; they are not total CPU-seconds. The causes of the large UTC gaps remain unverified.

The last recorded model fit started at 2026-10-17T12:16:50.137483+00:00; deadline 2026-10-17T14:53:18+00:00. No fit was initiated after that deadline. Closeout documentation was prepared after the window. Operational deviations, including the per-component rather than combined-fold timeout, are disclosed in PROTOCOL_EXECUTION_NOTES.md.

# 4. What remains incomplete

Full numerical-embedding and paired TabM performance search

Target CatBoost/XGBoost and other family search coverage

Five-year, three-seed shortlist stability and final selection

One frozen 2014 confirmation batch

Same-cutoff 2014 final fits with five registered seeds

New frozen 2015-2018 historical batch and paired company intervals

New company-generalization stress tests and saved final model pipelines

Final publication checks and authenticated GitHub branch push

All 1345 protected originals matched their initial hashes. No new final pipeline was saved, and the GitHub branch was not pushed because final experimental/validation gates are unmet.

Source, all logical trials, interrupted attempts, development predictions and curves remain local. A new compute window requires fresh owner authorization; the expired clock must not be reset. The original team remains credited in the parent acknowledgements; this extension is owner-directed and Codex-assisted. Company-constant labels and missing event/publication dates preclude verified one-year PD claims. Official implementation sources and versions are in REFERENCES.md.
