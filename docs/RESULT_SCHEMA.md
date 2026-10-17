# Result schema and numerical conventions

## Selection and evaluation

`AP` is `sklearn.metrics.average_precision_score`, not trapezoidal PR area. `macro_year_AP` gives each available scoring year equal weight; development selection uses the seven registered scoring years. Pooled AP weights the empirical row population implicitly. `prevalence` must be read alongside AP, particularly across years/cohorts.

`ROC_AUC` is ordinary binary ROC area. Precision, recall, F1, TN/FP/FN/TP and alert_fraction in final calibrated rows use the model-specific threshold selected only on 2014. Uncalibrated rows use 0.5 as a descriptive reference. No historical threshold optimization occurs.

Annual budgets select `ceil(budget * year_rows)` by descending score with an ascending SHA256(company|year) tie key, independent of labels. Precision@budget is positives selected / selected rows; recall is positives selected / all positive rows in that year; lift divides precision by the same year's prevalence. `macro_*_top1/5/10` are equal-weight annual averages, not pooled review policies. Unsupported single-class/zero-positive quantities remain unavailable rather than receiving fabricated good scores.

Brier and log_loss evaluate probability prediction quality, not pure calibration error. Reliability bins retain count, positive_rows, mean_score and observed_rate. Platt fitting uses natural-prevalence 2013 records; 2014 is a separate decision/check role and its selected F1 is not an independent final estimate.

## Configuration and information

`config_id` identifies a recipe, information set and hash of the complete configuration. Run labels additionally identify calendar folds, seed or company holdout. `row_hash` hashes the sorted source row_id set and makes coverage differences detectable. Model comparison on different row sets is not a controlled improvement estimate.

S0 = 17 deduplicated current amounts; S1 = registered financial structure/signed transforms; S2 = S1 plus exact-calendar causal history; LAGS = exact three-calendar-year snapshots/masks. Candidate feature count differs from actual tensor dimensions after train-only constant/duplicate checks and missing indicators.

For the fixed rank ensemble, internal `feature_set=ALL` denotes the helper union container built for component dispatch. Only each frozen component's saved input-column list is consumed. In this run both selected components use S1; the ensemble does not gain new dynamic information merely because the helper container can construct it. Annual rank scoring requires an annual reference cohort.

## Cost and fit-call accounting

`fit_seconds` is pipeline fitting wall time, including training-role preprocessing, nested selection, stopping/refitting and intermediate checkpoint writes. Feature construction, independent calibration, final pipeline serialization and report generation are outside that timer. Final inference timers include model transformation/scoring/calibration with financial features already constructed. Development cost summaries average unique fold runs; AP selection averages scoring years.

Neural `epochs` is the fixed full-refit length selected inside training. Stop histories retain the actual stopping length and best-AP trajectory. Tree `epochs` denotes fitted boosting rounds/tree count; Logistic cost labels denote actual solver iterations and coefficient counts include the intercept. Tree node counts are not represented as directly comparable neural parameter counts.

The research ledger counts actual model fitting attempts, including inner selectors, neural/boosting stopping refits, calibration and the failed first real-data pilot. Analytical prevalence/Altman scoring and preprocessing operations are not additional estimator-training calls. Synthetic diagnostic/unit classifier-fit attempts are separately recorded and included conservatively in the 600-call local model-fit limit; one-update gradient contract checks are disclosed separately and never produce research scores. A fit start event is not alone a success certificate: use per-run completion/prediction records.

## Uncertainty and novelty

Paired company bootstrap carries all a company's annual records together using shared integer multiplicities across models. Percentile intervals are conditional on the fitted models and score sample, not a full repetition of training/search/calibration. Invalid single-class draws are counted. Rank-ensemble intervals additionally hold the original annual rank-reference maps fixed rather than rebuilding a new scoring universe.

Seen/unseen is relative to base-model fitting through 2012; some such companies can appear later in the global calibration/decision roles. The 2014 cutoff is a sensitivity definition. Neither is called an untouched external cohort. The separate three-group untuned stress analysis excludes held-out companies from all fitting, stopping, selector and calibration roles.

Seeds describe stochastic pipeline variation on the same historical observations; they are not independent test sets. Primary seed42 was fixed in advance. `results/CLAIMS.csv` connects numeric report entries to source tables, row filters, protocol and run labels. The complete row predictions and formal checkpoint states remain in ignored local artifacts.
