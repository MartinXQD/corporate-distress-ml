# Methods and evaluation contract

This project classifies records carrying an ultimate company status and ranks those records within each year. A firm-year is an accounting observation, not a distinct failure event.

## Data and features

The authorized snapshot has 78,682 records, 8,971 companies, and 18 raw financial fields across 1999–2018. There are 5,220 failed-labeled rows belonging to 609 companies. X9 and X16 are exact duplicates. Each company's status is constant over time. Preserve negative earnings and other valid negative accounts. Ratios, signed logs, and exact t−1/t−2 calendar joins are implemented in [features](../src/distress_v2/data.py) and [shared utilities](../src/distress_ml/features.py). Calendar gaps remain missing; future values never fill them.

The selected tree uses LEGACY33: the inherited financial construction with duplicate-derived columns removed, native tree preprocessing, and train-role missing-value treatment. The selected PyTorch MLP uses the 17 deduplicated raw fields, train-only 1%/99% clipping and standardization, width 128, two layers, dropout 0.15, AdamW, and no resampling or class weighting. Both use a three-year exponential half-life. Full parameters are in [the recipe index](../configs/final_model.json).

## Calendar roles

| Role | Records / scoring years | Purpose |
| --- | --- | --- |
| Initial screening | 2009, 2011, 2013; fit strictly earlier years | Bounded broad search |
| Shortlist stability and choice | Every year 2009–2013; seeds 42, 2026, 31415 | Recipe and blend selection |
| Confirmation | 2014, once after selection freeze | Confirmation without reopening search |
| Final refit | Permitted records through 2014 | Five seeds: 42, 2026, 31415, 7, 2718 |
| Historical evaluation | 2015–2018; 12,282 records | Previously inspected retrospective comparison |

For each train role, early stopping uses its latest training year. Preprocessing is fit on the appropriate inner training rows, then refit on the full permitted window. Neural full-window refitting uses the selected optimizer-update count. Tree refitting uses the selected boosting length. The model does not ingest company IDs, year, target, source row IDs, or future lifecycle summaries as financial predictors.

## Frozen score construction

Average the five raw prediction vectors within each selected family. Convert each family mean to percentile ranks separately within each evaluation year. Average the two ranks with weights 0.5 and 0.5. Annual ranks depend on the contemporaneous cohort and are not calibrated probabilities; any operational use would need a defined scoring cohort.

## Metrics and comparisons

Mean annual AP is the equally weighted average of the four annual sklearn average-precision scores. Pooled AP and ROC-AUC combine all years. Recall@Top 5% selects ceil(0.05 × yearly rows), with deterministic company/year tie ordering, then averages yearly recall. Mean seed-level metrics and metrics of mean predictions are separate estimands.

The strongest fixed same-cutoff reference is identified by annual AP among four preregistered strong baselines, all refit through 2014. It is not the old 0.206 pooled-AP course-extension observation. Original-role archive rows, cutoff-2012 transfers, and same-cutoff comparisons retain separate channels in [the unmodified ledger](../results/historical_2015_2018.csv).

Paired company bootstrap uses 2,000 shared company resamples and the same fixed year set. Resamples missing a required class do not silently change the annual estimand. The positive conditional AP interval against the strongest baseline is approximately [0.002, 0.144]; it excludes full HPO, distribution shift, label-timing, and macro uncertainty.

## Negative results and corrections

All logical trials, actual attempts, incomplete family coverage, invalidated migration runs, and numerical corrections remain available. Oversampling is compared with equal optimizer updates and separately with equal epochs. Two archive-only feature-cache order errors were corrected without refitting or choosing new models; the records distinguish the actual included checkpoint from an earlier archived score. See [research history](RESEARCH_HISTORY.md) and [execution notes](research_origin/search_record/PROTOCOL_EXECUTION_NOTES.md).
