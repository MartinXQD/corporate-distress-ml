# Verified aggregate research results

Start with [the readable model comparison](MODEL_COMPARISON.md), [CSV](model_comparison.csv), and [annual performance figure](figures/annual_comparison.png). The table is a labeled view of the unchanged historical source ledger; it does not refit or recompute model scores.

| Evidence | Files |
| --- | --- |
| Historical models and original-role context | [Full scored-model table](historical_2015_2018.csv), [annual rows](historical_by_year.csv), [baseline bridge](baseline_bridge.csv) |
| Selection and confirmation | [Selection trace](selection_trace.csv), [development blend grid](ensemble_development.csv), [2014 confirmation](confirmation_2014.csv) |
| Inputs and optimization | [Matched inputs](matched_input_controls.csv), [optimization](optimization_controls.csv), [embeddings](numeric_matched_controls.csv), [training budgets](training_budget_ablation.csv) |
| Search accounting, including failures | [Logical trials](trials.csv), [attempts](trial_attempts.csv), [coverage](trial_coverage.csv), [fit accounting](fit_accounting.csv) |
| Variation and stress | [Seeds](seeds.csv), [paired company intervals](paired_company_intervals.csv), [company stress](company_stress_summary.csv), [seen/unseen](seen_unseen.csv) |
| Original numerical verification | [Verification receipt](verification.json), [all 47 pipeline reloads](pipeline_row_alignment_reloads.json), [model artifact status](model_artifacts_status.json) |
| Current release verification | [Public checks](release_verification.json), [notebook execution](notebook_execution.json) |

Original receipts refer to the exact saved run and are preserved without being recast as newly executed private-model checks. Old partial-publication status files are historical snapshots, superseded by the final experiment summary and current release checks. Private data, individual predictions, and serialized models are excluded.
