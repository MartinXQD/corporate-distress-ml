# Financial Distress - Performance V2

**Qidian (Martin) Xing.** An owner-directed, Codex-assisted performance extension of the earlier team project; see [parent acknowledgements](../../../ACKNOWLEDGEMENTS.md) and [contribution statement](CONTRIBUTIONS.md).

The development-selected fixed annual-rank blend has historical annual AP 0.373, versus 0.303 for the strongest frozen same-cutoff baseline point estimate (baseline_v1_lags_xgb). The difference is +0.070. It exceeds all four annual-AP baseline point estimates. All four paired annual-AP lower bounds are positive under the conditional company bootstrap.

This version completed 108 three-year screening trials (107 effective distinct recipes, accounting for matched no-embedding controls) and 30 bounded local-refinement configurations, with 1312 recorded model-fit calls and 47 frozen final pipelines. Folds, seeds and shared TabM heads are not counted as different methods. Coverage targets and actual shortfalls are in [trial coverage](../../../results/trial_coverage.csv). Active-window time to report: 6.37 hours. Original eight-hour window expired partial; a separate owner-authorized eight-hour continuation reused saved development evidence. Neither clock resets on resume. See [continuation registration](../../../configs/continuation_window_02.json) for the separately authorized window; this is additional compute, not a claim that all work fit the first eight hours. No paid compute. Fit seconds use the monotonic process timer, while the UTC budget includes observed time gaps; these clocks are not interchangeable.

The 2015-2018 historical labels were inspected in earlier versions. V2 is retrospective, not a new holdout or an annual probability-of-default study. All P2 strong baselines are retrained through 2014 on the same permitted records; P1 and P2-selected cutoff-2012 transfers remain separately identified.

## Evidence

- [Delivery index and result interpretation](DELIVERY_INDEX.md), [freshly executed results notebook](../../../notebooks/02_Model_Comparison.ipynb), [notebook receipt](../../../results/notebook_verification.json), and [all frozen-pipeline source-row reloads](../../../results/pipeline_row_alignment_reloads.json).
- [First comparison table](../../../results/baseline_bridge.csv), [selection trace](../../../results/selection_trace.csv), [2014 confirmation](../../../results/confirmation_2014.csv), [historical results](../../../results/historical_2015_2018.csv).
- [All logical trials](../../../results/trials.csv), [attempt history](../../../results/trial_attempts.csv), [budget shortfalls](../../../results/search_budget_summary.csv), [fold results](../../../results/development_by_year.csv), [matched embedding controls](../../../results/numeric_matched_controls.csv), [matched input controls](../../../results/matched_input_controls.csv), [training-budget controls](../../../results/training_budget_ablation.csv), [one-factor optimization controls](../../../results/optimization_controls.csv), [reconstructed ranking-pair accounting](../../../results/ranking_pair_accounting.csv).
- [Five-seed results](../../../results/seeds.csv), [paired company intervals](../../../results/paired_company_intervals.csv), [costs](../../../results/costs.csv), [numerical verification](../../../results/verification.json).
- [English report](../../../reports/Research_Report_EN.pdf), [Chinese performance report](../../../reports/Research_Guide_CN.pdf), [Chinese interview addendum](../frozen_reporting/Interview_Addendum_V2_CN.md).
- [Frozen protocol](../../../configs/protocol.yaml), [execution corrections and deviations](PROTOCOL_EXECUTION_NOTES.md), [sources and installed versions](REFERENCES.md), [AI and contribution disclosure](CONTRIBUTIONS.md).

## Local execution

Work from this `performance_v2` directory. The wrapper uses an isolated environment and the preserved parent implementation; the installed package versions are in `requirements_v2.txt` and `configs/runtime_versions.json`. Supply the authorized original dataset at the parent project's documented path; raw records and all serialized models remain local.

```bash
scripts/run_performance_v2.sh scripts/progress_v2.py
scripts/resume_performance_v2.sh --phase all
scripts/run_performance_v2.sh -m distress_v2.verify
scripts/run_performance_v2.sh -m distress_v2.inference --help
scripts/run_performance_v2.sh scripts/create_local_replication.py --name rerun1
```

Resume never resets either registered wall clock or reopens search after confirmation. After the registered run is complete, preserve its freezes; a new search requires a newly registered version, not deleting a lock. Local model files are indexed by `configs/final_inventory.json`; each has a preprocessing/config sidecar, and neural states additionally have `.pt` files. Models are trusted local joblib artifacts, not safe files to load from strangers. Private data, company-level predictions, audit manifests, runtime state and resume drafts are excluded from public Git.

The replication command above is a read-only preflight. When the owner explicitly adds `--prepare`, it creates a new sibling directory, preserves this directory and clock, reuses the existing local runtime, and registers a separate eight-hour run; training still requires manually running its resume script. Full original-role reconstruction requires the owner's original authorized source archive and data, which are intentionally absent from a public clone. Earlier bridge source snapshots and adaptive budget-dependent HPO counts need not be bit-identical to a fresh corrected run; per-trial configs, source commits and saved predictions identify the original evidence. No additional replication was launched during this bounded run.

The two archive-only inference rows required an explicit cached-feature order correction after scoring. The correction scripts replay the diagnosed permutation and align the same frozen pipelines to source row IDs, without fitting or reselecting. Original pre-correction evidence remains private. A new replication receives the one-line inference order fix before its first fit/freeze; the current frozen source remains preserved. See execution note 14 and the two aggregate archive correction receipts. Final verification now reloads all 47 pipelines in source-row order. Both original archive metrics and corrected retraining results remain distinct.

The inherited V1 paths and historical lock are unchanged. The V2 bridge used a source snapshot before the compact-mask implementation; its purpose is a controlled mechanism diagnostic, explicitly distinct from P2 HPO. One migrated V1 ROS trial was invalidated for an epoch-refit step error and replaced before selection; it is retained in the ledger and does not explain original V1 performance.

The final preservation check confirmed 1344 readable protected files against their initial hashes. 1 private original file(s) could not be read under current system permissions, including an escalated read-only attempt; no current hash-equality claim is made for those files. Exact local exceptions remain in the private audit directory and are not published.

Publication is complete on the [verified public branch](https://github.com/MartinXQD/corporate-distress-ml/tree/research/performance-v2/performance_v2). [Publication checks](PUBLICATION.md) record identity, remote content and preserved main history.
