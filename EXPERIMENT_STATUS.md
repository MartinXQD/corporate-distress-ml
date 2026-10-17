# Experiment status

Real local data only. Protocol: distress_individual_extension_v1.

- Completed 41 development configurations and 164 four-fold runs.
- Recorded 542 actual model fit calls, including selectors, internal stopping refits, calibration and pilot attempts. Fit call 2 crashed during the first software pilot; its repair is documented. Synthetic software tests are separate.
- Froze 41 final artifacts, including untuned company stress and seed repeats; evaluated 24 primary pipelines on the previously inspected historical period.
- CPU device, two threads; no paid compute. Runtime and model costs are in the aggregate tables and durable fit ledger.
- 5 pre-registered seeds for each selected ML/DL representative. Paired company bootstrap: 2,000 draws, conditional on fitted models.
- GRU status: completed. CatBoost availability and measured rows are shown explicitly. The equal-rank candidate's decision is in configs/ensemble_choice.json.
- English research and Chinese teaching PDFs generated; rendered QA passed for all 15 English and 26 Chinese pages. Four notebooks executed in fresh kernels with zero errors. Remote release evidence is recorded separately.

No source score is represented as a new experiment. Historical evaluation remains previously inspected. No one-year default-event horizon, production readiness or causal explanation is claimed.

Supplementary synthetic diagnostic/unit model-fit attempts: 27. Total counted local model-fit attempts: 569/600. These software fixtures never enter research scores.

Final local verification: 141 original files unchanged, 98 frozen artifact hashes verified, 41 prediction files aligned with the actual source rows. All 28 software tests passed (0 skipped). Both representative checkpoints reproduced; the inference CLI ignored supplied labels and matched frozen scores at 1e-6 tolerance. No research model was refitted after freeze.

Public release: https://github.com/MartinXQD/corporate-distress-ml. Authenticated owner MartinXQD; public visibility, content commit `67764ef91116397c3be7488dc23ed6c68071c9fd` and five key files were verified remotely at 2026-10-17T05:50:23.753997+00:00. The publication receipt is saved in results/public/publication.json; follow-up documentation commits do not change the frozen research artifacts.
