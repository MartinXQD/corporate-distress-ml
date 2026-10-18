# Release checklist

Owner and new extension author: Qidian (Martin) Xing. Original team work is acknowledged separately. New-code license remains pending owner choice; public visibility does not grant an inferred MIT license.

## Local acceptance evidence

- [x] Work is confined to the independent `individual_extension` Git root. All 141 original source files match the pre-execution SHA-256 manifest; no original file was overwritten or deleted.
- [x] Protocol and optional sequence choices were registered before their experiments. Final selection, seeds, model/source hashes and ensemble admission were frozen before one-time historical scoring.
- [x] 41 configurations / 164 calendar-fold runs completed on real data. Recorded model-fit attempts: 542 research + 27 synthetic diagnostic/unit = 569 / 600. Analytical baselines and 15 separately disclosed single-update software checks do not produce research scores. CPU, two threads; no paid training service.
- [x] Required Logistic, Random Forest, XGBoost, PyTorch MLP, Residual MLP and FT-Transformer ran; CatBoost, GRU and same-row flattened/snapshot controls also completed. Five pre-registered seeds per selected representative and three untuned company holdouts were retained.
- [x] Final tests: 28 passed, 0 failed, 0 skipped. Local logs preserve the earlier OpenMP pilot failure, its diagnostics, one corrected test fixture, the initial notebook port restriction and corrected numerical-verifier precision assumptions.
- [x] Four notebooks executed in separate fresh kernels, 17 code cells total, zero error outputs. Local-only branches reloaded the actual DL checkpoint and registered FN case.
- [x] Every primary saved metric was recomputed from predictions; all 41 final prediction sets were aligned to source IDs/years/labels. All 98 frozen artifact hashes verified. Both selected checkpoints reproduced at 1e-6 tolerance. Changed input labels did not change CLI classifications.
- [x] English report: 15 pages. Chinese teaching/interview guide: 26 pages including 24 interview questions. All pages rendered and visually inspected; no observed clipped/overlapping text, broken Chinese glyphs or unintended blank pages. PDF hashes and review details are in `results/public/pdf_quality_checks.json`.
- [x] Code, registered configs, executed notebooks, editable reports, aggregate tables/figures, claim index, model/data cards and attribution disclosures are available locally. Model states, preprocessors, calibration, thresholds, row predictions, full numerical case and bilingual resume drafts remain in ignored local artifacts/reports.
- [x] HTTPS GitHub credential and connector identity were independently verified as MartinXQD. Credential material is never printed or committed. The publishing command rechecks authenticated identity before creation.
- [x] Initial final-content staged-index audit passed: 123 files, 11,914,833 bytes, zero secret/private-path/forbidden-file/size/link issues. Prior Git history blobs were also scanned without findings. The index audit is refreshed after this receipt before upload.
- [x] Created a new public repository under MartinXQD; remote content commit `67764ef91116397c3be7488dc23ed6c68071c9fd`, public visibility and five key files verified. No existing repository was overwritten; no force push was used.

## Required eight-question audit

1. Were old CSV/report scores copied as new runs? **No.** Source scores are archival only; new results come from local fit records and predictions.
2. Was historical evaluation described as new untouched data? **No.** Prior 2015-2018 exposure remains explicit throughout.
3. Was selection, SMOTE or scaling fitted globally before CV? **No.** Learning operations are training-role local; pure causal feature construction is separately tested for future invariance.
4. Were different protocols or row coverage used for an improvement claim? **No.** Paired comparisons use the same rows/protocol. Company stress mixes a smaller fit population with novelty and is explicitly sensitivity evidence.
5. Were historical labels used to choose models, seeds, thresholds or case rules? **No new selection.** Choices were frozen before this one-time scoring; historical exposure from the original project cannot be eliminated. Post-hoc explanations never feed back into this version.
6. Was company ultimate status presented as a one-year event or calibrated annual PD? **No.** Missing event/publication dates and horizon remain unresolved.
7. Were unverified personal contributions assigned to the owner? **No.** Owner direction, Codex assistance, original team attribution and pending personal comprehension review are explicit. Resume wording is a local draft.
8. Were private team materials, raw data or credentials made public? **No.** The explicit Git index passed the source/data/credential/size/path/link scan; the audit is refreshed before upload. The public allowlist contains only new work and aggregate evidence. Local/upstream financial-value disagreement and uncertain local data rights are disclosed; neither raw nor processed data is redistributed.

## Measured conclusions

Development-selected RF historical AP = 0.205641 versus same-protocol S0 Logistic 0.153109. The paired 95% company-bootstrap AP-difference interval [-0.001797, 0.101701] crosses zero. Selected ROS Residual MLP AP = 0.105345; fixed ensemble AP = 0.184940. Development gains failed to transfer; no retrospective winner substitution or weight change occurred. Seeds, annual review budgets, calibration, seen/unseen and untuned group results remain in full aggregate tables.

## Publication

Published: https://github.com/MartinXQD/corporate-distress-ml. Authenticated owner and public visibility verified. Content commit: `67764ef91116397c3be7488dc23ed6c68071c9fd`. Both PDFs, the README, historical result table and Notebook 04 were verified through the remote API. Final publication metadata is committed as a follow-up documentation change; its HEAD is verified separately in the local final receipt. Public data/checkpoints/team materials remain excluded. No existing remote was replaced and no force push was used.
