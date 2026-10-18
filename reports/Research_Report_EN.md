# Corporate Financial Distress Classification

Qidian (Martin) Xing. Individual research extension of a UCLA MFE 413 team project, with disclosed Codex assistance.

This reading guide explains the current project and links its frozen evidence. The [English research PDF](Research_Report_EN.pdf) is the original seven-page verified report, preserved byte for byte. Its research-stage labels remain source identifiers rather than separate current project versions.

## Objective and data

Use U.S. company financial statements to classify records labeled failed/alive and rank financial distress risk. The authorized snapshot contains 78,682 firm-years from 8,971 companies across 1999–2018. Its company-constant ultimate-status labels and unavailable failure/publication dates mean this is retrospective classification rather than verified one-year failure prediction.

## Results

The development-frozen equal blend of five-seed XGBoost and MLP annual ranks has mean annual AP **0.373320**, pooled ROC-AUC **0.921602**, and mean annual Recall@Top 5% **0.619824** on previously inspected 2015–2018 records. The strongest fixed same-cutoff baseline has mean annual AP **0.302835**, giving a difference of **+0.070485**. [Full comparison](../results/MODEL_COMPARISON.md)

The selected MLP alone has historical annual AP 0.392916. It is retained as a comparator; the official blend was chosen on development data and is not replaced after historical inspection. [Exact selection process](../docs/MODEL_SELECTION.md)

## Methods and findings

Financial transforms and preprocessing are fitted inside each training role. Expanding calendar validation, three-seed stability, frozen 2014 confirmation, five-seed final training, and annual rank blending structure the evaluation. Three-year recency weights help both selected backbones in matched development controls. More elaborate feature sets, embeddings, larger networks, and ranking losses show mixed or negative outcomes. Equal-update sampling controls prevent conflating resampling with extra optimization.

[Methods](../docs/METHODOLOGY.md), [matched controls](../results/optimization_controls.csv), [training budgets](../results/training_budget_ablation.csv), and [all attempts](../results/trial_attempts.csv) preserve the numerical detail and negative results.

## Interpretation and limits

Company-bootstrap intervals are conditional on the fitted models and observed sample. They do not include full selection or macro uncertainty. Only five failed companies are naturally unseen at the final cutoff. Company-group stress is post-selection and substantially weaker. Previously inspected historical labels, fixed ultimate company status, and missing point-in-time accounting information preclude claims of independent external validation, calibrated annual PD, production readiness, or investment profitability.

See [data card](../docs/DATA_CARD.md), [provenance](../docs/RESEARCH_HISTORY.md), [contributions](../docs/CONTRIBUTIONS.md), and [reproduction](../docs/REPRODUCTION.md). The current [executed notebook](../notebooks/02_Model_Comparison.ipynb) displays the same aggregate evidence without training.
