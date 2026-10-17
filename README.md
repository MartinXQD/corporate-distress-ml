# Corporate Financial Distress

**A Reproducible ML & Deep Learning Benchmark - Independent Extension**

**Qidian (Martin) Xing** | 2026-10-16 | Codex-assisted implementation and documentation

This is retrospective failed-company classification on 78,682 firm-years and 8,971 companies. Company labels are constant over time; true event and financial-publication dates are unavailable. The previously inspected 2015-2018 period is a historical evaluation, not a new untouched external holdout or verified one-year default prediction.

## Measured evidence

41 complete development configurations, 164 calendar-fold runs, 542 recorded model fit calls, 41 final artifacts, 5 representative seeds, and 2,000 paired company-bootstrap draws. Required Logistic, Random Forest, XGBoost, PyTorch MLP, Residual MLP and FT-Transformer were run, together with prevalence/Altman baselines, available CatBoost and registered GRU/flattened-history controls.

| Development-selected representative | Set | Historical AP | ROC-AUC | Annual Recall@Top5% |
| --- | --- | --- | --- | --- |
| Random Forest | S1 | 0.206 | 0.862 | 0.460 |
| Residual MLP | S1 | 0.105 | 0.815 | 0.294 |
| Same-protocol Logistic baseline | S0 | 0.153 | 0.833 | 0.329 |

These are frozen seed-42 results, not test-selected winners. The Random Forest AP difference versus Logistic has a paired 95% company-bootstrap interval of [-0.0018, 0.1017], crossing zero. The selected network and fixed ensemble failed to transfer their development advantage; no historical winner was substituted. Read [full historical tables](results/public/historical_metrics.csv), [controlled ablations](results/public/controlled_ablation_summary.csv), [paired company intervals](results/public/paired_company_bootstrap.csv), [seed variation](results/public/seed_results.csv) and [company stress](results/public/group_stress_metrics.csv). Old team scores are not an improvement denominator.

## Read the artifacts

- [English research report](reports/Research_Report_EN.pdf) and [editable source](reports/Research_Report_EN.md)
- [Chinese teaching and interview guide](reports/Project_Interview_Guide_CN.pdf) and [editable source](reports/Project_Interview_Guide_CN.md)
- [Four executable notebooks](notebooks/01_Data_and_Features.ipynb) and [notebook execution evidence](results/public/notebook_execution.json)
- [Registered protocol](docs/RESEARCH_PROTOCOL.md), [data card](docs/DATA_CARD.md), [model card](docs/MODEL_CARD.md), [source audit](docs/SOURCE_AUDIT.md)
- [Claim-to-result index](results/CLAIMS.csv), [experiment status](EXPERIMENT_STATUS.md), [release checklist](RELEASE_CHECKLIST.md)
- [Contributions](docs/INDIVIDUAL_CONTRIBUTIONS.md), [provenance](docs/PROVENANCE.md), [AI use](docs/AI_USAGE.md), [acknowledgements](ACKNOWLEDGEMENTS.md) and [license status](LICENSE_NOTICE.md)

## Reproduce locally

Use an authorized raw snapshot with SHA-256 `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`. The [live upstream comparison](results/public/upstream_verification.json) finds matching IDs/years/labels but different financial values; do not silently substitute the current public CSV. Data, original team materials, full row predictions and model artifacts are excluded from Git.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
export DISTRESS_DATA=/path/to/american_bankruptcy.csv
scripts/run.sh -m pytest -q
scripts/run.sh -m distress_ml.cli audit
```

Full retraining belongs in a fresh run copy that preserves current freeze/history evidence. Run smoke, develop, sequences, freeze, historical-evaluate and report; details and checkpoint/predict commands are in [reproduction instructions](docs/REPRODUCTION.md). Reports need an available Chinese TrueType font on the report-authoring host. On this macOS run, the launcher shares PyTorch's local OpenMP library to avoid a verified multi-runtime crash. No system library is changed. CI runs only synthetic software tests; it does not download data or perform full research training.

## Attribution and limits

This repository presents an independent extension of an earlier UCLA MFE 413 team project. The original course submission is not represented as a sole-authored work. Original team authors are acknowledged separately. This is owner-directed, AI-assisted work; personal comprehension and independent manual review are not asserted.

New-code license is pending owner choice. No MIT or other license is inferred for this extension, the original team code or source data. Research outputs do not establish production readiness, an annual PD, causal financial drivers or investment profitability.
