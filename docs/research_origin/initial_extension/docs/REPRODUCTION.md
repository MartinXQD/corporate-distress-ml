# Reproduction and recovery

## Existing local artifacts

Use `scripts/run.sh -m distress_ml.cli reproduce --model FINAL_LABEL`, with a label from `configs/final_freeze.json`. This verifies frozen hashes and compares checkpoint scores with the saved local predictions at 1e-6 tolerance. It does not fit or tune anything. Notebook 04 performs the same audit through the shared package.

`predict --model artifacts/models/FINAL_LABEL.joblib --input /path/to/raw_inputs.csv --output artifacts/new_scores.csv` accepts the source financial schema. Provide company/year history through each target year for S2 or sequence models; never supply future labels. For annual rank ensembles and review rules, supply the complete relevant annual cohort. The API does not establish real-time event or publication-date availability.

## Full training in a fresh copy

Create a new local directory/copy of this extension to preserve the existing freeze, predictions and ledger. Obtain an authorized copy of the exact raw snapshot; do not take upstream descriptions as version equivalence. Run `python scripts/prepare_data.py /path/to/american_bankruptcy.csv` and set `DISTRESS_DATA` to it.

Create a project virtual environment with Python 3.13.9 (this run used the installed Anaconda base with system-site packages, then installed missing dependencies only in the project venv). Install `requirements-lock.txt`, then install this package with `pip install --no-deps --no-build-isolation -e .`. The launcher explicitly sets a relative PYTHONPATH because macOS may hide editable-install .pth files. On macOS it also directs libraries to the PyTorch OpenMP runtime before Python starts; no system library is edited.

Run audit, smoke, develop, sequences, freeze, historical-evaluate and report in that order. The historical lock prevents repeated evaluation for selection. The registered protocol remains unchanged; code and numerical environment differences must be disclosed. Exact CPU/device bit equality across different package builds is not promised.

## Recovery after interruption

`RUN_STATUS.json`, `NEXT_ACTIONS.md`, per-fold completion files and the append-only fit ledger describe local progress. Completed development folds are skipped by their completion records; incomplete fold attempts remain logged and must receive a new artifact/run identifier before resuming. Do not erase a historical lock. If a substantive bug invalidates scoring, record the prior run, evidence, invalidated claims, fix and amendment before any approved re-evaluation.

The original source manifest is in the ignored audit directory. Recompute hashes at delivery to verify source preservation. Reports and figures may be regenerated from saved scores without refitting models. Public notebooks run aggregate analysis without requiring data; local-only branches additionally reproduce saved trusted artifacts and cases without displaying identifiers or full row predictions.

## Tests and dependencies

CI runs only synthetic software fixtures. It never downloads the research data or triggers full training. The lock file pins the installed macOS dependency closure; `results/public/environment.json` records versions used, while the complete inherited-environment freeze remains local. A clean installation should also preserve a resolved environment lock for its own platform.

To initialize the fresh training directory without copying old result tables or deleting the current evidence, run:

```sh
scripts/run.sh scripts/new_run.py --destination artifacts/reproductions/reproduction_v1
```

Choose a destination that does not already exist. This copies only implementation/tests/docs and the original registered protocol, initializes empty artifact/result/report directories, and reuses the local venv by symlink when present. Set DISTRESS_DATA explicitly and run the new directory's launcher. Its historical exposure disclosure remains unchanged; a new run directory is not new external data.
