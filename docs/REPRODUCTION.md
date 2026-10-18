# Reproduction and verification

## Install

Use Python 3.13 and the recorded [dependency lock](../requirements-lock.txt). The lock combines the actual initial environment with the numerical-embedding and search dependencies. Native wheels may differ by platform. No data is downloaded automatically.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

`scripts/run.sh` sets bounded numerical threads and the local import path. On macOS it points the process to PyTorch's local OpenMP library to match the verified research environment. It does not modify system libraries. `DISTRESS_PYTHON`, `DISTRESS_EXTRA_SITE`, and `DISTRESS_THREADS` can point to an existing compatible local runtime; they are optional.

## Public checks without data

```bash
scripts/run.sh -m pytest -q
scripts/run.sh scripts/verify_project.py
scripts/run.sh scripts/execute_notebooks.py
scripts/run.sh -m distress_v2.cli status
scripts/run.sh -m distress_v2.cli infer --help
```

Verification checks original aggregate hashes, registered protocol identity, frozen selection, arithmetic across saved annual tables, required reports/figures, local Markdown links, executed-notebook outputs, and the public-file policy. It does not claim to recompute scores from private predictions. Both notebooks execute without private data or models; optional private checks say explicitly when files are absent.

CI uses synthetic software fixtures and these public evidence checks. Synthetic model updates validate software behavior and are never reported as research results.

## Frozen recipe replay

Supply the authorized dataset with SHA-256 `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`. Its 78,682 records / 8,971 companies differ numerically from the current upstream CSV despite matching IDs, years, and labels. Obtain the correct snapshot through authorized channels; do not silently substitute a download.

```bash
export DISTRESS_DATA=/absolute/path/to/american_bankruptcy.csv
# Read-only preflight: validates the snapshot, choices, and new output path.
scripts/run.sh scripts/reproduce.py --output artifacts/replications/demo
# Explicit training request: creates a fresh, separately timed run.
scripts/run.sh scripts/reproduce.py --output artifacts/replications/demo --run
```

The replay uses the selected XGBoost/MLP recipes, the fixed strongest lag-XGBoost reference, and the five registered seeds. It refits through 2014 using each latest training year for stopping, averages predictions, and applies the frozen within-year rank blend. Output includes separate models, row scores under ignored artifacts, aggregate results, config copies, fit accounting, and its own eight-hour clock. It never writes to the released results, resets the original clock, or chooses a new winner. Running it constitutes additional local computation, not an independent test. No replay training was launched as part of repository consolidation.

The bounded public replay does not reproduce the entire adaptive HPO trajectory, original team archive roles, or all 47 original artifacts. The original corrected source, per-trial configurations, population hashes, and aggregate receipts preserve those details. Results may vary with native libraries, hardware, and tie behavior. A rigorous new performance study needs fresh data and a separately registered selection protocol.

## Optional trusted original artifacts

The original full verification module expects private predictions, model files, execution state, original source context, and historical freezes. They are intentionally absent from a public clone. Do not treat a public receipt as a fresh local model reload.

For the owner's preserved original workspace, use its historical launcher and verification module. A relocated model context can be selected before import with `DISTRESS_PROJECT_ROOT`, `DISTRESS_SOURCE_ROOT`, and `DISTRESS_INITIAL_ROOT`. Preserve frozen source/configuration hashes. Do not reset locks or call old report/publishing scripts against the current project root.

The current inference entry accepts only a trusted local pipeline in the selected run's ignored models directory and verifies its frozen hash:

```bash
DISTRESS_PROJECT_ROOT=/absolute/path/to/trusted/run \
  scripts/run.sh -m distress_v2.cli infer \
  --model /absolute/path/to/trusted/run/artifacts/models/model.joblib \
  --input /absolute/path/to/authorized_accounts.csv \
  --output /absolute/path/to/trusted/run/artifacts/predictions/new_scores.csv
```

This inference command exposes one uncalibrated component. The final blend additionally requires both families' seed means and the complete contemporaneous yearly cohort. Scores are not annual default probabilities. Never load serialized files from untrusted sources.
