#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
TASK_V2="$PWD"
TASK_V1="$(dirname "$PWD")"
TASK_PYTHON="$TASK_V2/.venv/bin/python"
TASK_V1_SITE="$TASK_V1/.venv/lib/python3.13/site-packages"
TASK_V2_SITE="$TASK_V2/.venv/lib/python3.13/site-packages"
export PYTHONPATH="$TASK_V2/src:$TASK_V1/src:$TASK_V2_SITE:$TASK_V1_SITE${PYTHONPATH:+:$PYTHONPATH}"
TASK_TORCH_LIB="$("$TASK_PYTHON" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent/"lib")')"
export DYLD_LIBRARY_PATH="$TASK_TORCH_LIB"
export DISTRESS_OMP_READY=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${V2_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$TASK_V2/artifacts/cache/matplotlib"
exec "$TASK_PYTHON" "$@"
