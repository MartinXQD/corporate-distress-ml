#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
TASK_ROOT="$PWD"
TASK_PYTHON="${DISTRESS_PYTHON:-$TASK_ROOT/.venv/bin/python}"
export PYTHONPATH="$TASK_ROOT/src${DISTRESS_EXTRA_SITE:+:$DISTRESS_EXTRA_SITE}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${DISTRESS_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
if [ "$(uname -s)" = Darwin ]; then
  TASK_TORCH_LIB="$("$TASK_PYTHON" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent/"lib")')"
  export DYLD_LIBRARY_PATH="$TASK_TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export DISTRESS_OMP_READY=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TASK_ROOT/artifacts/cache/matplotlib}"
exec "$TASK_PYTHON" "$@"
