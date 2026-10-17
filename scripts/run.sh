#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
TASK_PYTHON="${DISTRESS_PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$(uname -s)" = Darwin ]; then
    TASK_TORCH_LIB="$($TASK_PYTHON -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent/"lib")')"
    export DYLD_LIBRARY_PATH="$TASK_TORCH_LIB"
fi
export DISTRESS_OMP_READY=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
exec "$TASK_PYTHON" "$@"
