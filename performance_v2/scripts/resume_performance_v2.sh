#!/bin/sh
set -eu
exec "$(dirname "$0")/run_performance_v2.sh" -u -m distress_v2.runner "$@"
