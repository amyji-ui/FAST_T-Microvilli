#!/usr/bin/env bash
# Finds a good per-class confidence threshold by sweeping values and
# reporting Dice/precision/recall at each, on the TEST set. See
# utils/sweep_threshold.py's docstring for the full explanation and how to
# apply the result via quantify.sh's --class-threshold.
#
# Usage:
#   ./sweep_threshold.sh --run-dir ../trained_model_01 --class F-actin
#   ./sweep_threshold.sh --run-dir ../trained_model_01 --class F-actin --values 0.3,0.4,0.5,0.6,0.7,0.8

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="config.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift
fi

PYTHON="python3.11/Scripts/python.exe"
"$PYTHON" utils/sweep_threshold.py --config "$CONFIG" "$@"
