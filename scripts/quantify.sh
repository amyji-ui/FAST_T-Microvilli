#!/usr/bin/env bash
# Compare the model's predicted masks with your manual masks: per-class Dice /
# precision / recall / F1, object counts, area fractions, and comparison
# between groups (CD8AA / CD8BB / Control). Uses the held-out TEST images of
# the training run in config.yaml's out-dir. Results go to
# <out-dir>/quantification_test/. Run run_train.sh first.
# All parameters live in config.yaml - edit that file, not this script.
#
# Usage:
#   ./quantify.sh                          # uses ./config.yaml, test images
#   ./quantify.sh other.yaml               # uses a different config file
#   ./quantify.sh config.yaml --subset val # extra args are passed through
#   ./quantify.sh --run-dir ../some_other_run

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../FAST"

CONFIG="config.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift
fi

PYTHON="../python3.11/Scripts/python.exe"
"$PYTHON" quantification.py --config "$CONFIG" "$@"
