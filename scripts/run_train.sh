#!/usr/bin/env bash
# Train/fine-tune the FAST model on the converted dataset.
# All parameters (pretrained-weights, epochs, out-dir, etc.) live in
# config.yaml - edit that file, not this script. Run convert.sh first so
# converted-dir/images and converted-dir/masks exist.
#
# Usage:
#   ./run_train.sh               # uses ./config.yaml
#   ./run_train.sh other.yaml    # uses a different config file
#   ./run_train.sh config.yaml --epochs 5   # extra args are passed through

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="config.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift
fi

PYTHON="../python3.11/Scripts/python.exe"
"$PYTHON" train_supervisely.py --config "$CONFIG" "$@"
