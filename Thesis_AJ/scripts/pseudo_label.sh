#!/usr/bin/env bash
# Run the trained model on new, unlabeled images and package the predictions
# as a Supervisely-importable dataset for correction (model-assisted
# labeling), instead of labeling from scratch. See utils/pseudo_label.py's
# module docstring for the full workflow (import into Supervisely, correct,
# export back into Images/Labeled, then convert.sh + run_train.sh again).
#
# Usage (paths are relative to the Thesis_AJ folder, since that's where this
# switches to before running):
#   ./pseudo_label.sh --images-dir "Images/TIRF_HEK_CD8AA_Out" "Images/TIRF_HEK_Control_Out" --out pseudo_labels
#   ./pseudo_label.sh --images-dir "Images/TIRF_HEK_CD8AA_Out" --out pseudo_labels --limit 15
#   ./pseudo_label.sh --config other.yaml --images-dir ... --out ...

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="python3.11/Scripts/python.exe"
"$PYTHON" utils/pseudo_label.py "$@"
