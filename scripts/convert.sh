#!/usr/bin/env bash
# Convert a Supervisely-labeled project into image/mask PNGs for training.
# All parameters (project-dir, converted-dir, class map, paint order, etc.)
# live in config.yaml - edit that file, not this script.
#
# Usage:
#   ./convert.sh                # uses ./config.yaml
#   ./convert.sh other.yaml     # uses a different config file
#   ./convert.sh config.yaml --no-preview   # extra args are passed through

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="config.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift
fi

PYTHON="../python3.11/Scripts/python.exe"
"$PYTHON" convert_supervisely_annotations.py --config "$CONFIG" "$@"
