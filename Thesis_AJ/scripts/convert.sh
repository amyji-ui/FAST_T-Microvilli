#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="config.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift
fi

PYTHON="python3.11/Scripts/python.exe"
"$PYTHON" utils/convert_supervisely_annotations.py --config "$CONFIG" "$@"
