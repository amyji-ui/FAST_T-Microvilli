#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="python3.11/Scripts/python.exe"
"$PYTHON" utils/pseudo_label.py "$@"
