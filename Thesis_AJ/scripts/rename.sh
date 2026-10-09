#!/usr/bin/env bash
# Rename every image in a folder to <prefix>_<index>.<ext>, sequentially
# (00, 01, 02, ... by default), and write rename_manifest.csv (new name ->
# original name) alongside them.
#
# Usage:
#   ./rename.sh <folder> <prefix>
#   ./rename.sh ../Images/Source/TIRF_HEK_CD8BB_Out CD8BB
#   ./rename.sh <folder> <prefix> --start 1 --digits 3   # extra args pass through

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="python3.11/Scripts/python.exe"
"$PYTHON" utils/rename_sequential.py "$@"
