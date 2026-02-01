#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-/path/to/h5adify_v0.0.7_final.zip}"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt

# Install your toolkit
python -m pip install "${ZIP_PATH}"

python -c "import h5adify; print('h5adify version:', getattr(h5adify, '__version__', 'unknown'))"
