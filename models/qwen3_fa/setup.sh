#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/DATA/alignment-benchmark/models/qwen3_fa
if [ ! -d .venv ]; then
  uv venv --python 3.12
fi
uv pip install --python .venv/bin/python "mlx-audio>=0.3.1" || uv pip install --python .venv/bin/python --prerelease allow "mlx-audio>=0.3.1"
