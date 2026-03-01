#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/DATA/alignment-benchmark/models/ctc_aligner
uv venv --python 3.12
uv pip install --python .venv/bin/python git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
