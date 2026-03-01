#!/usr/bin/env bash
set -euo pipefail

cd "/Volumes/DATA/alignment-benchmark/models/faster_whisper"
uv venv --python 3.12
uv pip install --python .venv/bin/python faster-whisper
