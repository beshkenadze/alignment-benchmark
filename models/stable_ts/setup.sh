#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv --python 3.12 "$SCRIPT_DIR/.venv"
uv pip install --python "$SCRIPT_DIR/.venv/bin/python" stable-ts

printf 'stable-ts environment ready at %s\n' "$SCRIPT_DIR/.venv"
