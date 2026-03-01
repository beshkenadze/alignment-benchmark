#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="/opt/homebrew/bin/uv"

if [[ ! -x "$UV_BIN" ]]; then
  UV_BIN="uv"
fi

"$UV_BIN" venv --python 3.12 "$SCRIPT_DIR/.venv"
"$UV_BIN" pip install --python "$SCRIPT_DIR/.venv/bin/python" parakeet-mlx

printf 'parakeet-mlx environment ready at %s\n' "$SCRIPT_DIR/.venv"
