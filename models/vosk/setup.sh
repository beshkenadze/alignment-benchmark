#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  UV_BIN="uv"
fi

download_model() {
  local model_name="$1"
  local model_url="$2"
  local archive_path="$SCRIPT_DIR/${model_name}.zip"
  local model_dir="$SCRIPT_DIR/$model_name"

  if [[ -d "$model_dir" ]]; then
    printf 'model already exists: %s\n' "$model_dir"
    return 0
  fi

  printf 'downloading %s\n' "$model_name"
  curl -fL "$model_url" -o "$archive_path"

  printf 'extracting %s\n' "$model_name"
  unzip -q "$archive_path" -d "$SCRIPT_DIR"
  rm -f "$archive_path"

  if [[ ! -d "$model_dir" ]]; then
    printf 'expected model directory not found after extract: %s\n' "$model_dir" >&2
    return 1
  fi
}

"$UV_BIN" venv --python 3.12 "$SCRIPT_DIR/.venv"
"$UV_BIN" pip install --python "$SCRIPT_DIR/.venv/bin/python" vosk

download_model "vosk-model-small-en-us-0.15" "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

if ! download_model "vosk-model-ru-0.42" "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"; then
  printf 'falling back to vosk-model-small-ru-0.22\n'
  download_model "vosk-model-small-ru-0.22" "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
fi

printf 'vosk environment ready at %s\n' "$SCRIPT_DIR/.venv"
