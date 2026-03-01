#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"
PYTHON_PATH="$VENV_PATH/bin/python"
FALLBACK_REPO_DIR="$SCRIPT_DIR/third_party/whisper-char-alignment"

uv venv --python 3.12 "$VENV_PATH"
uv pip install --python "$PYTHON_PATH" stable-ts

if "$PYTHON_PATH" - <<'PY'
import inspect
from stable_whisper.whisper_word_level.original_whisper import transcribe_stable

raise SystemExit(0 if "aligner" in inspect.signature(transcribe_stable).parameters else 1)
PY
then
  printf 'stable-ts char aligner support detected in installed package\n'
else
  printf 'stable-ts from PyPI does not expose char aligner; installing latest from git\n'
  uv pip install --python "$PYTHON_PATH" --upgrade "git+https://github.com/jianfch/stable-ts.git"
fi

uv pip install --python "$PYTHON_PATH" num2words

if [ ! -d "$FALLBACK_REPO_DIR/.git" ]; then
  mkdir -p "$SCRIPT_DIR/third_party"
  git clone --depth 1 "https://github.com/30stomercury/whisper-char-alignment.git" "$FALLBACK_REPO_DIR"
fi

printf 'whisper_char_align environment ready at %s\n' "$VENV_PATH"
