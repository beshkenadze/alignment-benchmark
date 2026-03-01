#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ENV_NAME="mfa_bench"
MICROMAMBA_HOME="$SCRIPT_DIR/.micromamba"
MICROMAMBA_BIN="$MICROMAMBA_HOME/bin/micromamba"

log() {
  printf '[mfa-setup] %s\n' "$1"
}

download_model() {
  local model_type="$1"
  local primary_model="$2"
  local fallback_model="${3:-}"

  if "${MFA_RUNNER[@]}" model download "$model_type" "$primary_model"; then
    return 0
  fi

  if [ -n "$fallback_model" ]; then
    log "model $primary_model unavailable for $model_type, trying $fallback_model"
    "${MFA_RUNNER[@]}" model download "$model_type" "$fallback_model"
    return 0
  fi

  return 1
}

create_python_venv() {
  if command -v uv >/dev/null 2>&1; then
    log "creating Python venv via uv at $VENV_DIR"
    uv venv --clear --python 3.12 "$VENV_DIR"
  else
    log "creating Python venv via python3 at $VENV_DIR"
    python3 -m venv --clear "$VENV_DIR"
  fi

}

install_micromamba() {
  local os_arch
  os_arch="osx-arm64"
  mkdir -p "$MICROMAMBA_HOME"
  log "installing micromamba into $MICROMAMBA_HOME"
  curl -Ls "https://micro.mamba.pm/api/micromamba/${os_arch}/latest" | tar -xj -C "$MICROMAMBA_HOME" bin/micromamba
}

ensure_conda_env() {
  local manager="$1"
  local create_cmd
  create_cmd=("$manager" create -y -n "$ENV_NAME" -c conda-forge montreal-forced-aligner python=3.10)

  if "$manager" run -n "$ENV_NAME" mfa version >/dev/null 2>&1; then
    log "existing environment '$ENV_NAME' found for $manager"
  else
    log "creating environment '$ENV_NAME' with $manager"
    "${create_cmd[@]}"
  fi
}

MFA_RUNNER=()

create_python_venv

if command -v conda >/dev/null 2>&1; then
  ensure_conda_env "conda"
  MFA_RUNNER=(conda run -n "$ENV_NAME" mfa)
elif command -v mamba >/dev/null 2>&1; then
  ensure_conda_env "mamba"
  MFA_RUNNER=(mamba run -n "$ENV_NAME" mfa)
elif command -v micromamba >/dev/null 2>&1; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
  ensure_conda_env "micromamba"
  MFA_RUNNER=(micromamba run -n "$ENV_NAME" mfa)
else
  install_micromamba
  export MAMBA_ROOT_PREFIX="$MICROMAMBA_HOME/root"
  ensure_conda_env "$MICROMAMBA_BIN"
  MFA_RUNNER=("$MICROMAMBA_BIN" run -n "$ENV_NAME" mfa)
fi

log "downloading MFA models"
download_model acoustic english_mfa
download_model dictionary english_mfa
download_model g2p english_mfa english_us_mfa
download_model acoustic russian_mfa
download_model dictionary russian_mfa
download_model g2p russian_mfa

log "setup complete; venv at $VENV_DIR"
