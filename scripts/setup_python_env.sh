#!/usr/bin/env bash
set -euo pipefail

# Create a local Python venv and install requirements for the RLBot Gym env.

main() {
  local SKIP_PIP=0
  # Parse simple flags
  for arg in "$@"; do
    case "$arg" in
      --no-pip|--skip-install) SKIP_PIP=1 ;;
    esac
  done
  # Resolve repo root robustly (works in bash, zsh, sh)
  local REPO_ROOT
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
  else
    local THIS_DIR
    THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
  fi

  cd "$REPO_ROOT"
  echo "Repo root: $REPO_ROOT"

  # Pick a python executable (allow override via PYTHON_BIN)
  local PY
  if [ "${PYTHON_BIN:-}" != "" ]; then
    PY="${PYTHON_BIN}"
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    echo "Error: python3 or python not found in PATH" >&2
    exit 1
  fi
  echo "Using Python: $($PY -V 2>&1)"

  # Create venv under .venv
  local VENV="${REPO_ROOT}/.venv"
  if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    "$PY" -m venv "$VENV" || { echo "venv creation failed with $PY; trying 'python -m venv' fallback" >&2; python -m venv "$VENV"; }
  else
    echo "Using existing venv at $VENV"
  fi

  # Activate and install requirements (optional)
  if [ ! -f "$VENV/bin/activate" ]; then
    echo "Error: venv activate script not found at $VENV/bin/activate" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  if [ "$SKIP_PIP" = "1" ]; then
    echo "Skipping pip install (per --no-pip)"
  else
    python -m pip install --upgrade pip || true
    if [ -f "$REPO_ROOT/requirements.txt" ]; then
      pip install -r "$REPO_ROOT/requirements.txt" || true
    else
      # Fallback minimal deps
      pip install gymnasium numpy || true
    fi
  fi

  echo
  echo "✅ Python venv ready: $VENV"
  echo "To activate:"
  echo "  source .venv/bin/activate"
  echo
  echo "Run the demo once RuneLite is running and RLBot (Dev) is enabled:"
  echo "  python scripts/gym_env/rlbot_gym_env.py --ipc-dir rlbot-ipc --episodes 1 --max-steps 200"
}

main "$@"
