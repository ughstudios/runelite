#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained woodcutting policy with greedy actions.

main() {
  # Resolve repo root
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
  else
    THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
  fi
  cd "$REPO_ROOT"

  # Ensure venv exists
  ./scripts/setup_python_env.sh --no-pip >/dev/null 2>&1 || true
  # shellcheck disable=SC1091
  source .venv/bin/activate

  LOAD_PATH="${1:-rlbot/models/wc_policy.npz}"
  if [ ! -f "$LOAD_PATH" ]; then
    echo "Error: model not found at $LOAD_PATH" >&2
    exit 1
  fi

  echo "Evaluating policy: $LOAD_PATH"
  python scripts/gym_env/rlbot_gym_env.py \
    --mode eval \
    --load "$LOAD_PATH" \
    --episodes 20 \
    --max-steps 300
}

main "$@"

