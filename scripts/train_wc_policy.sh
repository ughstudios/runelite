#!/usr/bin/env bash
set -euo pipefail

# Train a woodcutting policy using the RLBot gym environment and save it.

main() {
  # Resolve repo root
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
  else
    THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
  fi
  cd "$REPO_ROOT"

  # Ensure venv exists (do not reinstall deps by default)
  ./scripts/setup_python_env.sh --no-pip >/dev/null 2>&1 || true
  # shellcheck disable=SC1091
  source .venv/bin/activate

  SAVE_PATH="rlbot/models/wc_policy.npz"
  mkdir -p "$(dirname "$SAVE_PATH")"

  echo "Training policy -> $SAVE_PATH"
  python scripts/gym_env/rlbot_gym_env.py \
    --mode train \
    --episodes 300 \
    --max-steps 360 \
    --lr 0.01 \
    --stochastic \
    --timeout 10 \
    --space-timeout -1 \
    --action-preset all \
    --actions "BankWhenFullTask,ChopNearestTreeTask,IdleTask,CameraRotateLeftTask,CameraRotateRightTask,CameraZoomInTask,CameraZoomOutTask" \
    --near-full-slots 0 \
    --temp-start 2.2 \
    --temp-end 1.0 \
    --temp-decay-episodes 220 \
    --eps-start 0.35 \
    --eps-end 0.12 \
    --eps-decay-episodes 240 \
    --entropy-beta 0.05 \
    --save "$SAVE_PATH"
}

main "$@"
