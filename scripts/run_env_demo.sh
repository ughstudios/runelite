#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the RLBot Gym environment.

main() {
  local SRC="${BASH_SOURCE[0]:-$0}"
  while [ -h "$SRC" ]; do
    local DIR="$(cd -P "$(dirname "$SRC")" && pwd)"
    local TARGET="$(readlink "$SRC")"
    case "$TARGET" in
      /*) SRC="$TARGET" ;;
      *)  SRC="$DIR/$TARGET" ;;
    esac
  done
  local REPO_ROOT
  REPO_ROOT="$(cd -P "$(dirname "$SRC")/.." && pwd)"
  cd "$REPO_ROOT"

  # Activate venv if present
  if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi

  local IPC_DIR="rlbot-ipc"
  local EPISODES=1
  local STEPS=200
  local LOG_LEVEL=${RLBOT_LOG_LEVEL:-INFO}

  mkdir -p "$IPC_DIR"
  rm -f "$IPC_DIR/action.json" "$IPC_DIR/obs.json"

  python scripts/gym_env/rlbot_gym_env.py \
    --ipc-dir "$IPC_DIR" \
    --episodes "$EPISODES" \
    --max-steps "$STEPS" \
    --log-level "$LOG_LEVEL" \
    "$@"
}

main "$@"
