#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLBotEnv: Gymnasium-compatible environment for RuneScape macro actions via simple IPC.

IPC directory layout (default: ./rlbot-ipc):
  - action_space.json : {"state_dim": int, "actions": [str, ...]}
  - obs.json          : single JSON object written by game-side writer each step
  - action.json       : {"seq": int, "action": int} written by env each step

This version:
  * Honors writer-provided reward (`last_reward`) and done (`done`)
  * Derives fallback reward when writer sends 0.0 (inventory delta + bank deposit bonus + step penalty)
  * Adds episode termination on progress-complete or no progress for too long
  * Tracks episode_return and episode_len in `info`
  * Warns on state shape mismatch vs `state_dim`
  * Resets counters on `reset()` and waits briefly for a fresh observation
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import time
from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple

import numpy as np

# Always use gymnasium (not gym)
import gymnasium
from gymnasium import spaces


# ----------------------------- Utilities -------------------------------------


def _setup_logger(level: str | None = None) -> logging.Logger:
    lvl = (level or os.getenv("RLBOT_LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger("RLBotEnv")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(lvl)
    return logger


def _resolve_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _resolve_ipc_dir(ipc_dir: str, log: logging.Logger | None = None) -> Path:
    """Resolve IPC dir; prefer the repo-local `rlbot-ipc` (same as run_runelite)."""
    p = Path(ipc_dir)
    if p.is_absolute():
        return p

    repo_root = _resolve_repo_root()
    candidates = [repo_root / "rlbot-ipc", Path.cwd() / ipc_dir]
    # also check a few parents relative to cwd for backwards compatibility
    cur = Path.cwd()
    for _ in range(3):
        cur = cur.parent
        candidates.append(cur / ipc_dir)

    for c in candidates:
        if c.exists():
            if log:
                log.debug("Resolved IPC dir -> %s", str(c))
            return c

    default = repo_root / "rlbot-ipc"
    if log:
        log.debug("IPC dir not found; using repo default -> %s", str(default))
    return default


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


# ----------------------------- Environment -----------------------------------


class RLBotEnv(gymnasium.Env):
    """
    Gymnasium environment that communicates with a game-side writer through JSON files.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ipc_dir: str = "rlbot-ipc",
        step_timeout_s: float = 5.0,
        poll_interval_s: float = 0.05,
        action_space_timeout_s: float | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__()
        self.log = logger or _setup_logger()
        self.dir = _resolve_ipc_dir(ipc_dir, self.log)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log.debug(
            "Init RLBotEnv(ipc_dir=%s, timeout=%.2fs, poll=%.2fs)",
            str(self.dir),
            step_timeout_s,
            poll_interval_s,
        )

        # IPC files
        self._obs_file = self.dir / "obs.json"
        self._action_file = self.dir / "action.json"

        # Step control (set before loading space; loader references _action_timeout)
        self._last_seq: int | None = None
        self._timeout = float(step_timeout_s)
        self._poll = float(poll_interval_s)
        self._action_timeout = (
            None if action_space_timeout_s is None else float(action_space_timeout_s)
        )

        # Action space (block until writer publishes or timeout)
        self._space = self._load_action_space(block=True)
        self.action_space = spaces.Discrete(len(self._space["actions"]))
        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(int(self._space["state_dim"]),),
            dtype=np.float32,
        )

        # --- Learning helpers (added) ---
        self._prev_obs: Dict[str, Any] | None = None
        self._no_progress_steps: int = 0
        self._progress_steps: int = 0
        self._free_idx: int | None = None
        self._bank_idx: int | None = None
        self._step_penalty: float = 0.01
        self._deposit_bonus: float = 5.0
        self._gather_bonus: float = 1.0
        self._max_no_progress: int = 200
        self._last_episode_return: float = 0.0
        self._last_episode_len: int = 0

        # Handle interrupts cleanly (propagate KeyboardInterrupt)
        def _on_sigint(*_):
            self.close()
            raise KeyboardInterrupt

        def _on_sigterm(*_):
            self.close()
            raise SystemExit(0)

        atexit.register(self.close)
        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigterm)

    # ----------------------- File/Space helpers -----------------------

    def _load_action_space(self, block: bool = True) -> Dict[str, Any]:
        space_file = self.dir / "action_space.json"
        t0 = time.time()
        while True:
            try:
                # If the configured dir doesn't have the file yet, try common fallbacks
                if not space_file.exists():
                    # Candidate dirs to probe if user-provided path is empty/mismatched
                    candidates = []
                    try:
                        # Repo-local default
                        candidates.append(Path.cwd() / "rlbot-ipc")
                    except Exception:
                        pass
                    try:
                        # RuneLite home default
                        candidates.append(Path.home() / ".runelite" / "rlbot-ipc")
                    except Exception:
                        pass
                    try:
                        # Temp fallback used by some environments
                        candidates.append(Path(tempfile.gettempdir()) / "rlbot-ipc")
                    except Exception:
                        pass

                    for c in candidates:
                        if c and c != self.dir:
                            sf = c / "action_space.json"
                            if sf.exists():
                                # Switch to discovered IPC dir
                                self.dir = c
                                self._obs_file = self.dir / "obs.json"
                                self._action_file = self.dir / "action.json"
                                space_file = sf
                                try:
                                    self.log.info("Auto-detected IPC dir: %s", str(self.dir))
                                except Exception:
                                    pass
                                break

                data = _read_json(space_file)
                assert isinstance(data.get("actions"), list) and isinstance(
                    data.get("state_dim"), (int, float)
                )
                # coerce to int
                data["state_dim"] = int(data["state_dim"])
                self.log.debug("Loaded action_space.json: %s", data)
                return data
            except Exception as e:
                if not block:
                    raise
                elapsed = time.time() - t0
                timeout = getattr(self, "_action_timeout", None)
                if timeout is not None and elapsed > timeout:
                    raise TimeoutError(f"Timed out waiting for {space_file}") from e
                if int(elapsed) % 5 == 0:
                    try:
                        self.log.debug("Waiting for agent to publish %s...", space_file)
                    except Exception:
                        pass
                time.sleep(self._poll)

    def _read_obs(self, block: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            try:
                return _read_json(self._obs_file)
            except Exception as e:
                if not block:
                    raise
                if time.time() - t0 > self._timeout:
                    raise TimeoutError(f"Timed out waiting for {self._obs_file}") from e
                time.sleep(self._poll)

    def _read_obs_blocking(self, min_next_seq: int) -> Dict[str, Any]:
        """Block until we see obs with seq >= min_next_seq (strictly greater than previous)."""
        t0 = time.time()
        while True:
            obs = self._read_obs(block=True)
            try:
                seq = int(obs["seq"])
            except Exception:
                seq = -1
            if seq >= int(min_next_seq):
                return obs
            if time.time() - t0 > self._timeout:
                self.log.warning(
                    "Timeout waiting for next obs (wanted >= %s)", min_next_seq
                )
                return obs
            time.sleep(self._poll)

    def _write_action(self, seq: int, action_index: int) -> None:
        msg = {"seq": int(seq), "action": int(action_index)}
        _write_json_atomic(self._action_file, msg)
        self.log.debug("Wrote action seq=%s -> %s (%s)", seq, action_index, self.action_name(action_index))

    # ----------------------------- Gym API ------------------------------------

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Try to read a fresh obs; if stale, poll briefly for a new seq
        obs = self._read_obs(block=True)
        start_seq = int(obs.get("seq", 0))
        for _ in range(10):
            nxt = self._read_obs(block=True)
            if int(nxt.get("seq", -1)) != start_seq:
                obs = nxt
                break

        self._last_seq = int(obs["seq"]) if "seq" in obs else None

        # Reset episode helpers
        self._prev_obs = None
        self._no_progress_steps = 0
        self._progress_steps = 0
        self._last_episode_return = 0.0
        self._last_episode_len = 0
        self._free_idx = None
        self._bank_idx = None

        state = np.asarray(obs.get("state", []), dtype=np.float32)

        # State dimension sanity
        state_dim = int(self._space.get("state_dim", len(state)))
        if state.shape[0] != state_dim:
            self.log.warning("RESET: state dim mismatch: got %s, expected %s", state.shape, state_dim)

        info = {
            "seq": self._last_seq,
            "state_names": obs.get("state_names"),
            "episode_return": 0.0,
            "episode_len": 0,
        }
        self.log.info("Reset -> seq=%s", self._last_seq)
        return state, info

    def step(self, action: int):
        if self._last_seq is None:
            raise RuntimeError("Call reset() before step().")

        # Write the chosen action for the writer to consume
        self._write_action(self._last_seq, int(action))

        # Wait for next observation (seq strictly greater)
        next_obs = self._read_obs_blocking(min_next_seq=self._last_seq + 1)
        self._last_seq = int(next_obs.get("seq", self._last_seq))

        # Extract state and validate shape
        state = np.asarray(next_obs.get("state", []), dtype=np.float32)
        try:
            state_dim = int(self._space.get("state_dim", len(state)))
        except Exception:
            state_dim = len(state)
        if state.shape[0] != state_dim:
            self.log.warning("State dim mismatch: got %s, expected %s", state.shape, state_dim)

        # Prefer writer's reward; if zero or missing, compute a fallback reward from deltas
        reward = float(next_obs.get("last_reward", 0.0) or 0.0)

        # Build indices once from state_names
        state_names = next_obs.get("state_names") or []
        if self._free_idx is None and state_names:
            try:
                self._free_idx = state_names.index("inventory_free_slots_bucket")
            except ValueError:
                self._free_idx = None
        if self._bank_idx is None and state_names:
            try:
                self._bank_idx = state_names.index("bank_open")
            except ValueError:
                self._bank_idx = None

        # Fallback reward if writer didn't provide signal
        if reward == 0.0 and self._prev_obs is not None:
            prev_state = self._prev_obs.get("state", [])
            if isinstance(prev_state, list) and len(prev_state) == len(state):
                delta_free = None
                if self._free_idx is not None:
                    try:
                        delta_free = float(state[self._free_idx]) - float(prev_state[self._free_idx])
                    except Exception:
                        delta_free = None
                bank_open_now = None
                if self._bank_idx is not None:
                    try:
                        bank_open_now = float(state[self._bank_idx]) > 0.5
                    except Exception:
                        bank_open_now = None

                # Step penalty to encourage speed
                reward -= self._step_penalty

                # Positive when inventory fills (free slots drop => delta_free < 0)
                if delta_free is not None and delta_free < -1e-6:
                    reward += (-delta_free) * self._gather_bonus
                    self._no_progress_steps = 0
                    self._progress_steps += 1

                # Strong positive when inventory empties while bank is open (deposit)
                if delta_free is not None and delta_free > 1e-6 and bank_open_now:
                    reward += (delta_free) * self._deposit_bonus
                    self._no_progress_steps = 0
                    self._progress_steps += 10

                if delta_free is None:
                    self._no_progress_steps += 1
            else:
                # No prior or shape mismatch; apply only step penalty
                reward -= self._step_penalty
                self._no_progress_steps += 1
        else:
            # Non-zero writer reward counts as progress
            if reward > 0:
                self._no_progress_steps = 0
                self._progress_steps += 1
            else:
                self._no_progress_steps += 1

        # Termination logic
        done_flag = bool(next_obs.get("done", False))
        terminated = bool(done_flag)
        truncated = False

        # If no explicit done, end episode after big deposit or when stalled too long
        if not terminated:
            if self._progress_steps >= 50:
                terminated = True  # likely completed a bank cycle
            elif self._no_progress_steps >= self._max_no_progress:
                truncated = True

        info = {
            "last_action_index": next_obs.get("last_action_index"),
            "last_action_name": next_obs.get("last_action_name"),
            "seq": self._last_seq,
            "state_names": state_names,
            "episode_return": self._last_episode_return,
            "episode_len": self._last_episode_len,
        }

        # Episode accounting
        self._last_episode_return += reward
        self._last_episode_len += 1
        if terminated or truncated:
            info["episode_return"] = self._last_episode_return
            info["episode_len"] = self._last_episode_len
            # Reset counters for next episode
            self._last_episode_return = 0.0
            self._last_episode_len = 0
            self._no_progress_steps = 0
            self._progress_steps = 0

        # Save prev obs for next-step deltas
        self._prev_obs = next_obs

        try:
            self.log.info(
                "Step -> seq=%s reward=%.3f last=%s(%s) term=%s trunc=%s",
                self._last_seq,
                reward,
                info.get("last_action_index"),
                info.get("last_action_name"),
                terminated,
                truncated,
            )
        except Exception:
            pass

        return state, reward, terminated, truncated, info

    # ----------------------------- Helpers ------------------------------------

    def render(self, mode: str = "human"):
        return None

    def close(self):
        # Nothing special, but keep for symmetry
        try:
            self.log.debug("Env closed.")
        except Exception:
            pass

    def action_name(self, idx: int) -> str:
        try:
            return self._space["actions"][int(idx)]
        except Exception:
            return f"A{idx}"


# ----------------------------- CLI Demo --------------------------------------


def _demo(env: RLBotEnv, episodes: int, max_steps: int, sleep_s: float = 0.0) -> None:
    """
    Tiny demo loop to check that rewards and terminations flow.
    It takes random valid actions (no masking here).
    """
    rng = np.random.default_rng(0)
    total = 0.0
    for ep in range(episodes):
        obs, info = env.reset()
        ep_ret = 0.0
        ep_len = 0
        for t in range(max_steps):
            a = int(rng.integers(env.action_space.n))
            obs, r, term, trunc, inf = env.step(a)
            ep_ret += r
            ep_len += 1
            if term or trunc:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)
        total += ep_ret
        env.log.info("Episode %d done: return=%.3f len=%d", ep + 1, ep_ret, ep_len)
    env.log.info("Demo finished: episodes=%d, avg_return=%.3f", episodes, total / max(1, episodes))


def main():
    parser = argparse.ArgumentParser(description="RLBotEnv IPC test runner")
    parser.add_argument("--ipc-dir", default="rlbot-ipc", help="IPC directory path")
    parser.add_argument("--log-level", default=os.getenv("RLBOT_LOG_LEVEL", "INFO"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--poll", type=float, default=0.05)
    parser.add_argument(
        "--space-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for action_space.json (negative = wait forever)",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between steps (demo only)")
    args = parser.parse_args()

    logger = _setup_logger(args.log_level)

    env = RLBotEnv(
        ipc_dir=args.ipc_dir,
        step_timeout_s=args.timeout,
        poll_interval_s=args.poll,
        action_space_timeout_s=None if args.space_timeout is not None and args.space_timeout < 0 else args.space_timeout,
        logger=logger,
    )

    try:
        _demo(env, episodes=args.episodes, max_steps=args.max_steps, sleep_s=args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
