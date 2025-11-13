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


# ----------------------------- Simple Policy ---------------------------------

class LinearSoftmaxPolicy:
    """Minimal linear softmax policy for discrete actions.

    - logits = W @ state + b
    - pi(a|s) = softmax(logits)
    """

    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-2, seed: int | None = 0, entropy_beta: float = 0.0):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.entropy_beta = float(entropy_beta)
        # Xavier init
        limit = np.sqrt(6.0 / (self.state_dim + self.n_actions))
        self.W = self.rng.uniform(-limit, limit, size=(self.n_actions, self.state_dim)).astype(np.float32)
        self.b = np.zeros((self.n_actions,), dtype=np.float32)

    def logits(self, s: np.ndarray) -> np.ndarray:
        return (self.W @ s) + self.b

    def probs(self, s: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = self.logits(s)
        t = max(1e-3, float(temperature))
        z = z / t
        z = z - np.max(z)
        e = np.exp(z)
        p = e / np.maximum(np.sum(e), 1e-8)
        return p

    def sample(self, s: np.ndarray, temperature: float = 1.0, epsilon: float = 0.0) -> tuple[int, np.ndarray]:
        p = self.probs(s, temperature=temperature)
        eps = float(np.clip(epsilon, 0.0, 1.0))
        if self.rng.random() < eps:
            a = int(self.rng.integers(0, self.n_actions))
        else:
            a = int(self.rng.choice(self.n_actions, p=p))
        return a, p

    def update_reinforce(self, states: list[np.ndarray], actions: list[int], returns: list[float]) -> None:
        # Normalize returns for stability
        G = np.asarray(returns, dtype=np.float32)
        if G.size > 1:
            G = (G - G.mean()) / (G.std() + 1e-6)
        u = np.full((self.n_actions,), 1.0 / self.n_actions, dtype=np.float32)
        for s, a, g in zip(states, actions, G):
            p = self.probs(s)
            # grad of log pi(a|s) wrt logits is (onehot(a) - p)
            grad_logits = np.zeros_like(p)
            grad_logits[a] = 1.0
            grad_logits -= p
            grad_logits *= g
            # Entropy/KL-to-uniform bonus: push distribution broader
            if self.entropy_beta > 0.0:
                grad_entropy = (u - p)  # minimize KL(p||u) == maximize entropy
                grad_logits += self.entropy_beta * grad_entropy
            # W gradient: outer(grad_logits, s)
            self.W += self.lr * np.outer(grad_logits, s)
            self.b += self.lr * grad_logits

    def save(self, path: str | os.PathLike) -> None:
        np.savez(str(path), W=self.W, b=self.b)

    def load(self, path: str | os.PathLike) -> None:
        data = np.load(str(path))
        self.W = data["W"].astype(np.float32)
        self.b = data["b"].astype(np.float32)


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
        action_preset: str | None = "core",
        allowed_actions: list[str] | None = None,
        near_full_slots: int = 2,
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
        self._raw_actions: list[str] = list(self._space.get("actions", []))
        # Filter to allowed actions (by name) if requested
        self._action_preset = (action_preset or "").strip().lower() or None
        self._allowed_actions_cfg = allowed_actions[:] if allowed_actions else None
        self._action_names, self._action_map = self._build_filtered_actions()
        self._name_to_filtered = {n: i for i, n in enumerate(self._action_names)}
        self.action_space = spaces.Discrete(len(self._action_names))
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
        # Step penalty is applied only when the writer provides zero reward
        # to avoid double-penalizing steps.
        self._step_penalty: float = 0.005
        self._deposit_bonus: float = 6.0
        self._gather_bonus: float = 1.2
        self._max_no_progress: int = 120
        self._last_episode_return: float = 0.0
        self._last_episode_len: int = 0
        # Additional shaping for WC
        self._wood_idx: int | None = None
        self._tree_idx: int | None = None
        self._bankdist_idx: int | None = None
        # Shaping weights (positive shaping will be added on top of writer reward)
        self._approach_tree_w: float = 2.0
        self._approach_bank_w: float = 4.0
        self._avoid_bank_w: float = 0.2
        # Banking should trigger only when inventory is FULL.
        # We still accept a parameter for compatibility but always use strict-full detection.
        self._full_eps: float = 0.01  # free_slots_bucket <= 0.01 -> exactly 0 free slots
        self._wood_start_bonus: float = 1.5
        self._wood_tick_bonus: float = 0.25
        self._xp_bonus_per_point: float = 0.05  # applied if woodcut_xp present in obs
        # Post-deposit guidance to return to trees
        self._post_deposit_window: int = 0
        self._post_deposit_window_steps: int = 60
        self._post_deposit_bonus: float = 2.0
        # Bank proximity nudges
        self._bank_near_thresh: float = 0.2
        self._near_bank_bonus: float = 0.6
        self._bank_task_bonus: float = 0.4
        self._avoid_leaving_bank_w: float = 0.35
        # Encourage navigating to trees when far, discourage chopping when far
        self._nav_far_bonus: float = 0.15
        self._chop_far_penalty: float = 0.30
        # Strong near-full heuristics to bias learning even when starting full
        self._near_full_penalty_chop: float = 2.0
        self._near_full_penalty_camera: float = 0.4
        self._near_full_bonus_bankflow: float = 1.5
        self._near_full_dwell_bank_bonus: float = 0.15
        self._near_full_stall_limit: int = 40  # shorten stall horizon when near-full

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

    def _build_filtered_actions(self) -> tuple[list[str], list[int]]:
        raw = self._raw_actions or []
        if not raw:
            return [], []
        # Determine desired names
        desired: list[str]
        if self._allowed_actions_cfg is not None:
            desired = [str(x) for x in self._allowed_actions_cfg]
        elif self._action_preset == "core":
            desired = [
                "BankWhenFullTask",
                "ChopNearestTreeTask",
                "NavigateToTreeHotspotTask",
                "IdleTask",
                "CameraRotateLeftTask",
                "CameraRotateRightTask",
                "CameraZoomInTask",
                "CameraZoomOutTask",
            ]
        else:
            # No filtering
            desired = raw
        # Build mapping from filtered to raw indices
        name_to_raw = {n: i for i, n in enumerate(raw)}
        filtered: list[str] = []
        fmap: list[int] = []
        for name in desired:
            if name in name_to_raw:
                filtered.append(name)
                fmap.append(name_to_raw[name])
        # If nothing matched, fall back to raw
        if not filtered:
            filtered = raw
            fmap = list(range(len(raw)))
        try:
            self.log.info("Using filtered actions: %s", ", ".join(filtered))
        except Exception:
            pass
        return filtered, fmap

    # ----------------------- File/Space helpers -----------------------

    def _load_action_space(self, block: bool = True) -> Dict[str, Any]:
        space_file = self.dir / "action_space.json"
        t0 = time.time()
        while True:
            try:
                # If the configured dir doesn't have the file yet, try common fallbacks.
                # Prefer the most recently updated candidate to avoid stale files.
                if not space_file.exists():
                    candidates = []
                    try:
                        candidates.append(Path.cwd() / "rlbot-ipc")
                    except Exception:
                        pass
                    try:
                        candidates.append(Path.home() / ".runelite" / "rlbot-ipc")
                    except Exception:
                        pass
                    try:
                        candidates.append(Path(tempfile.gettempdir()) / "rlbot-ipc")
                    except Exception:
                        pass

                    best_dir = None
                    best_mtime = -1.0
                    for c in candidates:
                        if not c or c == self.dir:
                            continue
                        sf = c / "action_space.json"
                        if sf.exists():
                            try:
                                m = sf.stat().st_mtime
                            except Exception:
                                m = 0.0
                            if m > best_mtime:
                                best_mtime = m
                                best_dir = c
                    if best_dir is not None:
                        self.dir = best_dir
                        self._obs_file = self.dir / "obs.json"
                        self._action_file = self.dir / "action.json"
                        space_file = self.dir / "action_space.json"
                        try:
                            self.log.info("Auto-detected IPC dir: %s", str(self.dir))
                        except Exception:
                            pass

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
        try:
            raw_idx = int(action_index)
            # Find filtered index if exists
            filt_idx = None
            try:
                filt_idx = self._action_map.index(raw_idx)  # type: ignore[attr-defined]
            except Exception:
                filt_idx = None
            if filt_idx is not None and 0 <= filt_idx < len(getattr(self, "_action_names", [])):
                self.log.debug(
                    "Wrote action seq=%s -> raw=%s(%s) filtered=%s(%s)",
                    seq,
                    raw_idx,
                    (self._raw_actions[raw_idx] if 0 <= raw_idx < len(self._raw_actions) else str(raw_idx)),
                    filt_idx,
                    (self._action_names[filt_idx] if 0 <= filt_idx < len(self._action_names) else str(filt_idx)),
                )
            else:
                self.log.debug(
                    "Wrote action seq=%s -> raw=%s(%s)",
                    seq,
                    raw_idx,
                    (self._raw_actions[raw_idx] if 0 <= raw_idx < len(self._raw_actions) else str(raw_idx)),
                )
        except Exception:
            self.log.debug("Wrote action seq=%s -> %s", seq, action_index)

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

        # Reset episode helpers; seed prev_obs so step() can gate on immediate state
        self._prev_obs = obs
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

        # Map filtered action index -> raw agent index and write for the writer to consume
        try:
            a_filtered = int(action)
        except Exception:
            a_filtered = 0
        mapping = getattr(self, "_action_map", None)
        # Heuristic action gating: if previous obs indicates inventory is FULL and bank is not open,
        # force the action to BankWhenFullTask to avoid futile chopping/camera.
        try:
            prev = self._prev_obs
            if prev is not None:
                state_prev = prev.get("state", [])
                bank_open_prev = False
                near_full_prev = False
                if isinstance(state_prev, list):
                    if getattr(self, "_free_idx", None) is not None and 0 <= self._free_idx < len(state_prev):
                        try:
                            near_full_prev = float(state_prev[self._free_idx]) <= self._full_eps
                        except Exception:
                            near_full_prev = False
                    if getattr(self, "_bank_idx", None) is not None and 0 <= self._bank_idx < len(state_prev):
                        try:
                            bank_open_prev = float(state_prev[self._bank_idx]) > 0.5
                        except Exception:
                            bank_open_prev = False
                if near_full_prev and not bank_open_prev:
                    bank_idx_f = self._name_to_filtered.get("BankWhenFullTask")
                    if bank_idx_f is not None and a_filtered != bank_idx_f:
                        try:
                            self.log.debug("Overriding action -> BankWhenFullTask due to near-full prev state")
                        except Exception:
                            pass
                        a_filtered = int(bank_idx_f)
        except Exception:
            pass

        a_raw = int(mapping[a_filtered]) if (isinstance(mapping, list) and 0 <= a_filtered < len(mapping)) else a_filtered
        self._write_action(self._last_seq, int(a_raw))

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

        # Prefer writer's reward (base). We'll add our shaping on top of it.
        base_reward = float(next_obs.get("last_reward", 0.0) or 0.0)
        reward = base_reward

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
        if self._wood_idx is None and state_names:
            try:
                self._wood_idx = state_names.index("is_woodcutting")
            except ValueError:
                self._wood_idx = None
        if self._tree_idx is None and state_names:
            try:
                self._tree_idx = state_names.index("tree_distance_norm")
            except ValueError:
                self._tree_idx = None
        if self._bankdist_idx is None and state_names:
            try:
                self._bankdist_idx = state_names.index("bank_distance_norm")
            except ValueError:
                self._bankdist_idx = None

        # Determine FULL inventory from state for global shaping (strict-only)
        near_full = False
        try:
            if self._free_idx is not None:
                free_bucket = float(state[self._free_idx])
                near_full = free_bucket <= self._full_eps
        except Exception:
            near_full = False

        # Immediate shaping: if far from any tree, nudge policy choices
        try:
            last_name = (next_obs.get("last_action_name") or "").strip()
            if self._tree_idx is not None:
                tree_norm = float(state[self._tree_idx])
                far_from_tree = tree_norm >= 0.95  # 1.0 means unknown/very far
                if far_from_tree:
                    if last_name == "ChopNearestTreeTask":
                        reward -= self._chop_far_penalty
                    elif last_name == "NavigateToTreeHotspotTask":
                        reward += self._nav_far_bonus
        except Exception:
            pass

        # Near-full immediate action shaping: heavily discourage chopping/camera, encourage bank flow
        try:
            if near_full:
                last_name = (next_obs.get("last_action_name") or "").strip()
                if last_name in ("ChopNearestTreeTask", "NavigateToTreeHotspotTask"):
                    reward -= self._near_full_penalty_chop
                elif last_name in ("CameraRotateLeftTask", "CameraRotateRightTask", "CameraZoomInTask", "CameraZoomOutTask", "IdleTask"):
                    reward -= self._near_full_penalty_camera
                elif last_name in ("BankWhenFullTask", "BankDepositTask"):
                    reward += self._near_full_bonus_bankflow
                    # small dwell bonus while persisting with bank flow to encourage staying on path
                    reward += self._near_full_dwell_bank_bonus
        except Exception:
            pass

        # Compute shaping from deltas and context. Always applied.
        # If writer reward is zero, we also apply a small step penalty to
        # encourage quicker progress.
        if self._prev_obs is not None:
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

                # Step penalty only if writer gave no signal this step
                if base_reward == 0.0:
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
                    # Activate post-deposit window to encourage return to trees
                    self._post_deposit_window = self._post_deposit_window_steps

                # Reward starting woodcutting and staying in woodcutting
                wood_prev = None
                wood_now = None
                if self._wood_idx is not None:
                    try:
                        wood_prev = float(prev_state[self._wood_idx]) > 0.5
                        wood_now = float(state[self._wood_idx]) > 0.5
                    except Exception:
                        wood_prev = wood_now = None
                if wood_prev is not None and wood_now is not None:
                    if (not wood_prev) and wood_now:
                        reward += self._wood_start_bonus
                        self._no_progress_steps = 0
                        self._progress_steps += 5
                    if wood_now:
                        reward += self._wood_tick_bonus

                # Reward approaching nearest tree when inventory not full, bank closed, not already chopping
                if self._tree_idx is not None and self._bank_idx is not None and self._free_idx is not None:
                    try:
                        tree_prev = float(prev_state[self._tree_idx])
                        tree_now = float(state[self._tree_idx])
                        bank_open_flag = float(state[self._bank_idx]) > 0.5
                        free_bucket = float(state[self._free_idx])
                        near_full = free_bucket <= self._full_eps
                        if (not bank_open_flag) and (wood_now is False or wood_now is None) and not near_full:
                            # smaller tree_distance_norm is better; reward positive decrease
                            delta_tree = (tree_prev - tree_now)
                            if delta_tree > 0.0:
                                reward += self._approach_tree_w * delta_tree
                                self._no_progress_steps = 0
                                self._progress_steps += 1
                            # Small bonus for explicitly choosing a tree-approach action when inventory not full
                            last_name = (next_obs.get("last_action_name") or "").strip()
                            if last_name in ("ChopNearestTreeTask", "NavigateToTreeHotspotTask"):
                                reward += 0.05
                    except Exception:
                        pass

                # Encourage approaching bank when inventory is near-full and bank not open
                if self._bankdist_idx is not None and self._free_idx is not None and self._bank_idx is not None:
                    try:
                        bank_prev = float(prev_state[self._bankdist_idx])
                        bank_now = float(state[self._bankdist_idx])
                        free_bucket = float(state[self._free_idx])
                        bank_open_flag = float(state[self._bank_idx]) > 0.5
                        near_full = free_bucket <= self._full_eps
                        if near_full and not bank_open_flag:
                            delta_bank = (bank_prev - bank_now)  # positive when approaching bank
                            if delta_bank > 0.0:
                                reward += self._approach_bank_w * delta_bank
                                self._no_progress_steps = 0
                                self._progress_steps += 1
                            else:
                                # Penalize dithering when moving away from bank while near-full
                                if (bank_now - bank_prev) > 0.0:
                                    reward -= 0.3 * (bank_now - bank_prev)
                            # If already near the bank, nudge the agent to perform the bank task
                            if bank_now < self._bank_near_thresh:
                                reward += self._near_bank_bonus
                                # Optional: small bonus if trying the right task
                                last_name = next_obs.get("last_action_name") or ""
                                if last_name in ("BankDepositTask", "NavigateToBankHotspotTask", "BankWhenFullTask"):
                                    reward += self._bank_task_bonus
                        # Small positive for having bank UI open while near-full
                        if near_full and bank_open_flag:
                            reward += 0.2
                        # Penalize leaving the bank area when near-full and bank closed
                        if near_full and not bank_open_flag:
                            delta_bank_away = (bank_now - bank_prev)  # positive when moving away
                            if delta_bank_away > 0.0:
                                reward -= self._avoid_leaving_bank_w * delta_bank_away
                        elif (not near_full) and not bank_open_flag:
                            # Slightly penalize approaching bank when we have space
                            delta_bank = (bank_prev - bank_now)
                            if delta_bank > 0.0:
                                reward -= self._avoid_bank_w * delta_bank
                    except Exception:
                        pass
                # Discourage chopping/approaching trees when inventory is near-full (should bank first)
                try:
                    last_name = next_obs.get("last_action_name") or ""
                    if self._free_idx is not None:
                        free_bucket = float(state[self._free_idx])
                        near_full = free_bucket <= self._full_eps
                        if near_full:
                            # Penalize explicit chop attempts and navigation to trees when near-full
                            if last_name in ("ChopNearestTreeTask", "NavigateToTreeHotspotTask"):
                                reward -= self._near_full_penalty_chop
                            # Penalize being in woodcutting state while near-full
                            if self._wood_idx is not None:
                                try:
                                    wood_now_nf = float(state[self._wood_idx]) > 0.5
                                except Exception:
                                    wood_now_nf = False
                                if wood_now_nf:
                                    reward -= 0.6
                            # Penalize getting closer to trees while near-full
                            if self._tree_idx is not None:
                                try:
                                    tree_prev = float(prev_state[self._tree_idx])
                                    tree_now = float(state[self._tree_idx])
                                    delta_tree_closer = (tree_prev - tree_now)
                                    if delta_tree_closer > 0.0:
                                        reward -= 0.35 * delta_tree_closer
                                except Exception:
                                    pass
                except Exception:
                    pass

                # Optional: reward woodcutting XP gain if writer includes it
                try:
                    xp_prev = self._prev_obs.get("woodcut_xp", self._prev_obs.get("woodcutXp"))
                    xp_now = next_obs.get("woodcut_xp", next_obs.get("woodcutXp"))
                    if xp_prev is not None and xp_now is not None:
                        dxp = float(xp_now) - float(xp_prev)
                        if dxp > 0:
                            reward += dxp * self._xp_bonus_per_point
                            self._no_progress_steps = 0
                            self._progress_steps += 2
                except Exception:
                    pass

                if delta_free is None and base_reward <= 0.0:
                    self._no_progress_steps += 1
            else:
                # No prior or shape mismatch; only penalize if no writer reward
                if base_reward == 0.0:
                    reward -= self._step_penalty
                    self._no_progress_steps += 1
        else:
            # First step after reset; if writer gave a positive base reward, count as progress
            if base_reward > 0:
                self._no_progress_steps = 0
                self._progress_steps += 1
            else:
                # Apply immediate near-full shaping on first step to steer choice toward banking
                try:
                    if near_full:
                        last_name = (next_obs.get("last_action_name") or "").strip()
                        if last_name in ("BankWhenFullTask", "BankDepositTask"):
                            reward += self._near_full_bonus_bankflow
                            self._progress_steps += 1
                            self._no_progress_steps = 0
                        elif last_name in ("ChopNearestTreeTask", "NavigateToTreeHotspotTask"):
                            reward -= self._near_full_penalty_chop
                            self._no_progress_steps += 1
                        else:
                            self._no_progress_steps += 1
                    else:
                        self._no_progress_steps += 1
                except Exception:
                    self._no_progress_steps += 1

        # Termination logic
        done_flag = bool(next_obs.get("done", False))
        terminated = bool(done_flag)
        truncated = False

        # Post-deposit guidance: if window active, boost tree approach and end episode once chopping resumes
        try:
            if self._post_deposit_window > 0:
                self._post_deposit_window -= 1
                # Encourage moving away from bank toward trees during this window
                if self._tree_idx is not None and self._bank_idx is not None:
                    try:
                        tree_prev = float(self._prev_obs["state"][self._tree_idx]) if self._prev_obs else None
                        tree_now = float(state[self._tree_idx])
                        bank_open_flag = float(state[self._bank_idx]) > 0.5
                        if tree_prev is not None and not bank_open_flag:
                            delta_tree = (tree_prev - tree_now)
                            if delta_tree > 0.0:
                                reward += 0.5 * self._approach_tree_w * delta_tree
                                self._no_progress_steps = 0
                                self._progress_steps += 1
                    except Exception:
                        pass
                # If chopping resumes, give a bonus and end episode (full cycle complete)
                try:
                    wood_flag = (self._wood_idx is not None) and (float(state[self._wood_idx]) > 0.5)
                except Exception:
                    wood_flag = False
                if wood_flag:
                    reward += self._post_deposit_bonus
                    terminated = True
                    self._post_deposit_window = 0
        except Exception:
            pass

        # If no explicit done, end episode after big deposit or when stalled too long
        if not terminated:
            if self._progress_steps >= 50:
                terminated = True  # likely completed a bank cycle
            else:
                # Shorter stall horizon when inventory is near-full
                stall_limit = self._max_no_progress
                try:
                    if near_full:
                        stall_limit = min(stall_limit, int(self._near_full_stall_limit))
                except Exception:
                    pass
                if self._no_progress_steps >= stall_limit:
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
            # Return filtered name if idx is filtered; otherwise raw
            if 0 <= int(idx) < len(self._action_names):
                return self._action_names[int(idx)]
            return self._space["actions"][int(idx)]
        except Exception:
            return f"A{idx}"


# ----------------------------- CLI Demo --------------------------------------


def _train(env: RLBotEnv, episodes: int, max_steps: int, lr: float, save_path: str | None = None, greedy: bool = True,
           temp_start: float = 1.5, temp_end: float = 0.7, temp_decay_episodes: int = 100,
           eps_start: float = 0.15, eps_end: float = 0.05, eps_decay_episodes: int = 150,
           entropy_beta: float = 0.02) -> None:
    state_dim = int(env._space.get("state_dim", 0))
    n_actions = int(env.action_space.n)
    policy = LinearSoftmaxPolicy(state_dim=state_dim, n_actions=n_actions, lr=lr, seed=0, entropy_beta=entropy_beta)

    best_return = -1e9
    best_path = save_path or (str(_resolve_repo_root() / "rlbot" / "models" / "wc_policy.npz"))
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    for ep in range(episodes):
        s, info = env.reset()
        traj_s: list[np.ndarray] = []
        traj_a: list[int] = []
        traj_r: list[float] = []

        ep_ret = 0.0
        ep_len = 0
        # Anneal temperature and epsilon per-episode
        frac_t = min(1.0, ep / max(1, temp_decay_episodes))
        frac_e = min(1.0, ep / max(1, eps_decay_episodes))
        temperature = float(temp_start + (temp_end - temp_start) * frac_t)
        epsilon = float(eps_start + (eps_end - eps_start) * frac_e)
        try:
            env.log.debug("Episode %d: temp=%.3f eps=%.3f", ep + 1, temperature, epsilon)
        except Exception:
            pass
        for t in range(max_steps):
            if greedy:
                p = policy.probs(s)
                a = int(np.argmax(p))
            else:
                a, _ = policy.sample(s, temperature=temperature, epsilon=epsilon)
            s_next, r, term, trunc, inf = env.step(a)
            traj_s.append(s)
            traj_a.append(a)
            traj_r.append(r)
            ep_ret += r
            ep_len += 1
            s = s_next
            if term or trunc:
                break

        # Compute returns (discount=1.0)
        G = []
        g = 0.0
        for r in reversed(traj_r):
            g = r + g
            G.append(g)
        G.reverse()

        # Update policy
        policy.update_reinforce(traj_s, traj_a, G)

        if ep_ret > best_return and best_path:
            policy.save(best_path)
            best_return = ep_ret

        try:
            env.log.info("Train ep %d/%d: return=%.3f len=%d best=%.3f saved=%s",
                         ep + 1, episodes, ep_ret, ep_len, best_return,
                         os.path.basename(best_path) if best_path else None)
        except Exception:
            pass

    if best_path:
        env.log.info("Training complete. Best policy saved to %s (return=%.3f)", best_path, best_return)


def _evaluate(env: RLBotEnv, episodes: int, max_steps: int, load_path: str) -> None:
    state_dim = int(env._space.get("state_dim", 0))
    n_actions = int(env.action_space.n)
    policy = LinearSoftmaxPolicy(state_dim=state_dim, n_actions=n_actions, lr=1e-3)
    policy.load(load_path)

    total = 0.0
    for ep in range(episodes):
        s, info = env.reset()
        ep_ret = 0.0
        ep_len = 0
        for t in range(max_steps):
            # Greedy action at eval time
            p = policy.probs(s)
            a = int(np.argmax(p))
            s_next, r, term, trunc, inf = env.step(a)
            ep_ret += r
            ep_len += 1
            s = s_next
            if term or trunc:
                break
        total += ep_ret
        env.log.info("Eval episode %d: return=%.3f len=%d", ep + 1, ep_ret, ep_len)
    env.log.info("Eval finished: episodes=%d, avg_return=%.3f", episodes, total / max(1, episodes))


def main():
    parser = argparse.ArgumentParser(description="RLBotEnv trainer/evaluator")
    parser.add_argument("--ipc-dir", default="rlbot-ipc", help="IPC directory path")
    parser.add_argument("--log-level", default=os.getenv("RLBOT_LOG_LEVEL", "INFO"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--poll", type=float, default=0.05)
    parser.add_argument(
        "--space-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for action_space.json (negative = wait forever)",
    )
    parser.add_argument("--action-preset", choices=["core", "all"], default="core", help="Action subset to use")
    parser.add_argument("--actions", type=str, default=None, help="Comma-separated action names to allow (overrides preset)")
    parser.add_argument("--near-full-slots", type=int, default=2, help="Consider inventory near-full at or below this many free slots (default 2)")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run training or evaluation")
    parser.add_argument("--greedy", action="store_true", default=True, help="Use greedy (argmax) actions during training")
    parser.add_argument("--stochastic", dest="greedy", action="store_false", help="Use stochastic sampling during training")
    parser.add_argument("--save", type=str, default=None, help="Path to save trained policy (npz) [train]")
    parser.add_argument("--load", type=str, default=None, help="Path to load policy for warm start/eval (npz)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for policy training")
    parser.add_argument("--temp-start", type=float, default=1.5, help="Softmax temperature at episode 0 (stochastic mode)")
    parser.add_argument("--temp-end", type=float, default=0.7, help="Softmax temperature after decay (stochastic mode)")
    parser.add_argument("--temp-decay-episodes", type=int, default=100, help="Episodes over which to anneal temperature")
    parser.add_argument("--eps-start", type=float, default=0.15, help="Epsilon-greedy start (stochastic mode)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Epsilon-greedy end (stochastic mode)")
    parser.add_argument("--eps-decay-episodes", type=int, default=150, help="Episodes over which to anneal epsilon")
    parser.add_argument("--entropy-beta", type=float, default=0.02, help="Entropy regularization strength (REINFORCE)")
    args = parser.parse_args()

    logger = _setup_logger(args.log_level)

    # Resolve allowed actions
    allowed_actions = None
    if args.actions:
        allowed_actions = [s.strip() for s in args.actions.split(",") if s.strip()]

    env = RLBotEnv(
        ipc_dir=args.ipc_dir,
        step_timeout_s=args.timeout,
        poll_interval_s=args.poll,
        action_space_timeout_s=None if args.space_timeout is not None and args.space_timeout < 0 else args.space_timeout,
        action_preset=args.action_preset,
        allowed_actions=allowed_actions,
        near_full_slots=args.near_full_slots,
        logger=logger,
    )

    try:
        if args.mode == "train":
            # Optional warm start if --load provided
            if args.load:
                # Load into a temp policy and copy weights into a fresh trainer
                state_dim = int(env._space.get("state_dim", 0))
                n_actions = int(env.action_space.n)
                warm = LinearSoftmaxPolicy(state_dim, n_actions, lr=args.lr)
                warm.load(args.load)
                # Save as initial weights so _train picks it up automatically
                init_path = args.save or (str(_resolve_repo_root() / "rlbot" / "models" / "wc_policy_init.npz"))
                os.makedirs(os.path.dirname(init_path), exist_ok=True)
                warm.save(init_path)
            _train(
                env,
                episodes=args.episodes,
                max_steps=args.max_steps,
                lr=args.lr,
                save_path=args.save,
                greedy=args.greedy,
                temp_start=args.temp_start,
                temp_end=args.temp_end,
                temp_decay_episodes=args.temp_decay_episodes,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_episodes=args.eps_decay_episodes,
                entropy_beta=args.entropy_beta,
            )
        else:  # eval
            if not args.load:
                raise SystemExit("--load is required for --mode eval")
            _evaluate(env, episodes=args.episodes, max_steps=args.max_steps, load_path=args.load)
    finally:
        env.close()


if __name__ == "__main__":
    main()
