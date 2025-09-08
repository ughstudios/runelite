import json
import os
import time
from pathlib import Path

try:
    import gym
    from gym import spaces
except Exception:  # gymnasium fallback
    import gymnasium as gym
    from gymnasium import spaces


class RLBotEnv(gym.Env):
    """OpenAI Gym-like environment that drives the RuneLite RLBot plugin via file-based IPC.

    Protocol directory contains:
      - action_space.json
      - obs.json (written by plugin each tick)
      - action.json (written by this env on step)
    """

    metadata = {"render.modes": []}

    def __init__(self, ipc_dir: str = "rlbot-ipc", step_timeout_s: float = 5.0, poll_interval_s: float = 0.05):
        super().__init__()
        self.dir = Path(ipc_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._space = self._load_action_space(block=True)
        self.action_space = spaces.Discrete(len(self._space["actions"]))
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(self._space["state_dim"],), dtype=float)
        self._last_seq = None
        self._timeout = float(step_timeout_s)
        self._poll = float(poll_interval_s)

    def _load_action_space(self, block: bool):
        f = self.dir / "action_space.json"
        if block:
            wait_s = 0.0
            while not f.exists() and wait_s < 10.0:
                time.sleep(0.1)
                wait_s += 0.1
        if not f.exists():
            raise RuntimeError(f"action_space.json not found in {self.dir.resolve()}")
        with f.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "state_dim" not in data or "actions" not in data:
            raise RuntimeError("Invalid action_space.json")
        return data

    def _read_obs_blocking(self, min_next_seq: int | None = None):
        f = self.dir / "obs.json"
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                with f.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                seq = int(data.get("seq", -1))
                if min_next_seq is None or seq >= min_next_seq:
                    return data
            except Exception:
                pass
            time.sleep(self._poll)
        raise TimeoutError("Timed out waiting for observation")

    def reset(self, *, seed=None, options=None):  # gym/gymnasium compatible
        super().reset(seed=seed)
        # Reload action space in case plugin restarted
        try:
            self._space = self._load_action_space(block=True)
            self.action_space = spaces.Discrete(len(self._space["actions"]))
            self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(self._space["state_dim"],), dtype=float)
        except Exception:
            pass
        obs = self._read_obs_blocking()
        self._last_seq = int(obs["seq"])
        state = obs.get("state", [])
        return state, {}

    def step(self, action: int):
        if self._last_seq is None:
            raise RuntimeError("Call reset() before step()")
        # Write action for current sequence
        a = int(action)
        act_file = self.dir / "action.json"
        payload = {"seq": int(self._last_seq), "action": a}
        tmp = self.dir / "action.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(act_file)

        # Wait for next observation (seq strictly greater)
        next_obs = self._read_obs_blocking(min_next_seq=self._last_seq + 1)
        self._last_seq = int(next_obs["seq"])

        state = next_obs.get("state", [])
        reward = float(next_obs.get("last_reward", 0.0))
        terminated = False
        truncated = False
        info = {
            "last_action_index": next_obs.get("last_action_index"),
            "last_action_name": next_obs.get("last_action_name"),
            "seq": self._last_seq,
        }
        return state, reward, terminated, truncated, info

    def render(self, mode="human"):
        return None


if __name__ == "__main__":
    # Quick manual test loop that prints rewards
    env = RLBotEnv(os.environ.get("RLBOT_IPC", "rlbot-ipc"))
    obs, _ = env.reset()
    print("Action space:", env._space["actions"])  # noqa: T201
    while True:
        # Naive action 0 loop; adapt to your desired policy
        obs, reward, term, trunc, info = env.step(0)
        print(f"seq={info['seq']} reward={reward:.3f} last={info['last_action_name']}")  # noqa: T201
        time.sleep(0.25)

