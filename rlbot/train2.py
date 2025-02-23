#!/usr/bin/env python3
"""
RuneScape AI Training Bot using PPO with Stable Baselines3.

This module defines a custom Gymnasium environment for RuneScape and associated training/testing
utilities for training a combat bot. It leverages PPO from Stable Baselines3 and logs custom metrics
to TensorBoard.
"""

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

import asyncio
import base64
import gc
import json
import logging
import os
import random
import threading
import time
import argparse
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import nest_asyncio
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage
import websockets
from jsonschema import validate, ValidationError
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -----------------------------------------------------------------------------
# Logging & Console Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.ERROR,  # Only show errors by default
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "rlbot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# -----------------------------------------------------------------------------
# Action Definitions
# -----------------------------------------------------------------------------
class Action(Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    DO_NOTHING = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6
    ZOOM_IN = 7
    ZOOM_OUT = 8
    ATTACK = 9

# -----------------------------------------------------------------------------
# Training Callback for Custom TensorBoard Logging
# -----------------------------------------------------------------------------
class TrainingCallback(BaseCallback):
    """
    Custom callback for logging metrics to TensorBoard during training.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.episode_count = 0
        self.current_reward = 0.0
        self.rewards: List[float] = []
        self.last_log_time = time.time()
        self.log_interval = 300  # Increased from 60 to 300 seconds (5 minutes)
        self.writer: Optional[SummaryWriter] = None
        self.step_in_episode = 0
        self.action_counts = {action.name: 0 for action in Action}
        self.last_screenshot_log = 0
        self.screenshot_log_interval = 2000  # Increased from 500 to 2000 steps

    def _init_callback(self) -> None:
        """Initialize the TensorBoard writer."""
        if not getattr(self.model, "tensorboard_log", None):
            return
        tb_log_dir = str(self.model.tensorboard_log)
        log_dir = os.path.join(tb_log_dir, "metrics")  # Simple, clean directory name
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, comment="")

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        """Helper method to log a scalar value to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def _on_step(self) -> bool:
        """Called after each environment step."""
        try:
            infos = self.locals.get("infos", [])
            info = infos[0] if infos else {}
            rewards = self.locals.get("rewards", [])
            reward = rewards[0] if rewards else 0.0
            new_obs = self.locals.get("new_obs", None)
            obs = new_obs[0] if isinstance(new_obs, (list, np.ndarray)) else {}
            actions = self.locals.get("actions", None)
            action = actions[0] if isinstance(actions, (list, np.ndarray)) else None

            self.current_reward += reward
            self.step_in_episode += 1
            timestep = self.num_timesteps

            # Update action frequency count
            if action is not None:
                try:
                    if isinstance(action, (int, np.integer)):
                        action_enum = Action(int(action))
                        self.action_counts[action_enum.name] += 1
                    else:
                        self.action_counts[str(action)] = self.action_counts.get(str(action), 0) + 1
                except Exception as e:
                    logger.error(f"Error updating action_counts: {e}")

            # Log to TensorBoard
            if self.writer and timestep % 1 == 0:  # Log every step for basic metrics
                # Basic metrics
                self._log_scalar("rewards/step", reward, timestep)
                self._log_scalar("rewards/cumulative", self.current_reward, timestep)
                self._log_scalar("episode/steps", self.step_in_episode, timestep)

                # Get state information
                state = info.get("state", {})
                player = state.get("player", {})

                # Player Stats (Detailed)
                health = player.get("health", {})
                current_health = int(health.get("current", 100))
                max_health = int(health.get("maximum", 100))
                health_ratio = current_health / max_health if max_health > 0 else 1.0

                self._log_scalar("player/health_current", current_health, timestep)
                self._log_scalar("player/health_max", max_health, timestep)
                self._log_scalar("player/health_ratio", health_ratio, timestep)
                self._log_scalar("player/prayer", player.get("prayer", 0), timestep)
                self._log_scalar("player/run_energy", player.get("runEnergy", 0.0), timestep)
                self._log_scalar("player/in_combat", int(player.get("inCombat", False)), timestep)

                # Combat Stats (Individual Skills)
                skills = player.get("skills", {})
                for skill_name, skill_data in skills.items():
                    self._log_scalar(f"skills/{skill_name.lower()}/level", skill_data.get("level", 1), timestep)
                    self._log_scalar(f"skills/{skill_name.lower()}/experience", skill_data.get("experience", 0), timestep)

                # NPC Information (Detailed)
                npcs = state.get("npcs", [])
                if npcs:
                    nearest_npc = min(npcs, key=lambda x: x.get("distance", float("inf")))
                    self._log_scalar("npcs/nearest_distance", nearest_npc.get("distance", 0), timestep)
                    self._log_scalar("npcs/nearest_level", nearest_npc.get("level", 0), timestep)
                    self._log_scalar("npcs/count", len(npcs), timestep)
                    self._log_scalar("npcs/nearest_health_ratio",
                        nearest_npc.get("health", {}).get("current", 0) / max(nearest_npc.get("health", {}).get("maximum", 1), 1),
                        timestep)

                # Location and Movement
                location = player.get("location", {})
                self._log_scalar("location/x", location.get("x", 0), timestep)
                self._log_scalar("location/y", location.get("y", 0), timestep)
                self._log_scalar("location/plane", location.get("plane", 0), timestep)

                # Action Information
                if action is not None:
                    if isinstance(action, (int, np.integer)):
                        self._log_scalar("actions/last_action_scalar", int(action), timestep)
                        if self.writer:
                            self.writer.add_text("actions/last_action_text", Action(action).name, timestep)
                    else:
                        if self.writer:
                            self.writer.add_text("actions/last_action", str(action), timestep)

                # Action frequencies
                if timestep % 1000 == 0:  # Log action frequencies periodically
                    for action_name, freq in self.action_counts.items():
                        self._log_scalar(f"actions/frequency/{action_name}", freq, timestep)

                # Observation Space Logging
                if isinstance(obs, dict):
                    # Log continuous observations
                    for key in ["player_position", "player_run_energy", "exploration_score"]:
                        if key in obs:
                            if isinstance(obs[key], np.ndarray):
                                for i, val in enumerate(obs[key].flatten()):
                                    self._log_scalar(f"observations/{key}_{i}", float(val), timestep)

                    # Log discrete observations
                    for key in ["in_combat", "interfaces_open", "path_obstructed"]:
                        if key in obs:
                            self._log_scalar(f"observations/{key}", int(obs[key][0]), timestep)

                    # Log NPC observations
                    if "npcs" in obs and isinstance(obs["npcs"], np.ndarray):
                        for i in range(min(3, obs["npcs"].shape[0])):  # Log first 3 NPCs
                            npc = obs["npcs"][i]
                            self._log_scalar(f"observations/npc_{i}_distance", float(npc[2]), timestep)
                            self._log_scalar(f"observations/npc_{i}_level", float(npc[1]), timestep)

                # Screenshot logging
                if timestep - self.last_screenshot_log >= self.screenshot_log_interval:
                    try:
                        if isinstance(obs, dict) and "screenshot" in obs:
                            screenshot_data = obs["screenshot"]
                            if isinstance(screenshot_data, np.ndarray):
                                if screenshot_data.shape[-1] == 3:  # Ensure it's RGB
                                    screenshot_normalized = screenshot_data.astype(np.float32) / 255.0
                                    screenshot_chw = np.transpose(screenshot_normalized, (2, 0, 1))
                                    if self.writer:
                                        self.writer.add_image("images/screenshot", screenshot_chw, timestep, dataformats="CHW")
                                        self.writer.flush()
                                    self.last_screenshot_log = timestep
                                else:
                                    logger.error(f"Invalid screenshot channels: {screenshot_data.shape}")
                            else:
                                logger.error(f"Screenshot is not a numpy array: {type(screenshot_data)}")
                        else:
                            logger.error("No screenshot in observation")
                    except Exception as e:
                        logger.error(f"Error logging screenshot: {str(e)}", exc_info=True)

                # Experience gains
                if info.get("exp_gain", 0) > 0:
                    self._log_scalar("rewards/exp_gain", info["exp_gain"], timestep)

                # Environment Status
                self._log_scalar("environment/interfaces_open", int(info.get("interfaces_open", False)), timestep)
                self._log_scalar("environment/path_obstructed", int(info.get("path_obstructed", False)), timestep)

            # Log significant events with reduced frequency
            if info.get("exp_gain", 0) > 5000:  # Increased threshold
                logger.warning(f"Major exp gain: {info['exp_gain']:,}")

            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                logger.warning(f"Status - Episode: {self.episode_count}, Current Reward: {self.current_reward:.2f}")
                self.last_log_time = current_time

            # Clear memory less frequently
            if timestep % 5000 == 0:  # Increased from 1000
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

        except Exception as e:
            logger.error(f"Error in callback: {e}", exc_info=True)
        return True

    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout."""
        self.episode_count += 1
        self.step_in_episode = 0

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if self.writer:
            self.writer.add_scalar("episode/total_reward", self.current_reward, self.episode_count)
            self.writer.add_scalar("episode/length", self.step_in_episode, self.episode_count)
            self.writer.flush()
        self.rewards.append(self.current_reward)
        self.current_reward = 0.0
        # Clear memory at the end of each rollout
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _on_training_end(self) -> None:
        """Cleanup when training ends."""
        if self.writer:
            self.writer.close()
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

# -----------------------------------------------------------------------------
# Device Utility
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.warning(f"Using CUDA device: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.warning("Using Apple MPS (Metal) device")
    else:
        device = "cpu"
        logger.warning("Using CPU device")
    return torch.device(device)

# -----------------------------------------------------------------------------
# Training & Testing Functions
# -----------------------------------------------------------------------------
def make_env(task: str = "combat") -> gym.Env:
    """Create and wrap the RuneScape environment."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    env = RuneScapeEnv(task=task)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"), allow_early_resets=True)
    return env

def train_combat_bot(total_timesteps: int = 1_000_000, checkpoint: Optional[str] = None) -> None:
    """Train the combat bot using PPO, automatically resuming from the latest checkpoint if available."""
    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    tb_log_dir = os.path.join(base_dir, "tb_logs")
    checkpoint_dir = os.path.join(tb_log_dir, "checkpoints")
    metrics_dir = os.path.join(tb_log_dir, "metrics")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    resume_training = False
    checkpoint_path = None

    # If a checkpoint is provided via command-line and exists, use it
    if checkpoint and os.path.exists(checkpoint):
        checkpoint_path = checkpoint
        try:
            steps_completed = int(checkpoint.split("_")[-2])
        except Exception:
            steps_completed = 0
        logger.warning(f"Resuming training from provided checkpoint: {checkpoint_path} ({steps_completed:,} steps completed)")
        resume_training = True
    else:
        # Otherwise, search the checkpoint directory for any existing checkpoints
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
        if checkpoints:
            # Sort by step number to get the latest checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[-2]))[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            try:
                steps_completed = int(latest_checkpoint.split("_")[-2])
            except Exception:
                steps_completed = 0
            logger.warning(f"Found checkpoint: {latest_checkpoint} ({steps_completed:,} steps completed)")
            resume_training = True

    # Create and wrap the environment with smaller buffer size
    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
        norm_obs_keys=[
            "player_position",
            "player_combat_stats",
            "player_health",
            "player_prayer",
            "player_run_energy",
            "skills",
            "npcs",
            "current_chunk",
            "visited_chunks_count",
            "nearby_areas",
            "exploration_score"
        ]
    )

    device = get_device()
    
    # Configure policy kwargs to reduce memory usage
    policy_kwargs = dict(
        net_arch=[64, 64],  # Smaller network architecture
        activation_fn=torch.nn.ReLU
    )
    
    # Initialize PPO with optimized parameters
    if resume_training and checkpoint_path:
        logger.warning("Resuming training from checkpoint...")
        model = PPO.load(
            checkpoint_path,
            env=env,
            device=device,
            tensorboard_log=metrics_dir,  # Use metrics directory
            learning_rate=0.0003,
            n_steps=512,
            batch_size=32,
            n_epochs=5,
            gamma=0.99,
            verbose=1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            gae_lambda=0.95
        )
        # Load environment normalization stats if they exist
        env_stats_path = os.path.join(checkpoint_dir, "vec_normalize.pkl")
        if os.path.exists(env_stats_path):
            env = VecNormalize.load(env_stats_path, env)
            env.training = True  # Continue updating running stats
            logger.warning("Loaded environment normalization stats")
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=0.0003,
            n_steps=512,  # Reduced from 2048
            batch_size=32,  # Reduced from 64
            n_epochs=5,    # Reduced from 10
            gamma=0.99,
            verbose=1,
            device=device,
            tensorboard_log=metrics_dir,  # Use metrics directory
            policy_kwargs=policy_kwargs,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,  # Disable stochastic dynamics estimation
            gae_lambda=0.95
        )

    # Configure torch for memory optimization
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Save checkpoints more frequently
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,  # Save every 1000 steps
        save_path=checkpoint_dir,
        name_prefix="combat_bot",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )

    # Create and wrap evaluation environment
    eval_vec_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
        norm_obs_keys=[
            "player_position",
            "player_combat_stats",
            "player_health",
            "player_prayer",
            "player_run_energy",
            "skills",
            "npcs",
            "current_chunk",
            "visited_chunks_count",
            "nearby_areas",
            "exploration_score"
        ]
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(tb_log_dir, "eval"),
        log_path=os.path.join(tb_log_dir, "eval"),
        eval_freq=10000,  # Increased from 5000
        deterministic=True,
        render=False,
        n_eval_episodes=3,  # Reduced from 5
        verbose=0
    )

    training_callback = TrainingCallback()

    # Enable garbage collection with more aggressive settings
    gc.enable()
    gc.set_threshold(100, 5, 5)  # More aggressive GC

    try:
        # Clear memory before training
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, training_callback],
            progress_bar=True,
            reset_num_timesteps=False,  # Don't reset timesteps when resuming training
            tb_log_name="training"  # Use fixed name for TensorBoard logs
        )
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted! Saving checkpoint...")
        model.save(os.path.join(checkpoint_dir, f"combat_bot_interrupted_{model.num_timesteps}_steps"))
        env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))
        logger.warning(f"Saved checkpoint at {model.num_timesteps:,} steps")
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        env.close()
        eval_env.close()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Save the final model and environment normalization stats
    model.save(os.path.join(tb_log_dir, "final_model"))
    env.save(os.path.join(tb_log_dir, "vec_normalize.pkl"))

def test_combat_bot(model_path: str, vec_normalize_path: str) -> None:
    """Test a trained combat bot."""
    logger.info(f"Testing model from {model_path}")
    vec_env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_normalize_path, vec_env)
    env.training = False
    env.norm_reward = False

    device = get_device()
    model = PPO.load(model_path, env=env, device=device)

    obs = env.reset()[0]  # Get first element of the tuple
    total_reward = 0.0
    episode_rewards: List[float] = []

    try:
        logger.info("Starting test episodes")
        while len(episode_rewards) < 10:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if done[0]:
                logger.info(f"Episode {len(episode_rewards) + 1} finished with reward: {total_reward}")
                episode_rewards.append(total_reward)
                total_reward = 0.0
                obs = env.reset()[0]
        logger.info("\nTest Results:")
        logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
        logger.info(f"Standard deviation: {np.std(episode_rewards):.2f}")
        logger.info(f"Min reward: {np.min(episode_rewards):.2f}")
        logger.info(f"Max reward: {np.max(episode_rewards):.2f}")
    finally:
        env.close()

# -----------------------------------------------------------------------------
# Environment Definitions
# -----------------------------------------------------------------------------
nest_asyncio.apply()

class RuneScapeEnv(gym.Env):
    """
    Gymnasium Environment for RuneScape focused on combat training.
    """
    def __init__(self, websocket_url: str = "ws://localhost:43595", task: str = "combat"):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)  # Only show errors
        for ws_logger in ["websockets", "websockets.client", "websockets.protocol"]:
            logging.getLogger(ws_logger).setLevel(logging.ERROR)

        self.base_action_cooldown = 0.5
        self.action_cooldown_variance = 0.2
        self.min_action_interval = 0.3
        self.last_action_time = 0.0
        self.consecutive_fast_actions = 0
        self.max_consecutive_fast_actions = 3
        self.actions_per_minute_limit = 100
        self.action_count_window: List[float] = []

        self.websocket_url = websocket_url
        self.task = task
        self.screenshot_shape = (120, 160, 3)
        schema_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(schema_dir, "command_schema.json")) as f:
            self.command_schema = json.load(f)
        with open(os.path.join(schema_dir, "state_schema.json")) as f:
            self.state_schema = json.load(f)

        self.current_state: Optional[Dict] = None
        self.last_combat_exp = 0
        self.visited_areas: Set[Tuple[int, int]] = set()
        self.interfaces_open = False
        self.path_obstructed = False
        self.last_action: Optional[Action] = None
        self.last_position = None
        self.consecutive_same_pos = 0
        self.last_command: Optional[str] = None
        self.command_time: float = 0.0
        self.min_command_interval = 0.1
        self.last_target_id = None

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.loop = asyncio.new_event_loop()
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()

        timeout = 30
        start_time = time.time()
        while not self.ws and time.time() - start_time < timeout:
            time.sleep(0.1)
        if not self.ws:
            self.logger.error(f"Failed to connect within {timeout} seconds")

        self.observation_space = gym.spaces.Dict({
            "screenshot": gym.spaces.Box(low=0, high=255, shape=self.screenshot_shape, dtype=np.uint8),
            "player_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "player_combat_stats": gym.spaces.Box(low=1, high=99, shape=(7,), dtype=np.int32),
            "player_health": gym.spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            "player_prayer": gym.spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            "player_run_energy": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "skills": gym.spaces.Box(low=1, high=99, shape=(23,), dtype=np.int32),
            "npcs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 6), dtype=np.float32),
            "in_combat": gym.spaces.Discrete(2),
            "interfaces_open": gym.spaces.Discrete(2),
            "path_obstructed": gym.spaces.Discrete(2),
            "current_chunk": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32),
            "visited_chunks_count": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "nearby_areas": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 5), dtype=np.float32),
            "exploration_score": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Discrete(len(Action))

    def _run_websocket_loop(self) -> None:
        """Run the websocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_client())

    async def _websocket_client(self) -> None:
        """Handle websocket connection and message processing."""
        while True:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.ws = websocket
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            if isinstance(data, dict) and data.get("type") == "screenshot":
                                if self.current_state is not None:
                                    self.current_state["screenshot"] = data.get("data")
                                continue
                            try:
                                validate(instance=data, schema=self.state_schema)
                                if self.current_state and "screenshot" in self.current_state:
                                    data["screenshot"] = self.current_state["screenshot"]
                                self.current_state = data
                                self.interfaces_open = data.get("interfacesOpen", False)
                                self.path_obstructed = data.get("pathObstructed", False)
                            except ValidationError as ve:
                                self.logger.error(f"State validation error: {ve}")
                        except websockets.ConnectionClosed:
                            break
                        except json.JSONDecodeError as je:
                            self.logger.error(f"JSON decode error: {je}")
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                self.ws = None
                await asyncio.sleep(5)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset internal state and wait for a valid game state."""
        self.last_combat_exp = 0
        self.visited_areas.clear()
        self.last_action = None
        self.last_action_time = 0.0
        self.last_position = None
        self.consecutive_same_pos = 0
        self.last_command = None
        self.command_time = 0.0
        self.last_target_id = None

        timeout = 10
        start_time = time.time()
        while not self.current_state and time.time() - start_time < timeout:
            time.sleep(0.1)
        if not self.current_state:
            self.logger.error("No state received during reset - OSRS connection may be down")
            return self._get_empty_observation(), {}
        player = self.current_state.get("player", {})
        health = player.get("health", {})
        self.logger.warning(f"Episode start - Health: {health.get('current', 0)}/{health.get('maximum', 1)}")
        return self._state_to_observation(self.current_state), {
            "episode_start": True,
            "health": health,
            "location": player.get("location", {}),
            "in_combat": player.get("inCombat", False)
        }

    async def _execute_command(self, command: Dict) -> None:
        """
        Asynchronously send a command via websocket with rate limiting and retry logic.
        """
        if self.ws is None:
            return
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            validate(instance=command, schema=self.command_schema)
            cmd_json = json.dumps(command)
            current_time = time.time()
            if self.last_command == cmd_json and (current_time - self.command_time) < self.base_action_cooldown:
                return
            wait_time = self.base_action_cooldown - (current_time - self.command_time)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_command = cmd_json
            self.command_time = current_time
            if not self.ws.open:
                return
            await self.ws.send(cmd_json)
            return
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(0.5 * retry_count)

    def _get_next_action_delay(self) -> float:
        """Calculate delay for the next action with human-like variance."""
        current_time = time.time()
        self.action_count_window = [t for t in self.action_count_window if current_time - t < 60]
        
        if len(self.action_count_window) >= self.actions_per_minute_limit:
            oldest_action = self.action_count_window[0]
            extra_delay = max(0, 60 - (current_time - oldest_action))
            return max(self.base_action_cooldown + extra_delay, self.min_action_interval)
        
        delay = self.base_action_cooldown + random.uniform(-self.action_cooldown_variance, self.action_cooldown_variance)
        
        if random.random() < 0.2:
            delay += random.uniform(0.5, 1.5)
        
        delay = max(delay, self.min_action_interval)
        
        if self.consecutive_fast_actions >= self.max_consecutive_fast_actions:
            delay = max(delay, self.base_action_cooldown * 2)
            self.consecutive_fast_actions = 0
        
        return delay

    def step(self, action: Action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        current_time = time.time()
        
        next_delay = self._get_next_action_delay()
        if self.last_action_time:
            time_since_last_action = current_time - self.last_action_time
            if time_since_last_action < next_delay:
                time.sleep(next_delay - time_since_last_action)
        
        self.action_count_window.append(time.time())
        
        command = self._action_to_command(action)
        if command is not None and self.loop and not self.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(self._execute_command(command), self.loop)
            future.result(timeout=2.0)
            self.last_action = action
            self.last_action_time = time.time()
            time.sleep(0.1)
        
        observation = self._state_to_observation(self.current_state)
        reward = self._calculate_reward(self.current_state)
        done = self._is_episode_done(self.current_state)
        info = {
            "action": action.name if isinstance(action, Action) else Action(action).name,
            "state": self.current_state,
            "command_sent": command is not None,
            "action_delay": time.time() - self.last_action_time if self.last_action_time else 0
        }
        return observation, reward, done, False, info

    def render(self, mode: str = "human") -> None:
        """Rendering is handled externally."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        if self.ws is not None and self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        self.logger.info("Environment closed")

    def _get_empty_observation(self) -> Dict[str, np.ndarray]:
        """Return an empty observation."""
        return {
            "screenshot": np.zeros(self.screenshot_shape, dtype=np.uint8),
            "player_position": np.zeros(3, dtype=np.float32),
            "player_combat_stats": np.ones(7, dtype=np.int32),
            "player_health": np.ones(1, dtype=np.int32),
            "player_prayer": np.zeros(1, dtype=np.int32),
            "player_run_energy": np.zeros(1, dtype=np.float32),
            "skills": np.ones(23, dtype=np.int32),
            "npcs": np.zeros((10, 6), dtype=np.float32),
            "in_combat": np.zeros(1, dtype=np.int32),
            "interfaces_open": np.zeros(1, dtype=np.int32),
            "path_obstructed": np.zeros(1, dtype=np.int32),
            "current_chunk": np.zeros(2, dtype=np.int32),
            "visited_chunks_count": np.zeros(1, dtype=np.int32),
            "nearby_areas": np.zeros((9, 5), dtype=np.float32),
            "exploration_score": np.zeros(1, dtype=np.float32)
        }

    def _process_screenshot(self, screenshot_base64: str) -> np.ndarray:
        """Convert a base64-encoded screenshot to a numpy array with memory optimization."""
        try:
            img_data = base64.b64decode(screenshot_base64)
            with Image.open(BytesIO(img_data)) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size != (self.screenshot_shape[1], self.screenshot_shape[0]):
                    img = img.resize((self.screenshot_shape[1], self.screenshot_shape[0]), Image.Resampling.LANCZOS)
                array = np.array(img, dtype=np.uint8)
                del img
                return array
        except Exception as e:
            self.logger.error(f"Error processing screenshot: {e}", exc_info=True)
            return np.zeros(self.screenshot_shape, dtype=np.uint8)

    def _state_to_observation(self, state: Optional[Dict]) -> Dict[str, np.ndarray]:
        """Convert the raw state from RuneLite into a gym observation."""
        if not state:
            self.logger.warning("No state available, returning empty observation")
            return self._get_empty_observation()

        def safe_get(d: Dict, *keys, default=0):
            for key in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(key, default)
            return d if d is not None else default

        screenshot = np.zeros(self.screenshot_shape, dtype=np.uint8)
        if state.get("screenshot"):
            screenshot = self._process_screenshot(state["screenshot"])
            if screenshot is None or screenshot.shape != self.screenshot_shape:
                self.logger.error(f"Invalid screenshot shape: {screenshot.shape if screenshot is not None else None}, expected {self.screenshot_shape}")
                screenshot = np.zeros(self.screenshot_shape, dtype=np.uint8)
        
        player = state.get("player", {})
        location = player.get("location", {})
        position = np.array([location.get("x", 0), location.get("y", 0), location.get("plane", 0)], dtype=np.float32)

        skills = player.get("skills", {})
        combat_stats = np.array([
            skills.get("ATTACK", {}).get("level", 1),
            skills.get("STRENGTH", {}).get("level", 1),
            skills.get("DEFENCE", {}).get("level", 1),
            skills.get("RANGED", {}).get("level", 1),
            skills.get("MAGIC", {}).get("level", 1),
            skills.get("HITPOINTS", {}).get("level", 1),
            skills.get("PRAYER", {}).get("level", 1)
        ], dtype=np.int32)

        health = player.get("health", {})
        health_array = np.array([health.get("current", 1)], dtype=np.int32)
        prayer = np.array([player.get("prayer", 0)], dtype=np.int32)
        run_energy = np.array([player.get("runEnergy", 0.0)], dtype=np.float32)

        npcs = state.get("npcs", [])
        npcs.sort(key=lambda x: x.get("distance", float("inf")))
        npc_features = np.zeros((10, 6), dtype=np.float32)
        for i, npc in enumerate(npcs[:10]):
            health_current = npc.get("health", {}).get("current", 0)
            health_max = max(npc.get("health", {}).get("maximum", 1), 1)
            health_ratio = health_current / health_max
            npc_features[i] = np.array([
                npc.get("id", 0),
                npc.get("level", 0),
                npc.get("distance", 0),
                float(npc.get("interacting", False)),
                health_ratio,
                1.0
            ], dtype=np.float32)

        skills_array = np.ones(23, dtype=np.int32)
        for i, skill in enumerate(skills.values()):
            if i < 23:
                skills_array[i] = skill.get("level", 1)

        return {
            "screenshot": screenshot,
            "player_position": position,
            "player_combat_stats": combat_stats,
            "player_health": health_array,
            "player_prayer": prayer,
            "player_run_energy": run_energy,
            "skills": skills_array,
            "npcs": npc_features,
            "in_combat": np.array([player.get("inCombat", False)], dtype=np.int32),
            "interfaces_open": np.array([1 if self.interfaces_open else 0], dtype=np.int32),
            "path_obstructed": np.array([1 if self.path_obstructed else 0], dtype=np.int32),
            "current_chunk": np.array([
                safe_get(state, "exploration", "currentChunk", "x", default=0),
                safe_get(state, "exploration", "currentChunk", "y", default=0)
            ], dtype=np.int32),
            "visited_chunks_count": np.array([safe_get(state, "exploration", "visitedChunks", default=0)], dtype=np.int32),
            "nearby_areas": np.zeros((9, 5), dtype=np.float32),
            "exploration_score": np.array([0.0], dtype=np.float32)
        }

    def _calculate_reward(self, state: Optional[Dict]) -> float:
        """
        Compute reward based on combat experience gains and health management.
        """
        if not state:
            return 0.0
        reward = 0.0
        player = state.get("player", {})
        skills = player.get("skills", {})
        combat_skills = ["ATTACK", "STRENGTH", "DEFENCE", "RANGED", "MAGIC", "HITPOINTS"]
        current_combat_exp = sum(skills.get(skill, {}).get("experience", 0) for skill in combat_skills)
        if current_combat_exp > self.last_combat_exp:
            exp_gain = current_combat_exp - self.last_combat_exp
            reward += exp_gain * 0.5
            if exp_gain > 1000:
                self.logger.warning(f"Major exp gain: {exp_gain}")
        self.last_combat_exp = current_combat_exp

        if player.get("inCombat", False):
            health = player.get("health", {})
            health_ratio = health.get("current", 0) / max(health.get("maximum", 1), 1)
            if health_ratio < 0.2:
                reward -= 5.0
                self.logger.error("Critical health during combat!")
        if self.consecutive_same_pos > 2 and not player.get("inCombat", False):
            reward -= 0.5
        return reward

    def _is_episode_done(self, state: Optional[Dict]) -> bool:
        """Determine if the episode is finished."""
        if not state:
            return False
        player = state.get("player", {})
        health = player.get("health", {})
        return health.get("current", 0) <= 0 or len(state.get("npcs", [])) == 0

    def _action_to_command(self, action: Action) -> Optional[Dict]:
        """Convert a high-level action into a command following the schema."""
        if not self.current_state:
            return None
        if isinstance(action, (int, np.integer)):
            action = Action(int(action))
        player = self.current_state.get("player", {})
        location = player.get("location", {})
        current_x = location.get("x", 0)
        current_y = location.get("y", 0)
        if action == Action.ATTACK:
            npcs = self.current_state.get("npcs", [])
            attackable_npcs = [
                npc for npc in npcs
                if npc.get("combatLevel", 0) > 0 and not npc.get("interacting", False) and npc.get("id") != self.last_target_id
            ]
            if not attackable_npcs:
                attackable_npcs = [
                    npc for npc in npcs
                    if npc.get("combatLevel", 0) > 0 and npc.get("id") != self.last_target_id
                ]
            if attackable_npcs:
                nearest_npc = min(attackable_npcs, key=lambda x: x.get("distance", float("inf")))
                self.last_target_id = nearest_npc.get("id")
                return {
                    "action": "moveAndClick",
                    "data": {
                        "targetType": "npc",
                        "action": "Attack",
                        "npcId": nearest_npc["id"]
                    }
                }
            else:
                action = random.choice([Action.MOVE_FORWARD, Action.MOVE_BACKWARD, Action.MOVE_LEFT, Action.MOVE_RIGHT])
        move_distance = 2
        if action == Action.MOVE_FORWARD:
            return {
                "action": "moveAndClick",
                "data": {"targetType": "coordinates", "action": "Move", "x": current_x, "y": current_y + move_distance}
            }
        elif action == Action.MOVE_BACKWARD:
            return {
                "action": "moveAndClick",
                "data": {"targetType": "coordinates", "action": "Move", "x": current_x, "y": current_y - move_distance}
            }
        elif action == Action.MOVE_LEFT:
            return {
                "action": "moveAndClick",
                "data": {"targetType": "coordinates", "action": "Move", "x": current_x - move_distance, "y": current_y}
            }
        elif action == Action.MOVE_RIGHT:
            return {
                "action": "moveAndClick",
                "data": {"targetType": "coordinates", "action": "Move", "x": current_x + move_distance, "y": current_y}
            }
        elif action == Action.ROTATE_LEFT:
            return {"action": "camera_rotate", "data": {"right": False}}
        elif action == Action.ROTATE_RIGHT:
            return {"action": "camera_rotate", "data": {"right": True}}
        elif action == Action.ZOOM_IN:
            return {"action": "camera_zoom", "data": {"in": True}}
        elif action == Action.ZOOM_OUT:
            return {"action": "camera_zoom", "data": {"in": False}}
        elif action == Action.DO_NOTHING:
            return None
        return None

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RuneScape AI Training Bot")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps to train for")
    args = parser.parse_args()

    console.print("RuneScape AI Training Bot")
    console.print("Make sure RuneLite is running with the RLBot plugin enabled")
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            console.print(f"[green]Resuming from checkpoint: {args.checkpoint}[/green]")
        else:
            console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
            console.print("Starting fresh training run...")
    
    console.print("Waiting for WebSocket connection...\n")
    train_combat_bot(total_timesteps=args.timesteps, checkpoint=args.checkpoint)