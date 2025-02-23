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
import math
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
import cv2
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
    level=logging.WARNING,  # Changed from ERROR to WARNING
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "rlbot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Reduce websocket logging
for ws_logger in ["websockets", "websockets.client", "websockets.protocol"]:
    logging.getLogger(ws_logger).setLevel(logging.CRITICAL)  # Only show critical errors

# -----------------------------------------------------------------------------
# Action Definitions
# -----------------------------------------------------------------------------
class Action(Enum):
    MOVE_N = 0
    MOVE_NE = 1
    MOVE_E = 2
    MOVE_SE = 3
    MOVE_S = 4
    MOVE_SW = 5
    MOVE_W = 6
    MOVE_NW = 7
    DO_NOTHING = 8
    ROTATE_LEFT = 9
    ROTATE_RIGHT = 10
    ZOOM_IN = 11
    ZOOM_OUT = 12
    ATTACK = 13
    INTERACT = 14  # For doors, gates, etc.

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
        self.log_interval = 600  # Increased from 300 to 600 seconds (10 minutes)
        self.writer: Optional[SummaryWriter] = None
        self.step_in_episode = 0
        self.action_counts = {action.name: 0 for action in Action}
        self.last_screenshot_log = 0
        self.screenshot_log_interval = 5000  # Increased from 2000 to 5000 steps
        self.last_action_log = 0
        self.action_log_interval = 2000  # Log actions every 2000 steps
        
        # Track player movement
        self.last_position = None
        self.total_distance_moved = 0.0
        self.movement_heatmap = {}
        self.last_heatmap_log = 0
        self.heatmap_log_interval = 5000  # Log heatmap every 5000 steps
        
        # Track combat stats
        self.total_damage_dealt = 0
        self.npcs_killed = 0
        self.deaths = 0
        self.combat_time = 0.0
        self.last_combat_state = False
        self.last_health = None
        self.damage_taken = 0
        
        # Track skill progression
        self.initial_skills = {}
        self.skill_milestones = set()

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

    def _log_player_state(self, info: Dict) -> None:
        """Log detailed player state information."""
        if not self.writer:
            return
            
        timestep = self.num_timesteps
        player = info.get("state", {}).get("player", {})
        if not player:
            return

        # Log position and movement
        location = player.get("location", {})
        current_pos = (location.get("x", 0), location.get("y", 0))
        if self.last_position:
            dx = current_pos[0] - self.last_position[0]
            dy = current_pos[1] - self.last_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            self.total_distance_moved += distance
            
            # Log movement metrics
            self._log_scalar("movement/distance_per_step", distance, timestep)
            self._log_scalar("movement/total_distance", self.total_distance_moved, timestep)
            
            # Update movement heatmap
            chunk_x = current_pos[0] // 8
            chunk_y = current_pos[1] // 8
            chunk_key = f"{chunk_x},{chunk_y}"
            self.movement_heatmap[chunk_key] = self.movement_heatmap.get(chunk_key, 0) + 1
            
            # Log heatmap periodically
            if timestep - self.last_heatmap_log >= self.heatmap_log_interval:
                for chunk, visits in self.movement_heatmap.items():
                    self._log_scalar(f"heatmap/chunk_{chunk}", visits, timestep)
                self.last_heatmap_log = timestep
        self.last_position = current_pos
        
        # Log combat stats
        health = player.get("health", {})
        current_health = health.get("current", 0)
        max_health = health.get("maximum", 1)
        health_ratio = current_health / max_health
        
        self._log_scalar("player/health_ratio", health_ratio, timestep)
        self._log_scalar("player/run_energy", player.get("runEnergy", 0.0), timestep)
        
        # Track damage taken
        if self.last_health is not None and current_health < self.last_health:
            damage = self.last_health - current_health
            self.damage_taken += damage
            self._log_scalar("combat/damage_taken", damage, timestep)
            self._log_scalar("combat/total_damage_taken", self.damage_taken, timestep)
        self.last_health = current_health
        
        # Track combat time and state
        in_combat = player.get("inCombat", False)
        if in_combat != self.last_combat_state:
            if in_combat:
                self._log_scalar("combat/engagements", 1, timestep)
            else:
                combat_duration = time.time() - self.combat_time
                self._log_scalar("combat/duration", combat_duration, timestep)
        if in_combat:
            self.combat_time = time.time()
        self.last_combat_state = in_combat
        
        # Log skill levels and experience
        skills = player.get("skills", {})
        for skill_name, skill_data in skills.items():
            level = skill_data.get("level", 1)
            exp = skill_data.get("experience", 0)
            
            # Initialize skill tracking
            if skill_name not in self.initial_skills:
                self.initial_skills[skill_name] = {"level": level, "exp": exp}
            
            # Log current levels and experience gains
            self._log_scalar(f"skills/{skill_name}/level", level, timestep)
            exp_gain = exp - self.initial_skills[skill_name]["exp"]
            self._log_scalar(f"skills/{skill_name}/exp_gain", exp_gain, timestep)
            
            # Log significant level milestones
            milestone_key = f"{skill_name}_{level}"
            if level > self.initial_skills[skill_name]["level"] and milestone_key not in self.skill_milestones:
                self.skill_milestones.add(milestone_key)
                self._log_scalar(f"skills/{skill_name}/level_ups", 1, timestep)
        
        # Log NPC information
        npcs = info.get("state", {}).get("npcs", [])
        if npcs:
            nearby_npcs = len(npcs)
            combat_npcs = len([npc for npc in npcs if npc.get("combatLevel", 0) > 0])
            avg_npc_level = sum(npc.get("combatLevel", 0) for npc in npcs) / len(npcs)
            
            self._log_scalar("npcs/nearby_count", nearby_npcs, timestep)
            self._log_scalar("npcs/combat_npcs", combat_npcs, timestep)
            self._log_scalar("npcs/average_level", avg_npc_level, timestep)

    def _on_step(self) -> bool:
        """Called after each environment step."""
        try:
            infos = self.locals.get("infos", [])
            info = infos[0] if infos else {}
            rewards = self.locals.get("rewards", [])
            reward = rewards[0] if rewards else 0.0
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

            # Log detailed player state
            self._log_player_state(info)

            # Log to TensorBoard less frequently
            if self.writer and timestep % 10 == 0:  # Changed from 1 to 10
                # Basic metrics
                self._log_scalar("rewards/step", reward, timestep)
                self._log_scalar("rewards/cumulative", self.current_reward, timestep)
                
                # Action frequencies logged less often
                if timestep % self.action_log_interval == 0:
                    for action_name, freq in self.action_counts.items():
                        self._log_scalar(f"actions/frequency/{action_name}", freq, timestep)

            # Log significant events
            if info.get("exp_gain", 0) > 10000:  # Increased from 5000
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
            logger.error(f"Error in callback: {e}")
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
        logger.warning("Attempting to resume training from checkpoint...")
        try:
            model = PPO.load(
                checkpoint_path,
                env=env,
                device=device,
                tensorboard_log=metrics_dir,
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
        except ValueError as e:
            if "Action spaces do not match" in str(e):
                logger.warning("Action space has changed - starting fresh training run")
                resume_training = False
            else:
                raise e
    
    if not resume_training:
        logger.warning("Starting fresh training run")
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
            reset_num_timesteps=not resume_training,  # Only reset if not resuming
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
        self.logger.setLevel(logging.WARNING)  # Changed from ERROR to WARNING
        for ws_logger in ["websockets", "websockets.client", "websockets.protocol"]:
            logging.getLogger(ws_logger).setLevel(logging.CRITICAL)  # Only show critical errors

        # Action rate limiting parameters
        self.actions_per_minute_limit = 100
        self.min_action_interval = 60.0 / self.actions_per_minute_limit  # 0.6 seconds between actions
        self.action_times: List[float] = []  # Track action timestamps
        self.last_action_time = 0.0
        
        # Action sequence timing
        self.action_sequence_count = 0
        self.max_sequence_length = random.randint(2, 4)  # Shorter sequences
        self.sequence_break_duration = (0.5, 1.0)  # Shorter breaks
        
        # Websocket and schema setup
        self.websocket_url = websocket_url
        self.task = task
        self.screenshot_shape = (120, 160, 3)
        schema_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(schema_dir, "command_schema.json")) as f:
            self.command_schema = json.load(f)
        with open(os.path.join(schema_dir, "state_schema.json")) as f:
            self.state_schema = json.load(f)

        # State tracking
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
        self.last_target_id = None
        self._last_health = None
        self._death_logged = False

        # Websocket setup
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
        validation_error_count = 0
        last_validation_error_time = 0
        
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
                                validation_error_count = 0  # Reset counter on success
                            except ValidationError as ve:
                                # Only log validation errors occasionally to reduce spam
                                current_time = time.time()
                                if current_time - last_validation_error_time > 60:  # One error log per minute max
                                    self.logger.warning(f"State validation error: {ve.message}")
                                    last_validation_error_time = current_time
                                validation_error_count += 1
                        except websockets.ConnectionClosed:
                            break
                        except json.JSONDecodeError:
                            pass  # Ignore JSON decode errors
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                self.ws = None
                await asyncio.sleep(5)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment state and action tracking."""
        # Reset action tracking
        self.action_times.clear()
        self.last_action_time = 0.0
        self.action_sequence_count = 0
        
        # Reset other state variables
        self.last_combat_exp = 0
        self.visited_areas.clear()
        self.last_action = None
        self.last_position = None
        self.consecutive_same_pos = 0
        self.last_command = None
        self.command_time = 0.0
        self.last_target_id = None
        self._death_logged = False

        # Wait for valid game state
        timeout = 10
        start_time = time.time()
        while not self.current_state and time.time() - start_time < timeout:
            time.sleep(0.1)
        if not self.current_state:
            self.logger.error("No state received during reset - OSRS connection may be down")
            return self._get_empty_observation(), {}
            
        player = self.current_state.get("player", {})
        health = player.get("health", {})
        current_health = health.get("current", 0)
        max_health = health.get("maximum", 1)
        self._last_health = current_health
        
        return self._state_to_observation(self.current_state), {
            "episode_start": True,
            "health": health,
            "location": player.get("location", {}),
            "in_combat": player.get("inCombat", False)
        }

    def _get_next_action_delay(self) -> float:
        """Calculate delay needed to maintain action rate limit."""
        current_time = time.time()
        
        # Clean up old action times (older than 1 minute)
        self.action_times = [t for t in self.action_times if current_time - t < 60.0]
        
        # If we've hit our limit, wait until the oldest action expires
        if len(self.action_times) >= self.actions_per_minute_limit:
            wait_time = self.action_times[0] + 60.0 - current_time
            if wait_time > 0:
                return wait_time
        
        # Calculate minimum time since last action
        time_since_last = current_time - self.last_action_time if self.last_action_time else float('inf')
        
        # Ensure minimum interval between actions
        if time_since_last < self.min_action_interval:
            return self.min_action_interval - time_since_last
        
        # Add variation to prevent exact timing patterns
        jitter = random.uniform(-0.1, 0.1)  # ±100ms variation
        return max(0.0, self.min_action_interval + jitter)

    async def _execute_command(self, command: Dict) -> None:
        """Execute command with rate limiting."""
        if self.ws is None:
            return

        try:
            # Validate command format
            validate(instance=command, schema=self.command_schema)
            
            # Send the command
            cmd_json = json.dumps(command)
            if not self.ws.open:
                return
                
            await self.ws.send(cmd_json)
            
            # Record command time
            current_time = time.time()
            self.action_times.append(current_time)
            self.last_action_time = current_time
            self.last_command = cmd_json
            
        except ValidationError as ve:
            self.logger.error(f"Command validation error: {ve}")
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")

    def step(self, action: Action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment with rate limiting."""
        # Reset death logged flag if health is above 0
        if self.current_state:
            player = self.current_state.get("player", {})
            health = player.get("health", {})
            current_health = health.get("current", 0)
            if current_health > 0:
                self._death_logged = False

        # Calculate and apply action delay
        next_delay = self._get_next_action_delay()
        if next_delay > 0:
            time.sleep(next_delay)
        
        # Execute the command
        command = self._action_to_command(action)
        if command is not None and self.loop and not self.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(self._execute_command(command), self.loop)
            try:
                future.result(timeout=1.0)
            except Exception as e:
                self.logger.error(f"Error executing command: {e}")
        
        # Get the next observation and calculate rewards
        observation = self._state_to_observation(self.current_state)
        reward = self._calculate_reward(self.current_state)
        done = self._is_episode_done(self.current_state)
        
        # Calculate current actions per minute
        current_time = time.time()
        actions_last_minute = len([t for t in self.action_times if current_time - t < 60.0])
        
        info = {
            "action": action.name if isinstance(action, Action) else Action(action).name,
            "state": self.current_state,
            "command_sent": command is not None,
            "action_delay": next_delay,
            "actions_per_minute": actions_last_minute,
            "sequence_position": self.action_sequence_count
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
            if not screenshot_base64:
                self.logger.warning("Empty screenshot data received")
                return np.zeros(self.screenshot_shape, dtype=np.uint8)

            # Ensure padding is correct for base64
            padding = 4 - (len(screenshot_base64) % 4)
            if padding != 4:
                screenshot_base64 += "=" * padding

            # Decode base64 data
            try:
                img_data = base64.b64decode(screenshot_base64)
            except Exception as e:
                self.logger.error(f"Base64 decoding error: {e}")
                return np.zeros(self.screenshot_shape, dtype=np.uint8)

            # Open and process image
            try:
                with Image.open(BytesIO(img_data)) as img:
                    if img.format not in ['JPEG', 'PNG']:
                        self.logger.warning(f"Unexpected image format: {img.format}")
                    
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    if img.size != (self.screenshot_shape[1], self.screenshot_shape[0]):
                        img = img.resize(
                            (self.screenshot_shape[1], self.screenshot_shape[0]),
                            Image.Resampling.LANCZOS
                        )
                    
                    # Convert to numpy array efficiently
                    array = np.array(img, dtype=np.uint8)
                    return array
            except Exception as e:
                self.logger.error(f"Image processing error: {e}")
                return np.zeros(self.screenshot_shape, dtype=np.uint8)

        except Exception as e:
            self.logger.error(f"Screenshot processing error: {e}")
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

        # Initialize screenshot with empty array
        screenshot = np.zeros(self.screenshot_shape, dtype=np.uint8)
        
        # Try to get screenshot from state
        try:
            if "screenshot" in state:
                screenshot_data = state["screenshot"]
                if isinstance(screenshot_data, str) and screenshot_data:
                    try:
                        screenshot = self._process_screenshot(screenshot_data)
                    except Exception as e:
                        self.logger.error(f"Error processing screenshot data: {e}", exc_info=True)
                elif isinstance(screenshot_data, dict) and "data" in screenshot_data:
                    try:
                        screenshot = self._process_screenshot(screenshot_data["data"])
                    except Exception as e:
                        self.logger.error(f"Error processing screenshot from data field: {e}", exc_info=True)
                else:
                    self.logger.debug(f"Invalid screenshot data type: {type(screenshot_data)}")
            else:
                self.logger.debug("No screenshot in state")
        except Exception as e:
            self.logger.error(f"Error handling screenshot: {e}", exc_info=True)

        player = state.get("player", {})
        health = player.get("health", {})
        current_health = health.get("current", 1)
        max_health = health.get("maximum", 1)

        # Log health changes, especially after death
        if hasattr(self, '_last_health') and self._last_health != current_health:
            self.logger.warning(f"Health changed from {self._last_health} to {current_health} (max: {max_health})")
            if self._last_health is not None and self._last_health <= 0 and current_health > 0:
                self.logger.warning("Player has respawned!")
        self._last_health = current_health

        health_array = np.array([current_health], dtype=np.int32)
        prayer = np.array([player.get("prayer", 0)], dtype=np.int32)
        run_energy = np.array([player.get("runEnergy", 0.0)], dtype=np.float32)

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
            "player_position": np.array([player.get("location", {}).get("x", 0), player.get("location", {}).get("y", 0), player.get("location", {}).get("plane", 0)], dtype=np.float32),
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
        Compute reward based on combat experience gains, health management, combat engagement,
        movement efficiency, and resource management.
        """
        if not state:
            return 0.0
        reward = 0.0
        player = state.get("player", {})
        skills = player.get("skills", {})

        # Combat experience rewards
        combat_skills = ["ATTACK", "STRENGTH", "DEFENCE", "RANGED", "MAGIC", "HITPOINTS"]
        current_combat_exp = sum(skills.get(skill, {}).get("experience", 0) for skill in combat_skills)
        if current_combat_exp > self.last_combat_exp:
            exp_gain = current_combat_exp - self.last_combat_exp
            # Scale exp rewards logarithmically to prevent exploitation
            reward += np.log1p(exp_gain) * 0.5
            if exp_gain > 1000:
                self.logger.warning(f"Major exp gain: {exp_gain}")
        self.last_combat_exp = current_combat_exp

        # Health management rewards/penalties
        health = player.get("health", {})
        current_health = health.get("current", 0)
        max_health = health.get("maximum", 1)
        health_ratio = current_health / max_health

        # Progressive health penalties
        if health_ratio < 0.7:  # Start penalties earlier
            # Exponential penalty scaling
            penalty = np.exp((0.7 - health_ratio) * 3) - 1  # Exponential scaling
            reward -= penalty
            if health_ratio < 0.3:
                self.logger.error("Critical health. Health is %i, health ratio is %f", current_health, health_ratio)
        elif health_ratio > 0.8:  # Reward for maintaining high health
            reward += 0.2

        # Death penalty
        if current_health <= 0:
            reward -= 100.0  # Increased death penalty
            self.logger.error("Death occurred!")

        # Combat engagement rewards
        in_combat = player.get("inCombat", False)
        if in_combat:
            reward += 0.3  # Base combat engagement reward
            # Additional reward for fighting appropriate level NPCs
            npcs = state.get("npcs", [])
            if npcs:
                player_combat_level = self._calculate_combat_level(skills)
                for npc in npcs:
                    if npc.get("interacting", False):  # If NPC is interacting with player
                        npc_level = npc.get("combatLevel", 0)
                        level_difference = abs(player_combat_level - npc_level)
                        if level_difference <= 10:  # Reward fighting appropriate level NPCs
                            reward += 0.2
                        elif level_difference > 20:  # Penalize fighting very weak/strong NPCs
                            reward -= 0.1

        # Movement and position rewards/penalties
        current_pos = (
            player.get("location", {}).get("x", 0),
            player.get("location", {}).get("y", 0)
        )
        
        # Penalize staying still when not in combat
        if self.consecutive_same_pos > 2 and not in_combat:
            reward -= 0.5 * (self.consecutive_same_pos - 2)  # Progressive penalty
        
        # Update position tracking
        if hasattr(self, 'last_position') and self.last_position:
            if current_pos == self.last_position:
                self.consecutive_same_pos += 1
            else:
                # Reward for exploring new areas
                if current_pos not in getattr(self, 'visited_positions', set()):
                    reward += 0.1
                    self.visited_positions = getattr(self, 'visited_positions', set()) | {current_pos}
                self.consecutive_same_pos = 0
        self.last_position = current_pos

        # Prayer and run energy management
        prayer_points = player.get("prayer", 0)
        run_energy = player.get("runEnergy", 0.0)
        
        # Penalize completely depleted resources
        if prayer_points == 0:
            reward -= 0.2
        if run_energy < 5.0:
            reward -= 0.2

        # Clip final reward to prevent extreme values
        return np.clip(reward, -100.0, 100.0)

    def _calculate_combat_level(self, skills: Dict) -> int:
        """Calculate the player's combat level based on their skills."""
        attack = skills.get("ATTACK", {}).get("level", 1)
        strength = skills.get("STRENGTH", {}).get("level", 1)
        defence = skills.get("DEFENCE", {}).get("level", 1)
        hitpoints = skills.get("HITPOINTS", {}).get("level", 10)
        prayer = skills.get("PRAYER", {}).get("level", 1)
        ranged = skills.get("RANGED", {}).get("level", 1)
        magic = skills.get("MAGIC", {}).get("level", 1)

        # Base combat calculation
        base = 0.25 * (defence + hitpoints + math.floor(prayer/2))
        melee = 0.325 * (attack + strength)
        range_level = 0.325 * (math.floor(3 * ranged/2))
        magic_level = 0.325 * (math.floor(3 * magic/2))
        
        # Use highest combat style
        combat = base + max(melee, range_level, magic_level)
        return math.floor(combat)

    def _is_episode_done(self, state: Optional[Dict]) -> bool:
        """Determine if the episode is finished."""
        if not state:
            return False
        player = state.get("player", {})
        health = player.get("health", {})
        current_health = health.get("current", 0)
        
        # Check for respawn
        if hasattr(self, '_last_health') and self._last_health is not None:
            if self._last_health <= 0 and current_health > 0:
                self.logger.warning("Player has respawned!")
                return True  # End episode on respawn
        
        # Update last health
        self._last_health = current_health
        
        # Log death events
        if current_health <= 0 and not getattr(self, '_death_logged', False):
            self.logger.warning("Episode ending due to death (health: 0)")
            self._death_logged = True
            
        # Log if no NPCs remain
        if len(state.get("npcs", [])) == 0:
            self.logger.warning("Episode ending due to no NPCs remaining")
            
        return current_health <= 0 or len(state.get("npcs", [])) == 0

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
        
        # Base movement distance varies by direction (diagonal vs cardinal)
        base_distance = random.randint(3, 5)  # Randomize movement distance slightly
        diagonal_distance = int(base_distance * 0.707)  # cos(45°) ≈ 0.707
        
        # Check for nearby objects that might need interaction
        nearby_objects = self._get_nearby_interactable_objects()
        
        if action == Action.ATTACK:
            npcs = self.current_state.get("npcs", [])
            # Filter for attackable NPCs, considering level difference
            player_combat_level = self._calculate_combat_level(player.get("skills", {}))
            attackable_npcs = [
                npc for npc in npcs
                if (npc.get("combatLevel", 0) > 0 and 
                    abs(npc.get("combatLevel", 0) - player_combat_level) < 20 and  # Level difference check
                    not npc.get("interacting", False) and 
                    npc.get("id") != self.last_target_id)
            ]
            
            if not attackable_npcs:
                # Fallback to any non-interacting NPCs if none match our criteria
                attackable_npcs = [
                    npc for npc in npcs
                    if npc.get("combatLevel", 0) > 0 and not npc.get("interacting", False)
                ]
            
            if attackable_npcs:
                # Sort by combination of distance and level appropriateness
                nearest_npc = min(attackable_npcs, 
                    key=lambda x: (x.get("distance", float("inf")) * 
                                 (1 + abs(x.get("combatLevel", 0) - player_combat_level) / 20)))
                self.last_target_id = nearest_npc.get("id")
                return {
                    "action": "moveAndClick",
                    "data": {
                        "targetType": "npc",
                        "action": "Attack",
                        "npcId": nearest_npc["id"]
                    }
                }
        
        elif action == Action.INTERACT and nearby_objects:
            obj = nearby_objects[0]  # Take closest interactable object
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "object",
                    "objectId": obj["id"],
                    "action": obj["action"],
                    "x": obj["x"],
                    "y": obj["y"]
                }
            }
        
        # Movement commands with collision avoidance
        movement_commands = {
            Action.MOVE_N:  (current_x, current_y + base_distance),
            Action.MOVE_NE: (current_x + diagonal_distance, current_y + diagonal_distance),
            Action.MOVE_E:  (current_x + base_distance, current_y),
            Action.MOVE_SE: (current_x + diagonal_distance, current_y - diagonal_distance),
            Action.MOVE_S:  (current_x, current_y - base_distance),
            Action.MOVE_SW: (current_x - diagonal_distance, current_y - diagonal_distance),
            Action.MOVE_W:  (current_x - base_distance, current_y),
            Action.MOVE_NW: (current_x - diagonal_distance, current_y + diagonal_distance)
        }
        
        if action in movement_commands:
            target_x, target_y = movement_commands[action]
            
            # Check if path is blocked
            if self._is_path_blocked(current_x, current_y, target_x, target_y):
                # Try to find alternative path or interact with blocking object
                alt_path = self._find_alternative_path(current_x, current_y, target_x, target_y)
                if alt_path:
                    target_x, target_y = alt_path
                elif nearby_objects:  # If blocked by interactable object, interact with it
                    obj = nearby_objects[0]
                    return {
                        "action": "moveAndClick",
                        "data": {
                            "targetType": "object",
                            "objectId": obj["id"],
                            "action": obj["action"],
                            "x": obj["x"],
                            "y": obj["y"]
                        }
                    }
            
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "coordinates",
                    "action": "Move",
                    "x": target_x,
                    "y": target_y
                }
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

    def _get_nearby_interactable_objects(self) -> List[Dict]:
        """Find nearby objects that can be interacted with (doors, gates, etc.)."""
        if not self.current_state:
            return []
        
        objects = self.current_state.get("objects", [])
        player_loc = self.current_state.get("player", {}).get("location", {})
        current_x = player_loc.get("x", 0)
        current_y = player_loc.get("y", 0)
        
        interactable_objects = []
        for obj in objects:
            obj_x = obj.get("location", {}).get("x", 0)
            obj_y = obj.get("location", {}).get("y", 0)
            
            # Calculate distance to object
            distance = math.sqrt((obj_x - current_x)**2 + (obj_y - current_y)**2)
            
            # Check if object has relevant actions
            actions = obj.get("actions", [])
            interactable_actions = ["Open", "Close", "Enter", "Exit", "Climb", "Use"]
            
            for action in actions:
                if action in interactable_actions and distance < 10:  # Within reasonable distance
                    interactable_objects.append({
                        "id": obj.get("id"),
                        "action": action,
                        "x": obj_x,
                        "y": obj_y,
                        "distance": distance
                    })
        
        # Sort by distance
        interactable_objects.sort(key=lambda x: x["distance"])
        return interactable_objects

    def _is_path_blocked(self, start_x: int, start_y: int, target_x: int, target_y: int) -> bool:
        """Check if there are obstacles between start and target positions."""
        if not self.current_state:
            return False
        
        # Get collision data from state
        collision_data = self.current_state.get("collisionData", {})
        if not collision_data:
            return False
        
        # Simple line-of-sight check
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1:
            return False
        
        # Check points along the path
        steps = int(distance)
        for i in range(steps):
            check_x = int(start_x + (dx * i / steps))
            check_y = int(start_y + (dy * i / steps))
            
            # Check if point is blocked in collision map
            if collision_data.get(f"{check_x},{check_y}", False):
                return True
        
        return False

    def _find_alternative_path(self, start_x: int, start_y: int, target_x: int, target_y: int) -> Optional[Tuple[int, int]]:
        """Find an alternative path when direct route is blocked."""
        # Simple implementation: try slight variations of the target position
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in offsets:
            new_x = target_x + dx
            new_y = target_y + dy
            if not self._is_path_blocked(start_x, start_y, new_x, new_y):
                return (new_x, new_y)
        
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