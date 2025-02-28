#!/usr/bin/env python3
"""
RuneScape AI Training Bot using PPO with Stable Baselines3.

This module defines a custom Gymnasium environment for RuneScape and associated training/testing
utilities for training a combat bot. It leverages PPO from Stable Baselines3 and logs custom metrics
to TensorBoard.
"""

import gymnasium as gym

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
from dotenv import load_dotenv
load_dotenv()

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
from openai import OpenAI

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
class WidgetID:
    """Widget group IDs from RuneLite's WidgetID.java"""
    WORLD_MAP_GROUP_ID = 595
    INVENTORY_GROUP_ID = 149
    SKILLS_GROUP_ID = 320
    EQUIPMENT_GROUP_ID = 387
    PRAYER_GROUP_ID = 541
    SPELLBOOK_GROUP_ID = 218
    BANK_GROUP_ID = 12
    DIALOG_GROUP_ID = 231
    CHATBOX_GROUP_ID = 162

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

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
    ATTACK = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    INTERFACE_ACTION = 5
    
    def __int__(self):
        return self.value

# -----------------------------------------------------------------------------
# Training Callback for Custom TensorBoard Logging
# -----------------------------------------------------------------------------
class TrainingCallback(BaseCallback):
    """
    Custom callback for logging metrics to TensorBoard during training.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        # Initialize all tracking variables
        self.episode_count = 0
        self.current_reward = 0.0
        self.rewards: List[float] = []
        self.last_log_time = time.time()
        self.log_interval = 600  # 10 minutes
        self.writer: Optional[SummaryWriter] = None
        self.step_in_episode = 0
        
        # Action tracking with moving windows
        self.action_counts: Dict[str, int] = {action.name: 0 for action in Action}
        self.action_window: List[Action] = []  # Track recent actions
        self.window_size = 100  # Size of moving window
        self.last_action_log = 0
        self.action_log_interval = 100
        
        # Movement tracking
        self.last_position: Optional[Tuple[float, float]] = None
        self.total_distance_moved = 0.0
        self.last_movement_log = 0
        self.movement_log_interval = 100
        self.position_history: List[Tuple[float, float]] = []  # Track position history
        self.stuck_threshold = 10  # Number of steps to consider "stuck"
        
        # Combat tracking
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.combat_engagements = 0
        self.combat_duration = 0.0
        self.last_combat_state = False
        self.combat_start_time: Optional[float] = None
        self.last_health = None
        self.combat_success_rate = 0.0
        self.successful_combats = 0
        
        # NPC tracking
        self.last_npc_log = 0
        self.npc_log_interval = 100
        self.npc_interaction_history: List[int] = []  # Track NPC IDs interacted with
        
        # Player state tracking
        self.last_player_log = 0
        self.player_log_interval = 100
        self.death_count = 0
        self.last_death_time = 0
        self.avg_survival_time = 0.0
        
        # Skills tracking
        self.initial_skills: Dict[str, Dict[str, int]] = {}
        self.skill_names = [
            "Agility", "Attack", "Construction", "Cooking", "Crafting", "Defence",
            "Farming", "Firemaking", "Fishing", "Fletching", "Herblore", "Hitpoints",
            "Hunter", "Magic", "Mining", "Prayer", "Ranged", "Runecraft", "Slayer",
            "Smithing", "Strength", "Thieving", "Woodcutting"
        ]
        self.exp_gain_rate: Dict[str, float] = {skill: 0.0 for skill in self.skill_names}
        
        # Training metrics
        self.approx_kl = 0.0
        self.clip_fraction = 0.0
        self.clip_range = 0.2
        self.entropy_loss = 0.0
        self.explained_variance = 0.0
        self.learning_rate = 0.0003
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.fps = 0
        
        # Performance metrics
        self.action_success_rate = 0.0
        self.failed_actions = 0
        self.successful_actions = 0
        self.exploration_efficiency = 0.0
        self.unique_areas_visited: Set[Tuple[int, int]] = set()
        self.visited_positions: Set[Tuple[float, float]] = set()
        
        # Rollout tracking
        self.episode_lengths: List[int] = []
        self.episode_rewards: List[float] = []
        self.reward_history: List[float] = []  # Track reward history
        self.avg_reward_window = 100  # Window size for moving average
        
        # Initialize the callback
        self._init_callback()

    def _init_callback(self) -> None:
        """Initialize the TensorBoard writer."""
        log_dir = os.path.join(get_tensorboard_dir(), "metrics")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, comment="")
        if self.verbose > 0:
            print(f"Initialized TensorBoard writer in {log_dir}")

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        """Helper method to log a scalar value to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def _update_action_metrics(self, action: Action, success: bool) -> None:
        """Update action-related metrics."""
        if success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1
        total_actions = self.successful_actions + self.failed_actions
        self.action_success_rate = self.successful_actions / total_actions if total_actions > 0 else 0.0
        
        # Update action window
        self.action_window.append(action)
        if len(self.action_window) > self.window_size:
            self.action_window.pop(0)

    def _update_movement_metrics(self, current_pos: Tuple[float, float]) -> None:
        """Update movement-related metrics."""
        self.position_history.append(current_pos)
        if len(self.position_history) > self.stuck_threshold:
            self.position_history.pop(0)
            
        # Calculate if stuck
        if len(self.position_history) >= self.stuck_threshold:
            recent_positions = self.position_history[-self.stuck_threshold:]
            unique_positions = set(recent_positions)
            if len(unique_positions) == 1:  # All positions are the same
                self._log_scalar("movement/stuck_duration", self.stuck_threshold, self.num_timesteps)

    def _update_combat_metrics(self, in_combat: bool, current_health: int, max_health: int) -> None:
        """Update combat-related metrics."""
        if in_combat != self.last_combat_state:
            if in_combat:
                self.combat_engagements += 1
                self.combat_start_time = time.time()
            elif self.combat_start_time is not None:
                combat_duration = time.time() - self.combat_start_time
                self.combat_duration += combat_duration
                # Consider combat successful if health is above 50%
                if current_health > max_health * 0.5:
                    self.successful_combats += 1
                self.combat_success_rate = self.successful_combats / self.combat_engagements

    def _on_step(self) -> bool:
        infos = self.locals["infos"] if "infos" in self.locals else []
        info = infos[0] if infos else {}
        rewards = self.locals["rewards"] if "rewards" in self.locals else []
        reward = float(rewards[0]) if rewards else 0.0
        actions = self.locals["actions"] if "actions" in self.locals else None
        action = actions[0] if isinstance(actions, (list, np.ndarray)) else None
        
        timestep = self.num_timesteps
        current_time = time.time()

        # Update basic metrics
        self.current_reward += reward
        self.step_in_episode += 1
        
        if self.writer:
            # Get state information first
            state = info["state"] if "state" in info else {}
            player = state["player"] if "player" in state else {}
            
            # Log screenshot if available
            if "screenshot" in state:
                screenshot_data = state["screenshot"]
                if isinstance(screenshot_data, str) and screenshot_data:
                    img_data = base64.b64decode(screenshot_data)
                    with Image.open(BytesIO(img_data)) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        screenshot = np.array(img).transpose(2, 0, 1)  # Convert to CHW format
                        screenshot = np.expand_dims(screenshot, 0)  # Add batch dimension
                        self.writer.add_images('environment/screenshot', screenshot, timestep)

            # Log rewards with moving average
            self.reward_history.append(reward)
            if len(self.reward_history) > self.avg_reward_window:
                self.reward_history.pop(0)
            avg_reward = float(np.mean(self.reward_history)) if self.reward_history else 0.0
            self._log_scalar("rewards/step", reward, timestep)
            self._log_scalar("rewards/moving_average", avg_reward, timestep)
            self._log_scalar("rewards/cumulative", self.current_reward, timestep)
            
            # Action analysis
            if action is not None:
                action_enum = Action(int(action)) if isinstance(action, (int, np.integer)) else action
                self.action_counts[action_enum.name] = self.action_counts[action_enum.name] + 1 if action_enum.name in self.action_counts else 1
                
                # Log action distributions and patterns every step
                total_actions = sum(self.action_counts.values())
                for name, count in self.action_counts.items():
                    self._log_scalar(f"actions/frequency/{name}", count, timestep)
                    self._log_scalar(f"actions/distribution/{name}", 
                                  float(count/total_actions if total_actions > 0 else 0), 
                                  timestep)
                
                # Log action success metrics
                self._log_scalar("actions/success_rate", self.action_success_rate, timestep)
                self._log_scalar("actions/failed_actions", self.failed_actions, timestep)
            
            # Player state and combat metrics - log every step
            if player:
                health = player["health"] if "health" in player else {}
                current_health = health["current"] if "current" in health else 0
                max_health = health["maximum"] if "maximum" in health else 1
                health_ratio = float(current_health / max_health if max_health > 0 else 0)
                
                # Health and combat metrics
                self._log_scalar("player/health_ratio", health_ratio, timestep)
                self._log_scalar("player/run_energy", float(player["runEnergy"] if "runEnergy" in player else 0.0), timestep)
                self._log_scalar("combat/in_combat", int(player["inCombat"] if "inCombat" in player else False), timestep)
                self._log_scalar("combat/success_rate", self.combat_success_rate, timestep)
                self._log_scalar("combat/total_engagements", self.combat_engagements, timestep)
                
                # Track damage and update combat metrics
                if self.last_health is not None and current_health < self.last_health:
                    damage = self.last_health - current_health
                    self.total_damage_taken += damage
                    self._log_scalar("combat/damage_taken", float(damage), timestep)
                    self._log_scalar("combat/total_damage_taken", float(self.total_damage_taken), timestep)
                self.last_health = current_health
                
                # Update combat state
                in_combat = player["inCombat"] if "inCombat" in player else False
                self._update_combat_metrics(in_combat, current_health, max_health)
                self.last_combat_state = in_combat
            
            # Movement and exploration metrics - log every step
            location = player["location"] if "location" in player else {}
            current_pos = (location["x"] if "x" in location else 0, location["y"] if "y" in location else 0)
            if current_pos != self.last_position:
                if self.last_position:
                    dx = current_pos[0] - self.last_position[0]
                    dy = current_pos[1] - self.last_position[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    self.total_distance_moved += distance
                    
                    # Log movement metrics
                    self._log_scalar("movement/step_distance", distance, timestep)
                    self._log_scalar("movement/total_distance", self.total_distance_moved, timestep)
                    
                    # Track unique areas
                    chunk_x, chunk_y = current_pos[0] // 8, current_pos[1] // 8  # Convert to chunk coordinates
                    self.unique_areas_visited.add((chunk_x, chunk_y))
                    self._log_scalar("exploration/unique_areas", len(self.unique_areas_visited), timestep)
                
                self._update_movement_metrics(current_pos)
            self.last_position = current_pos
            
            # NPC interaction metrics - log every step
            npcs = state["npcs"] if "npcs" in state else []
            if npcs:
                interacting_npcs = [npc for npc in npcs if npc["interacting"] if "interacting" in npc]
                self._log_scalar("npcs/total_nearby", len(npcs), timestep)
                self._log_scalar("npcs/interacting", len(interacting_npcs), timestep)
                
                # Track combat-level appropriate NPCs
                player_combat_level = self._calculate_combat_level(player["skills"] if "skills" in player else {})
                appropriate_level_npcs = [
                    npc for npc in npcs
                    if abs(npc["combatLevel"] if "combatLevel" in npc else 0 - player_combat_level) < 20
                ]
                self._log_scalar("npcs/appropriate_level", len(appropriate_level_npcs), timestep)
            
            # Skill progression metrics - log every step
            skills = player["skills"] if "skills" in player else {}
            for skill_name in self.skill_names:
                if skill_name in skills:
                    skill_data = skills[skill_name]
                    level = skill_data["level"] if "level" in skill_data else 1
                    exp = skill_data["experience"] if "experience" in skill_data else 0
                    
                    if skill_name not in self.initial_skills:
                        self.initial_skills[skill_name] = {"level": level, "exp": exp}
                    
                    # Calculate and log experience rates
                    exp_gain = exp - self.initial_skills[skill_name]["exp"]
                    time_elapsed = (current_time - self.last_log_time) / 3600  # Convert to hours
                    if time_elapsed > 0:
                        exp_rate = exp_gain / time_elapsed
                        self.exp_gain_rate[skill_name] = exp_rate
                        self._log_scalar(f"skills/{skill_name}/exp_rate", exp_rate, timestep)
                    
                    self._log_scalar(f"skills/{skill_name}/level", level, timestep)
                    self._log_scalar(f"skills/{skill_name}/total_exp_gain", exp_gain, timestep)
            
            # Training performance metrics
            if hasattr(self.locals, "approx_kl"):
                self._log_scalar("train/approx_kl", self.locals["approx_kl"], timestep)
                self._log_scalar("train/clip_fraction", self.locals["clip_fraction"], timestep)
                self._log_scalar("train/entropy_loss", self.locals["entropy_loss"], timestep)
                self._log_scalar("train/explained_variance", self.locals["explained_variance"], timestep)
                self._log_scalar("train/learning_rate", self.learning_rate, timestep)
                self._log_scalar("train/policy_loss", self.locals["policy_gradient_loss"], timestep)
                self._log_scalar("train/value_loss", self.locals["value_loss"], timestep)
            
            # Episode statistics
            if len(self.episode_lengths) > 0:
                self._log_scalar("episode/avg_length", float(np.mean(self.episode_lengths[-100:])), timestep)
            if len(self.episode_rewards) > 0:
                self._log_scalar("episode/avg_reward", float(np.mean(self.episode_rewards[-100:])), timestep)
            
            # Performance metrics
            self._log_scalar("performance/action_success_rate", self.action_success_rate, timestep)
            self._log_scalar("performance/exploration_efficiency", 
                           len(self.unique_areas_visited) / (self.step_in_episode + 1), timestep)
            
            # Flush writer to ensure all metrics are written
            self.writer.flush()

        return True

    def _on_rollout_start(self) -> None:
        """Called when collecting new experiences starts."""
        self.episode_count += 1
        self.step_in_episode = 0
        self.episode_lengths.append(self.step_in_episode)
        self.episode_rewards.append(self.current_reward)

    def _on_rollout_end(self) -> None:
        """Called when collecting new experiences ends."""
        if self.writer:
            # Log episode summary statistics
            self.writer.add_scalar("episode/total_reward", self.current_reward, self.episode_count)
            self.writer.add_scalar("episode/length", self.step_in_episode, self.episode_count)
            self.writer.add_scalar("episode/combat_success_rate", self.combat_success_rate, self.episode_count)
            self.writer.add_scalar("episode/exploration_coverage", 
                                 len(self.unique_areas_visited), self.episode_count)
            self.writer.flush()
        
        # Reset episode-specific metrics
        self.rewards.append(self.current_reward)
        self.current_reward = 0.0
        self.action_counts = {action.name: 0 for action in Action}
        
        # Memory management
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.writer:
            # Log final summary statistics
            self.writer.add_scalar("training/final_combat_success_rate", self.combat_success_rate, 0)
            self.writer.add_scalar("training/final_action_success_rate", self.action_success_rate, 0)
            self.writer.add_scalar("training/total_unique_areas", len(self.unique_areas_visited), 0)
            self.writer.add_scalar("training/total_combat_engagements", self.combat_engagements, 0)
            self.writer.add_scalar("training/total_distance_moved", self.total_distance_moved, 0)
            self.writer.close()
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _calculate_combat_level(self, skills: Dict[str, Dict[str, int]]) -> int:
        """Calculate the combat level based on skill levels."""
        # Get base skill levels, defaulting to 1 if not found
        attack = skills.get("ATTACK", {}).get("level", 1)
        strength = skills.get("STRENGTH", {}).get("level", 1)
        defence = skills.get("DEFENCE", {}).get("level", 1)
        hitpoints = skills.get("HITPOINTS", {}).get("level", 10)  # Base HP is 10
        prayer = skills.get("PRAYER", {}).get("level", 1)
        ranged = skills.get("RANGED", {}).get("level", 1)
        magic = skills.get("MAGIC", {}).get("level", 1)

        # Calculate base combat level using RuneScape formula
        base = 0.25 * (defence + hitpoints + math.floor(prayer/2))
        melee = 0.325 * (attack + strength)
        range_level = 0.325 * (math.floor(3 * ranged/2))
        magic_level = 0.325 * (math.floor(3 * magic/2))

        # Use the highest of melee, range, or magic
        combat = base + max(melee, range_level, magic_level)
        
        return math.floor(combat)

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

def get_tensorboard_dir() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    tb_log_dir = os.path.join(base_dir, "tb_logs")
    return tb_log_dir


def train_combat_bot(total_timesteps: int = 1_000_000, checkpoint: Optional[str] = None) -> None:
    """Train the combat bot using PPO, automatically resuming from the latest checkpoint if available."""
    tb_log_dir = get_tensorboard_dir()
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
    
    # Configure policy kwargs with improved architecture
    policy_kwargs = dict(
        net_arch=[
            dict(
                pi=[128, 128, 64],  # Policy network
                vf=[128, 128, 64]   # Value network
            )
        ],
        activation_fn=torch.nn.ReLU,
        ortho_init=True
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
                # Adjusted parameters for better stability
                learning_rate=1e-4,           # Reduced learning rate
                n_steps=2048,                 # Increased steps per update
                batch_size=256,               # Increased batch size
                n_epochs=10,                  # More epochs per update
                gamma=0.99,                   # Standard discount factor
                gae_lambda=0.95,              # GAE parameter
                clip_range=0.2,               # Standard clip range
                clip_range_vf=0.2,            # Value function clip range
                ent_coef=0.005,              # Reduced entropy coefficient
                vf_coef=0.5,                 # Value function coefficient
                max_grad_norm=0.5,           # Gradient clipping
                target_kl=0.015,             # Target KL divergence
                verbose=1
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
            learning_rate=1e-4,           # Reduced learning rate
            n_steps=2048,                 # Increased steps per update
            batch_size=256,               # Increased batch size
            n_epochs=10,                  # More epochs per update
            gamma=0.99,                   # Standard discount factor
            gae_lambda=0.95,              # GAE parameter
            clip_range=0.2,               # Standard clip range
            clip_range_vf=0.2,            # Value function clip range
            ent_coef=0.005,              # Reduced entropy coefficient
            vf_coef=0.5,                 # Value function coefficient
            max_grad_norm=0.5,           # Gradient clipping
            target_kl=0.015,             # Target KL divergence
            verbose=1,
            device=device,
            tensorboard_log=metrics_dir,
            policy_kwargs=policy_kwargs
        )

    # Configure torch for memory optimization
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Save checkpoints more frequently early in training
    checkpoint_freq = lambda steps: 1000 if steps < 50000 else 5000
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq(0),  # Initial frequency
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
        eval_freq=5000,  # Evaluate more frequently
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
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
            tb_log_name="training",  # Use fixed name for TensorBoard logs
            log_interval=100  # Log more frequently
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
        self.actions_per_minute_limit = 30
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
        self.visited_positions: Set[Tuple[float, float]] = set()  # Add visited_positions set
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

        # Track interface state
        self.interface_open = False
        self.last_interface_check = 0
        self.interface_check_cooldown = 2.0  # Check every 2 seconds
        self.consecutive_interface_frames = 0
        self.max_interface_frames = 5  # Number of consecutive frames before considering interface stuck

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
        self.visited_positions.clear()  # Clear visited positions
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
            "in_combat": 0,  # Changed to int for Discrete
            "interfaces_open": 0,  # Changed to int for Discrete
            "path_obstructed": 0,  # Changed to int for Discrete
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
            return self._get_empty_observation()

        # Initialize screenshot with empty array
        screenshot = np.zeros(self.screenshot_shape, dtype=np.uint8)
        
        # Process screenshot if available
        if "screenshot" in state:
            screenshot_data = state["screenshot"]
            if isinstance(screenshot_data, str) and screenshot_data:
                img_data = base64.b64decode(screenshot_data)
                with Image.open(BytesIO(img_data)) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    if img.size != (self.screenshot_shape[1], self.screenshot_shape[0]):
                        img = img.resize(
                            (self.screenshot_shape[1], self.screenshot_shape[0]),
                            Image.Resampling.LANCZOS
                        )
                    screenshot = np.array(img, dtype=np.uint8)

        player = state["player"] if "player" in state else {}
        health = player["health"] if "health" in player else {}
        current_health = health["current"] if "current" in health else 1
        max_health = health["maximum"] if "maximum" in health else 1

        # Log health changes, especially after death
        if hasattr(self, '_last_health') and self._last_health != current_health:
            self.logger.warning(f"Health changed from {self._last_health} to {current_health} (max: {max_health})")
            if self._last_health is not None and self._last_health <= 0 and current_health > 0:
                self.logger.warning("Player has respawned!")
        self._last_health = current_health

        health_array = np.array([current_health], dtype=np.int32)
        prayer = np.array([player["prayer"] if "prayer" in player else 0], dtype=np.int32)
        run_energy = np.array([player["runEnergy"] if "runEnergy" in player else 0.0], dtype=np.float32)

        skills = player["skills"] if "skills" in player else {}
        combat_stats = np.array([
            skills["ATTACK"]["level"] if "ATTACK" in skills and "level" in skills["ATTACK"] else 1,
            skills["STRENGTH"]["level"] if "STRENGTH" in skills and "level" in skills["STRENGTH"] else 1,
            skills["DEFENCE"]["level"] if "DEFENCE" in skills and "level" in skills["DEFENCE"] else 1,
            skills["RANGED"]["level"] if "RANGED" in skills and "level" in skills["RANGED"] else 1,
            skills["MAGIC"]["level"] if "MAGIC" in skills and "level" in skills["MAGIC"] else 1,
            skills["HITPOINTS"]["level"] if "HITPOINTS" in skills and "level" in skills["HITPOINTS"] else 10,
            skills["PRAYER"]["level"] if "PRAYER" in skills and "level" in skills["PRAYER"] else 1
        ], dtype=np.int32)

        npcs = state["npcs"] if "npcs" in state else []
        npcs.sort(key=lambda x: x["distance"] if "distance" in x else float("inf"))
        npc_features = np.zeros((10, 6), dtype=np.float32)
        for i, npc in enumerate(npcs[:10]):
            if "health" in npc:
                health_current = npc["health"]["current"] if "current" in npc["health"] else 0
                health_max = max(npc["health"]["maximum"] if "maximum" in npc["health"] else 1, 1)
            else:
                health_current = 0
                health_max = 1
            health_ratio = health_current / health_max
            npc_features[i] = np.array([
                npc["id"] if "id" in npc else 0,
                npc["level"] if "level" in npc else 0,
                npc["distance"] if "distance" in npc else 0,
                float(npc["interacting"] if "interacting" in npc else False),
                health_ratio,
                1.0
            ], dtype=np.float32)

        skills_array = np.ones(23, dtype=np.int32)
        for i, skill in enumerate(skills.values()):
            if i < 23 and "level" in skill:
                skills_array[i] = skill["level"]

        # Process interface information
        interfaces = np.zeros((10, 4), dtype=np.float32)
        interface_text = ""
        interface_options = ""
        
        if "interfaces" in state:
            for i, interface in enumerate(state["interfaces"][:10]):  # Limit to 10 interfaces
                interfaces[i] = [
                    interface["id"] if "id" in interface else 0,
                    interface["groupId"] if "groupId" in interface else 0,
                    1 if "text" in interface else 0,
                    len(interface["options"]) if "options" in interface else 0
                ]
                
                # Store text of first interface with text
                if not interface_text and "text" in interface:
                    interface_text = interface["text"]
                
                # Store options of first interface with options
                if not interface_options and "options" in interface:
                    interface_options = "|".join(opt["text"] for opt in interface["options"])
        
        location = player["location"] if "location" in player else {}
        return {
            "screenshot": screenshot,
            "player_position": np.array([
                location["x"] if "x" in location else 0,
                location["y"] if "y" in location else 0,
                location["plane"] if "plane" in location else 0
            ], dtype=np.float32),
            "player_combat_stats": combat_stats,
            "player_health": health_array,
            "player_prayer": prayer,
            "player_run_energy": run_energy,
            "skills": skills_array,
            "npcs": npc_features,
            "in_combat": int(player["inCombat"] if "inCombat" in player else False),
            "interfaces_open": int(self.interfaces_open),
            "path_obstructed": int(self.path_obstructed),
            "current_chunk": np.array([
                state["exploration"]["currentChunk"]["x"] if "exploration" in state and "currentChunk" in state["exploration"] and "x" in state["exploration"]["currentChunk"] else 0,
                state["exploration"]["currentChunk"]["y"] if "exploration" in state and "currentChunk" in state["exploration"] and "y" in state["exploration"]["currentChunk"] else 0
            ], dtype=np.int32),
            "visited_chunks_count": np.array([
                state["exploration"]["visitedChunks"] if "exploration" in state and "visitedChunks" in state["exploration"] else 0
            ], dtype=np.int32),
            "nearby_areas": np.zeros((9, 5), dtype=np.float32),
            "exploration_score": np.array([0.0], dtype=np.float32),
            "interfaces": interfaces,
            "interface_text": interface_text,
            "interface_options": interface_options
        }

    def _calculate_reward(self, state: Optional[Dict]) -> float:
        """Compute reward based on various factors."""
        if not state:
            return 0.0
        reward = 0.0
        
        # Get player data
        player = state["player"] if "player" in state else {}
        skills = player["skills"] if "skills" in player else {}
        
        # Add interface penalty
        if self.interfaces_open:
            self.consecutive_interface_frames += 1
            if self.consecutive_interface_frames > self.max_interface_frames:
                reward -= 0.5  # Penalty for leaving interfaces open
        else:
            self.consecutive_interface_frames = 0
            
        # Combat experience rewards
        combat_skills = ["ATTACK", "STRENGTH", "DEFENCE", "RANGED", "MAGIC", "HITPOINTS"]
        current_combat_exp = sum(skills[skill]["experience"] if skill in skills and "experience" in skills[skill] else 0 for skill in combat_skills)
        if current_combat_exp > self.last_combat_exp:
            exp_gain = current_combat_exp - self.last_combat_exp
            # Scale exp rewards logarithmically to prevent exploitation
            reward += np.log1p(exp_gain) * 0.5
            if exp_gain > 1000:
                self.logger.warning(f"Major exp gain: {exp_gain}")
        self.last_combat_exp = current_combat_exp

        # Health management rewards/penalties
        health = player["health"] if "health" in player else {}
        current_health = health["current"] if "current" in health else 0
        max_health = health["maximum"] if "maximum" in health else 1
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
        in_combat = player["inCombat"] if "inCombat" in player else False
        if in_combat:
            reward += 0.3  # Base combat engagement reward
            # Additional reward for fighting appropriate level NPCs
            npcs = state["npcs"] if "npcs" in state else []
            if npcs:
                player_combat_level = self._calculate_combat_level(skills)
                for npc in npcs:
                    if npc["interacting"] if "interacting" in npc else False:  # If NPC is interacting with player
                        npc_level = npc["combatLevel"] if "combatLevel" in npc else 0
                        level_diff = abs(npc_level - player_combat_level)
                        if level_diff < 10:  # Reward fighting NPCs close to player's level
                            reward += 0.2
                        elif level_diff > 20:  # Penalty for fighting NPCs too far from player's level
                            reward -= 0.1

        # Movement and position rewards/penalties
        location = player["location"] if "location" in player else {}
        current_pos = (location["x"] if "x" in location else 0, location["y"] if "y" in location else 0)
        
        # Penalize staying still when not in combat
        if not in_combat:
            if current_pos == self.last_position:
                self.consecutive_same_pos += 1
                if self.consecutive_same_pos > 10:  # After 10 ticks of not moving
                    reward -= 0.1 * (self.consecutive_same_pos - 10)  # Increasing penalty
            else:
                # Reward exploration of new areas
                if current_pos not in self.visited_positions:
                    reward += 0.5  # Significant reward for exploring new areas
                    self.visited_positions.add(current_pos)
                self.consecutive_same_pos = 0
        self.last_position = current_pos

        # Prayer and run energy management
        prayer_points = player["prayer"] if "prayer" in player else 0
        run_energy = player["runEnergy"] if "runEnergy" in player else 0.0
        
        # Penalize completely depleted resources
        if prayer_points == 0:
            reward -= 0.1
        if run_energy < 5.0:
            reward -= 0.1

        # Add rewards for successful interface interactions
        if "interfaces" in state:
            for interface in state["interfaces"]:
                if "text" in interface and "Enter Wilderness" in interface["text"]:
                    # Give a small reward for finding the wilderness interface
                    reward += 0.1
                    
                    # Check if we're in the wilderness in the next state
                    if self.current_state and self.current_state["inWilderness"] if "inWilderness" in self.current_state else False:
                        # Give a larger reward for successfully entering
                        reward += 1.0
        
        return reward

    def _is_episode_done(self, state: Optional[Dict]) -> bool:
        """Check if the episode is done."""
        if not state:
            return True

        # Only end episode on death
        current_health = state["player"]["health"]["current"] if "player" in state and "health" in state["player"] else 0
        return current_health <= 0

    def _calculate_combat_level(self, skills: Dict[str, Dict[str, int]]) -> int:
        """Calculate the combat level based on skill levels."""
        # Get base skill levels, defaulting to 1 if not found
        attack = skills.get("ATTACK", {}).get("level", 1)
        strength = skills.get("STRENGTH", {}).get("level", 1)
        defence = skills.get("DEFENCE", {}).get("level", 1)
        hitpoints = skills.get("HITPOINTS", {}).get("level", 10)  # Base HP is 10
        prayer = skills.get("PRAYER", {}).get("level", 1)
        ranged = skills.get("RANGED", {}).get("level", 1)
        magic = skills.get("MAGIC", {}).get("level", 1)

        # Calculate base combat level using RuneScape formula
        base = 0.25 * (defence + hitpoints + math.floor(prayer/2))
        melee = 0.325 * (attack + strength)
        range_level = 0.325 * (math.floor(3 * ranged/2))
        magic_level = 0.325 * (math.floor(3 * magic/2))

        # Use the highest of melee, range, or magic
        combat = base + max(melee, range_level, magic_level)
        
        return math.floor(combat)

    def _get_gpt4_interface_decision(self, interfaces: List[Dict]) -> Optional[Dict]:
        """Use GPT-4 to decide which interface option to select."""
        if not interfaces or not OPENAI_API_KEY:
            return None

        # Format the interface options for GPT-4
        interface_descriptions = []
        for interface in interfaces:
            if "options" in interface and interface["options"]:
                options_text = [opt["text"] for opt in interface["options"]]
                interface_text = interface.get("text", "")
                interface_descriptions.append({
                    "id": interface["id"],
                    "groupId": interface["groupId"],
                    "text": interface_text,
                    "options": options_text
                })

        if not interface_descriptions:
            return None

        # Define the function for structured output
        functions = [
            {
                "name": "select_interface_option",
                "description": "Select the most appropriate interface option based on the context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interfaceId": {
                            "type": "integer",
                            "description": "The ID of the selected interface"
                        },
                        "groupId": {
                            "type": "integer",
                            "description": "The group ID of the selected interface"
                        },
                        "optionText": {
                            "type": "string",
                            "description": "The text of the selected option"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation for why this option was chosen"
                        }
                    },
                    "required": ["interfaceId", "groupId", "optionText", "reasoning"]
                }
            }
        ]

        try:
            # Call GPT-4 API with function calling
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are an AI playing RuneScape. You need to decide which interface option to click.
Consider:
1. Safety (avoid dangerous options)
2. Progress (prefer options that advance gameplay)
3. Resource management (consider health/prayer/etc.)"""
                }, {
                    "role": "user",
                    "content": f"Here are the available interfaces and their options:\n\n{json.dumps(interface_descriptions, indent=2)}\n\nWhich option should be selected?"
                }],
                functions=functions,
                function_call={"name": "select_interface_option"},
                temperature=0.2,
                max_tokens=150
            )

            # Get the function call
            function_call = response.choices[0].message.function_call
            if not function_call:
                self.logger.warning("No function call in GPT-4 response")
                return None

            # Parse the function arguments
        except Exception as e:
            self.logger.error(f"Error getting GPT-4 interface decision: {e}")
            return None

    def _action_to_command(self, action: Action) -> Optional[Dict]:
        """Convert a high-level action into a command following the schema."""
        if not self.current_state:
            return None
            
        if isinstance(action, (int, np.integer)):
            action = Action(action)
        
        player = self.current_state["player"] if "player" in self.current_state else {}
        location = player["location"] if "location" in player else {}
        current_x = location["x"] if "x" in location else 0
        current_y = location["y"] if "y" in location else 0
        
        if action == Action.ATTACK:
            # Find nearest attackable NPC
            npcs = self.current_state["npcs"] if "npcs" in self.current_state else []
            # Filter for attackable NPCs, considering level difference
            player_combat_level = self._calculate_combat_level(player["skills"] if "skills" in player else {})
            attackable_npcs = [
                npc for npc in npcs
                if ("combatLevel" in npc and 
                    abs(npc["combatLevel"] - player_combat_level) < 20 and  # Level difference check
                    not npc["interacting"] if "interacting" in npc else False and 
                    "id" in npc and npc["id"] != self.last_target_id)
            ]
            
            if not attackable_npcs:
                # Fallback to any non-interacting NPCs if none match our criteria
                attackable_npcs = [
                    npc for npc in npcs
                    if ("combatLevel" in npc and npc["combatLevel"] > 0 and 
                        not npc["interacting"] if "interacting" in npc else False)
                ]
            
            if attackable_npcs:
                # Sort by combination of distance and level appropriateness
                nearest_npc = min(attackable_npcs, 
                    key=lambda x: (x["distance"] if "distance" in x else float("inf")) * 
                                 (1 + abs(x["combatLevel"] if "combatLevel" in x else 0 - player_combat_level) / 20))
                self.last_target_id = nearest_npc["id"]
                return {
                    "action": "moveAndClick",
                    "data": {
                        "targetType": "npc",
                        "action": "Attack",
                        "npcId": nearest_npc["id"]
                    }
                }
            else:
                # No NPCs found - explore more aggressively
                # Calculate exploration direction based on visited areas
                current_chunk = (
                    self.current_state["exploration"]["currentChunk"]["x"] if "exploration" in self.current_state and "currentChunk" in self.current_state["exploration"] and "x" in self.current_state["exploration"]["currentChunk"] else 0,
                    self.current_state["exploration"]["currentChunk"]["y"] if "exploration" in self.current_state and "currentChunk" in self.current_state["exploration"] and "y" in self.current_state["exploration"]["currentChunk"] else 0
                )
                
                # Get nearby chunks that haven't been visited
                nearby_chunks = [
                    (current_chunk[0] + dx, current_chunk[1] + dy)
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                    if (current_chunk[0] + dx, current_chunk[1] + dy) not in self.visited_areas
                ]
                
                if nearby_chunks:
                    # Move towards nearest unvisited chunk
                    target_chunk = min(nearby_chunks, key=lambda c: abs(c[0] - current_chunk[0]) + abs(c[1] - current_chunk[1]))
                    dx = target_chunk[0] - current_chunk[0]
                    dy = target_chunk[1] - current_chunk[1]
                    
                    if abs(dx) > abs(dy):
                        action = Action.MOVE_RIGHT if dx > 0 else Action.MOVE_LEFT
                    else:
                        action = Action.MOVE_FORWARD if dy > 0 else Action.MOVE_BACKWARD
                else:
                    # All nearby chunks visited, move randomly
                    action = random.choice([
                        Action.MOVE_FORWARD,
                        Action.MOVE_BACKWARD,
                        Action.MOVE_LEFT,
                        Action.MOVE_RIGHT
                    ])
        
        # Movement commands
        move_distance = 2
        if action == Action.MOVE_FORWARD:
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "coordinates",
                    "action": "Move",
                    "x": current_x,
                    "y": current_y + move_distance
                }
            }
        elif action == Action.MOVE_BACKWARD:
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "coordinates",
                    "action": "Move",
                    "x": current_x,
                    "y": current_y - move_distance
                }
            }
        elif action == Action.MOVE_LEFT:
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "coordinates",
                    "action": "Move",
                    "x": current_x - move_distance,
                    "y": current_y
                }
            }
        elif action == Action.MOVE_RIGHT:
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "coordinates",
                    "action": "Move",
                    "x": current_x + move_distance,
                    "y": current_y
                }
            }
        elif action == Action.INTERFACE_ACTION:
            # Get all available interfaces
            interfaces = self.current_state["interfaces"] if "interfaces" in self.current_state else []
            
            # Find interfaces with available options
            clickable_interfaces = [
                interface for interface in interfaces
                if "options" in interface and interface["options"]
            ]
            
            if clickable_interfaces:
                # Use GPT-4 to decide which option to select
                decision = self._get_gpt4_interface_decision(clickable_interfaces)
                if decision:
                    return {
                        "action": "interfaceAction",
                        "data": decision
                    }
                else:
                    # Fallback to selecting the first option if GPT-4 fails
                    interface = clickable_interfaces[0]
                    option = interface["options"][0]
                    return {
                        "action": "interfaceAction",
                        "data": {
                            "interfaceId": interface["id"],
                            "groupId": interface["groupId"],
                            "optionText": option["text"]
                        }
                    }
            
        return None

    def _get_nearby_interactable_objects(self) -> List[Dict]:
        """Find nearby objects that can be interacted with (doors, gates, etc.)."""
        if not self.current_state:
            return []
        
        objects = self.current_state["objects"] if "objects" in self.current_state else []
        player_loc = self.current_state["player"]["location"] if "player" in self.current_state and "location" in self.current_state["player"] else {}
        current_x = player_loc["x"] if "x" in player_loc else 0
        current_y = player_loc["y"] if "y" in player_loc else 0
        
        interactable_objects = []
        for obj in objects:
            if "location" not in obj:
                continue
            obj_x = obj["location"]["x"] if "x" in obj["location"] else 0
            obj_y = obj["location"]["y"] if "y" in obj["location"] else 0
            
            # Calculate distance to object
            distance = math.sqrt((obj_x - current_x)**2 + (obj_y - current_y)**2)
            
            # Check if object has relevant actions
            actions = obj["actions"] if "actions" in obj else []
            interactable_actions = ["Open", "Close", "Enter", "Exit", "Climb", "Use"]
            
            for action in actions:
                if action in interactable_actions and distance < 10:  # Within reasonable distance
                    interactable_objects.append({
                        "id": obj["id"] if "id" in obj else None,
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
        collision_data = self.current_state["collisionData"] if "collisionData" in self.current_state else {}
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
            key = f"{check_x},{check_y}"
            if key in collision_data and collision_data[key]:
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

    def _check_interfaces_open(self) -> bool:
        """Check if any interfaces are currently open."""
        if not self.current_state:
            return False
            
        # List of interface IDs that should be closed
        interface_groups = [
            WidgetID.WORLD_MAP_GROUP_ID,  # World map
            WidgetID.INVENTORY_GROUP_ID,   # Inventory
            WidgetID.SKILLS_GROUP_ID,      # Skills
            WidgetID.EQUIPMENT_GROUP_ID,   # Equipment
            WidgetID.PRAYER_GROUP_ID,      # Prayer
            WidgetID.SPELLBOOK_GROUP_ID,   # Magic spellbook
            WidgetID.BANK_GROUP_ID,        # Bank
            WidgetID.DIALOG_GROUP_ID,      # Dialog boxes
            WidgetID.CHATBOX_GROUP_ID      # Chat interfaces
        ]
        
        widgets = self.current_state["widgets"] if "widgets" in self.current_state else {}
        for group_id in interface_groups:
            if str(group_id) in widgets and not widgets[str(group_id)]["hidden"] if "hidden" in widgets[str(group_id)] else True:
                return True
                
        return False

    def _close_interface_command(self) -> Dict:
        """Generate command to close the current interface."""
        return {
            "action": "pressKey",
            "data": {
                "key": "ESCAPE"
            }
        }

    def _log_interface_metrics(self, info: Dict) -> None:
        """Log interface-related metrics to TensorBoard."""
        if not hasattr(self, 'writer') or not self.writer:
            return
            
        timestep = self.num_timesteps
        self._log_scalar("interfaces/open", int(self.interface_open), timestep)
        self._log_scalar("interfaces/consecutive_frames", self.consecutive_interface_frames, timestep)
        
        if self.interface_open:
            self._log_scalar("interfaces/time_spent", self.interface_check_cooldown, timestep)

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