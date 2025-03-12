#!/usr/bin/env python3
"""
Training functions for the RuneScape RL agent.

This module provides functions for training and evaluating reinforcement learning
agents for the RuneScape environment.
"""

import os
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .environment import RuneScapeEnv
from .extractors import CombinedExtractor


# Initialize console for pretty output
console = Console()

def make_env(
    task: str = "combat", 
    debug: bool = False, 
    seed: Optional[int] = None
) -> Callable[[], gym.Env]:
    """Factory function to create a RuneScape environment.
    
    Args:
        task: The task to perform ("combat", "fishing", etc.)
        debug: Whether to enable debug logging
        seed: Optional random seed
        
    Returns:
        A function that creates and returns a RuneScape environment
    """
    def _init() -> gym.Env:
        # Create the environment
        env = RuneScapeEnv(task=task, debug=debug)
        
        # Set the seed if provided
        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset()
            
        return env
    
    return _init


def create_ppo_agent(
    env: gym.Env, 
    model_dir: str, 
    log_dir: Optional[str] = None, 
    **kwargs
) -> Tuple[PPO, List[BaseCallback]]:
    """Create a PPO agent with the specified parameters.
    
    Args:
        env: The environment to train on
        model_dir: Directory to save models
        log_dir: Directory to save logs
        **kwargs: Additional parameters for PPO
        
    Returns:
        Tuple of (PPO model, list of callbacks)
    """
    # Set up the policy network architecture with the combined feature extractor
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix="runelite_bot",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    
    # Set up callbacks
    callbacks: List[BaseCallback] = [checkpoint_callback]
    
    # Create the PPO agent
    model = PPO(
        "MultiInputPolicy",  # For Dict observation spaces
        env,
        learning_rate=kwargs.get("learning_rate", 3e-4),
        n_steps=kwargs.get("n_steps", 2048),
        batch_size=kwargs.get("batch_size", 64),
        n_epochs=kwargs.get("n_epochs", 10),
        gamma=kwargs.get("gamma", 0.99),
        gae_lambda=kwargs.get("gae_lambda", 0.95),
        clip_range=kwargs.get("clip_range", 0.2),
        clip_range_vf=kwargs.get("clip_range_vf", None),
        normalize_advantage=kwargs.get("normalize_advantage", True),
        ent_coef=kwargs.get("ent_coef", 0.01),
        vf_coef=kwargs.get("vf_coef", 0.5),
        max_grad_norm=kwargs.get("max_grad_norm", 0.5),
        use_sde=kwargs.get("use_sde", False),
        sde_sample_freq=kwargs.get("sde_sample_freq", -1),
        target_kl=kwargs.get("target_kl", None),
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=kwargs.get("seed", 0),
        device=kwargs.get("device", "auto")
    )
    
    return model, callbacks


def test_connection(debug: bool = True) -> None:
    """Test the connection to RuneLite by creating an environment and getting initial state.
    
    Args:
        debug: Whether to enable debug logging
    """
    console.print("[bold cyan]Testing connection to RuneLite[/bold cyan]")
    console.print("Initializing environment...")
    
    # Initialize the environment
    env = RuneScapeEnv(task="combat", debug=debug)
    
    # Wait a bit for the connection to establish
    time.sleep(2.0)
    
    # Check if we received a state
    if env.state is None:
        console.print("[bold red]Failed to get state from RuneLite[/bold red]")
        console.print("Please make sure:")
        console.print("1. RuneLite is running with the RLBot plugin")
        console.print("2. You are logged into the game")
    else:
        console.print("[bold green]Successfully connected to RuneLite![/bold green]")
        
        # Show player position if available
        if "player" in env.state and "location" in env.state["player"]:
            player_x = env.state["player"]["location"]["x"]
            player_y = env.state["player"]["location"]["y"]
            console.print(f"Player position: ({player_x}, {player_y})")
        else:
            console.print("Player position: (N/A, N/A)")
    
    # Clean up
    env.close()


def train_combat_bot(debug: bool = False, verbose: bool = False, timesteps: int = 1000000) -> None:
    """Train a bot to fight NPCs.
    
    Args:
        debug: Whether to enable debug logging
        verbose: Whether to enable verbose output
        timesteps: Number of timesteps to train for
    """
    # Create a structured logging path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./rlbot/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = logging.DEBUG if debug or verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"train_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("RuneScapeBotTrainer")
    
    # Create directories
    checkpoint_dir = Path(f"./rlbot/checkpoints/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = Path("./rlbot/logs/tb_logs")
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Let the user know what's happening
    console.print("[bold cyan]Starting training session[/bold cyan]")
    
    # Create environment
    console.print("Creating environment...")
    env = RuneScapeEnv(task="combat", debug=debug)
    
    # Verify the environment is working
    logger.info("Verifying environment initialization")
    initial_obs = env.reset()[0]
    logger.info("Initial state verification successful!")
    
    # Wrap the environment for stable-baselines3
    monitor_path = f"./rlbot/logs/monitor_{timestamp}"
    env = Monitor(env, monitor_path)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO agent and callbacks
    model, callbacks = create_ppo_agent(
        env,
        model_dir=str(checkpoint_dir),
        log_dir=str(tb_log_dir),
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4
    )
    
    # Start training
    console.print(f"[bold green]Starting training for {timesteps} timesteps...[/bold green]")
    
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = f"./rlbot/models/runescape_bot_{timestamp}"
    model.save(final_model_path)
    
    logger.info("Training completed successfully!")
    console.print("[bold green]Training completed successfully![/bold green]")
    console.print(f"Final model saved to: {final_model_path}") 