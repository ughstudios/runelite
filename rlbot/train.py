from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
from runescape_env import RuneScapeEnv, Action
import torch
import numpy as np
from datetime import datetime
import logging
from rich.logging import RichHandler
from rich.console import Console
import json
from jsonschema import validate
import time
from typing import List

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.ERROR,  # Only show errors by default
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
console = Console()

class TrainingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.episode_count = 0
        self.current_reward = 0
        self.rewards: List[float] = []
        self.last_log_time = time.time()
        self.log_interval = 60  # Only log every 60 seconds
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Get info from the last step
        info = self.locals.get('infos', [{}])[0]
        reward = self.locals.get('rewards', [0.0])[0]
        
        self.current_reward += reward
        
        # Only log significant events
        if info.get('exp_gain', 0) > 1000:  # Only log major exp gains
            logger.warning(f"Major exp gain: {info['exp_gain']:,}")
        
        # Get health from state structure
        state = info.get('state', {})
        player = state.get('player', {})
        health = player.get('health', {})
        current_health = health.get('current', 100)
        max_health = health.get('maximum', 100)
        
        if current_health < max_health * 0.2:  # Critical health threshold
            logger.error(f"Critical health: {current_health}/{max_health}")
        
        # Periodic status update
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            logger.warning(f"Status - Episode: {self.episode_count}, Current Reward: {self.current_reward:.2f}")
            self.last_log_time = current_time
            
        return True  # If the callback returns False, training is aborted early
        
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        """
        self.episode_count += 1
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.rewards.append(self.current_reward)
        self.current_reward = 0

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.warning(f"Using CUDA device: {device_name}")  # Keep this as warning - important setup info
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.warning("Using Apple MPS (Metal) device")  # Keep this as warning - important setup info
    else:
        device = "cpu"
        logger.warning("Using CPU device")  # Keep this as warning - important setup info
    return torch.device(device)

def make_env(task="combat"):
    """Create and wrap the RuneScape environment"""
    env = RuneScapeEnv(task=task)
    env = Monitor(env, "logs/train", allow_early_resets=True)
    return env

def train_combat_bot(total_timesteps=1000000):
    """Train the combat bot using PPO"""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/combat_bot_{timestamp}"
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    
    # Create and vectorize environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
        norm_obs_keys=[
            'player_position',
            'player_combat_stats',
            'player_health',
            'player_prayer',
            'player_run_energy',
            'skills',
            'npcs',
            'current_chunk',
            'visited_chunks_count',
            'nearby_areas',
            'exploration_score'
        ]  # Only normalize continuous observation spaces
    )
    
    device = get_device()
    
    # Initialize PPO model with enhanced parameters
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,  # Reduce SB3 verbosity
        device=device
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="combat_bot",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=0  # Reduce callback verbosity
    )
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
        norm_obs_keys=[
            'player_position',
            'player_combat_stats',
            'player_health',
            'player_prayer',
            'player_run_energy',
            'skills',
            'npcs',
            'current_chunk',
            'visited_chunks_count',
            'nearby_areas',
            'exploration_score'
        ]  # Only normalize continuous observation spaces
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/eval",
        log_path=f"{log_dir}/eval",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0  # Reduce callback verbosity
    )
    
    training_callback = TrainingCallback()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, training_callback],
        progress_bar=True
    )
    
    model.save(f"{log_dir}/final_model")
    env.save(f"{log_dir}/vec_normalize.pkl")
    
    env.close()
    eval_env.close()

def test_combat_bot(model_path, vec_normalize_path):
    """Test a trained combat bot"""
    logger.info(f"Testing model from {model_path}")
    
    # Load the saved statistics
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    
    device = get_device()
    
    # Load the model
    model = PPO.load(model_path, env=env, device=device)
    
    # Test the model
    obs = env.reset()
    total_reward = 0
    episode_rewards = []
    
    try:
        logger.info("Starting test episodes")
        while len(episode_rewards) < 10:  # Run 10 episodes
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if done[0]:
                logger.info(f"Episode {len(episode_rewards) + 1} finished with reward: {total_reward}")
                episode_rewards.append(total_reward)
                total_reward = 0
                obs = env.reset()
        
        logger.info("\nTest Results:")
        logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
        logger.info(f"Standard deviation: {np.std(episode_rewards):.2f}")
        logger.info(f"Min reward: {np.min(episode_rewards):.2f}")
        logger.info(f"Max reward: {np.max(episode_rewards):.2f}")
        
    finally:
        env.close()

if __name__ == "__main__":
    console.print("RuneScape AI Training Bot")
    console.print("Make sure RuneLite is running with the RLBot plugin enabled")
    console.print("Waiting for WebSocket connection...\n")
    train_combat_bot(total_timesteps=1000000) 