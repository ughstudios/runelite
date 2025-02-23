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
from typing import List, Optional
from torch.utils.tensorboard import SummaryWriter

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
        self.writer: Optional[SummaryWriter] = None  # Fix type hint
        self.step_in_episode = 0
        
    def _init_callback(self) -> None:
        """Initialize the callback with TensorBoard writer"""
        if not hasattr(self.model, 'tensorboard_log') or not self.model.tensorboard_log:
            return
            
        # Use the same directory as the model but in a custom_metrics subdirectory
        tb_log_dir = str(self.model.tensorboard_log)  # Convert to str to satisfy type checker
        run_dir = os.path.join(tb_log_dir, "PPO_1")  # SB3's default run name
        log_dir = os.path.join(run_dir, "custom_metrics")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        try:
            # Get info from the last step more safely
            infos = self.locals.get('infos', [])
            info = infos[0] if infos else {}
            
            rewards = self.locals.get('rewards', [])
            reward = rewards[0] if rewards else 0.0
            
            new_obs = self.locals.get('new_obs', None)
            obs = new_obs[0] if isinstance(new_obs, (list, np.ndarray)) else {}
            
            actions = self.locals.get('actions', None)
            action = actions[0] if isinstance(actions, (list, np.ndarray)) else None
            
            self.current_reward += reward
            self.step_in_episode += 1
            
            # Log to TensorBoard
            if self.writer:
                # Basic metrics
                self.writer.add_scalar('rewards/step', reward, self.num_timesteps)
                self.writer.add_scalar('rewards/cumulative', self.current_reward, self.num_timesteps)
                self.writer.add_scalar('episode/steps', self.step_in_episode, self.num_timesteps)
                
                # Get state information
                state = info.get('state', {})
                player = state.get('player', {})
                
                # Player Stats
                health = player.get('health', {})
                current_health = int(health.get('current', 100))
                max_health = int(health.get('maximum', 100))
                health_ratio = current_health / max_health if max_health > 0 else 1.0
                
                self.writer.add_scalar('player/health_current', current_health, self.num_timesteps)
                self.writer.add_scalar('player/health_max', max_health, self.num_timesteps)
                self.writer.add_scalar('player/health_ratio', health_ratio, self.num_timesteps)
                self.writer.add_scalar('player/prayer', player.get('prayer', 0), self.num_timesteps)
                self.writer.add_scalar('player/run_energy', player.get('runEnergy', 0.0), self.num_timesteps)
                self.writer.add_scalar('player/in_combat', int(player.get('inCombat', False)), self.num_timesteps)
                
                # Combat Stats
                skills = player.get('skills', {})
                for skill_name, skill_data in skills.items():
                    self.writer.add_scalar(f'skills/{skill_name.lower()}/level', skill_data.get('level', 1), self.num_timesteps)
                    self.writer.add_scalar(f'skills/{skill_name.lower()}/experience', skill_data.get('experience', 0), self.num_timesteps)
                
                # NPC Information
                npcs = state.get('npcs', [])
                if npcs:
                    nearest_npc = min(npcs, key=lambda x: x.get('distance', float('inf')))
                    self.writer.add_scalar('npcs/nearest_distance', nearest_npc.get('distance', 0), self.num_timesteps)
                    self.writer.add_scalar('npcs/nearest_level', nearest_npc.get('level', 0), self.num_timesteps)
                    self.writer.add_scalar('npcs/count', len(npcs), self.num_timesteps)
                    self.writer.add_scalar('npcs/nearest_health_ratio', 
                        nearest_npc.get('health', {}).get('current', 0) / max(nearest_npc.get('health', {}).get('maximum', 1), 1),
                        self.num_timesteps)
                
                # Location and Movement
                location = player.get('location', {})
                self.writer.add_scalar('location/x', location.get('x', 0), self.num_timesteps)
                self.writer.add_scalar('location/y', location.get('y', 0), self.num_timesteps)
                self.writer.add_scalar('location/plane', location.get('plane', 0), self.num_timesteps)
                
                # Action Information
                if action is not None:
                    if isinstance(action, (int, np.integer)):
                        # Log the action as a scalar and as text (using the Action enum)
                        self.writer.add_scalar('actions/last_action_scalar', int(action), self.num_timesteps)
                        self.writer.add_text('actions/last_action_text', Action(action).name, self.num_timesteps)
                    else:
                        self.writer.add_text('actions/last_action', str(action), self.num_timesteps)
                
                # Observation Space Logging
                if isinstance(obs, dict):
                    # Log continuous observations
                    for key in ['player_position', 'player_run_energy', 'exploration_score']:
                        if key in obs:
                            if isinstance(obs[key], np.ndarray):
                                for i, val in enumerate(obs[key].flatten()):
                                    self.writer.add_scalar(f'observations/{key}_{i}', float(val), self.num_timesteps)
                    
                    # Log discrete observations
                    for key in ['in_combat', 'interfaces_open', 'path_obstructed']:
                        if key in obs:
                            self.writer.add_scalar(f'observations/{key}', int(obs[key][0]), self.num_timesteps)
                    
                    # Log NPC observations
                    if 'npcs' in obs and isinstance(obs['npcs'], np.ndarray):
                        for i in range(min(3, obs['npcs'].shape[0])):  # Log first 3 NPCs
                            npc = obs['npcs'][i]
                            self.writer.add_scalar(f'observations/npc_{i}_distance', float(npc[2]), self.num_timesteps)
                            self.writer.add_scalar(f'observations/npc_{i}_level', float(npc[1]), self.num_timesteps)
                
                # Screenshot logging (periodically to save space)
                if self.num_timesteps % 500 == 0:  # Log every 500 steps
                    try:
                        if isinstance(obs, dict) and 'screenshot' in obs:
                            screenshot_data = obs['screenshot']
                            if isinstance(screenshot_data, np.ndarray):
                                if screenshot_data.shape[-1] == 3:  # Ensure it's RGB
                                    # Normalize the image data to [0, 1]
                                    screenshot_normalized = screenshot_data.astype(np.float32) / 255.0
                                    # If the image is in HWC format, transpose it to CHW for TensorBoard
                                    if len(screenshot_normalized.shape) == 3:
                                        screenshot_chw = np.transpose(screenshot_normalized, (2, 0, 1))
                                    else:
                                        screenshot_chw = screenshot_normalized
                                    
                                    # Log the screenshot image
                                    self.writer.add_image(
                                        'observations/screenshot',
                                        screenshot_chw,
                                        self.num_timesteps,
                                        dataformats='CHW'
                                    )
                                    self.writer.flush()  # Force write to disk
                                else:
                                    logger.error(f"Invalid screenshot channels: {screenshot_data.shape}")
                            else:
                                logger.error(f"Screenshot is not a numpy array: {type(screenshot_data)}")
                        else:
                            logger.error("No screenshot in observation")
                    except Exception as e:
                        logger.error(f"Error logging screenshot: {str(e)}", exc_info=True)
                
                # Experience gains
                if info.get('exp_gain', 0) > 0:
                    self.writer.add_scalar('rewards/exp_gain', info['exp_gain'], self.num_timesteps)
                
                # Environment Status
                self.writer.add_scalar('environment/interfaces_open', 
                                         int(info.get('interfaces_open', False)), 
                                         self.num_timesteps)
                self.writer.add_scalar('environment/path_obstructed', 
                                         int(info.get('path_obstructed', False)), 
                                         self.num_timesteps)
            
            # Only log significant events to console/log files
            if info.get('exp_gain', 0) > 1000:  # Only log major exp gains
                logger.warning(f"Major exp gain: {info['exp_gain']:,}")
            
            if current_health < max_health * 0.2:  # Critical health threshold
                logger.error(f"Critical health: {current_health}/{max_health}")
            
            # Periodic status update
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                logger.warning(f"Status - Episode: {self.episode_count}, Current Reward: {self.current_reward:.2f}")
                self.last_log_time = current_time
                
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
            # Don't fail the training loop due to logging errors
            
        return True
        
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        """
        self.episode_count += 1
        self.step_in_episode = 0
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if self.writer:
            self.writer.add_scalar('episode/total_reward', self.current_reward, self.episode_count)
            self.writer.add_scalar('episode/length', self.step_in_episode, self.episode_count)
        self.rewards.append(self.current_reward)
        self.current_reward = 0
        
    def _on_training_end(self) -> None:
        """Cleanup when training ends"""
        if self.writer:
            self.writer.close()

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.warning(f"Using CUDA device: {device_name}")  # Important setup info
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.warning("Using Apple MPS (Metal) device")  # Important setup info
    else:
        device = "cpu"
        logger.warning("Using CPU device")  # Important setup info
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
    run_name = f"PPO_{timestamp}"
    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    tb_log_dir = os.path.join(base_dir, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    
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
        ]
    )
    
    device = get_device()
    
    # Initialize model with tensorboard logging
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,  # Enable verbosity for tensorboard
        device=device,
        tensorboard_log=tb_log_dir
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{tb_log_dir}/checkpoints",
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
        ]
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{tb_log_dir}/eval",
        log_path=f"{tb_log_dir}/eval",
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
    
    model.save(f"{tb_log_dir}/final_model")
    env.save(f"{tb_log_dir}/vec_normalize.pkl")
    
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