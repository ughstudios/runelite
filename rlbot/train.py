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
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

# Set up rich console logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("rlbot")

class VerboseCallback(BaseCallback):
    """Custom callback for detailed training information"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.step_count = 0
        self.last_total_exp = None
        self.episode_rewards = []
        self.current_reward = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        env = self.training_env.envs[0].env
        
        # Log the action taken
        action = self.locals.get('actions')[0]
        action_name = list(Action)[action].name
        logger.info(f"[blue]Taking action: {action_name}[/blue]")
        
        # Log reward received
        reward = self.locals.get('rewards')[0]
        if reward != 0:
            logger.info(f"[green]Received reward: {reward:.2f}[/green]")
        
        # Get current state information
        if hasattr(env, 'current_state') and env.current_state:
            # Track experience gains
            total_exp = sum(skill.get('experience', 0) for skill in env.current_state.get('skills', {}).values())
            if self.last_total_exp is not None:
                exp_gain = total_exp - self.last_total_exp
                if exp_gain > 0:
                    logger.info(f"[green]Experience gained: {exp_gain:,}[/green]")
                    self.logger.record('train/exp_gain', exp_gain)
            self.last_total_exp = total_exp
            
            # Log player state
            health = env.current_state.get('playerHealth', 0)
            max_health = env.current_state.get('playerMaxHealth', 1)
            prayer = env.current_state.get('playerPrayer', 0)
            run_energy = env.current_state.get('playerRunEnergy', 0)
            
            if self.step_count % 10 == 0:  # Log every 10 steps to avoid spam
                logger.info(f"Player State - Health: {health}/{max_health} | Prayer: {prayer} | Run Energy: {run_energy:.1f}%")
            
            # Log NPC interactions
            npcs = env.current_state.get('npcs', [])
            interacting_npcs = [npc for npc in npcs if npc.get('interacting')]
            if interacting_npcs:
                logger.info(f"[yellow]Interacting with {len(interacting_npcs)} NPCs[/yellow]")
                for npc in interacting_npcs:
                    logger.info(f"  - {npc.get('name', 'Unknown')} (Level {npc.get('combatLevel', '?')}) - Health: {npc.get('health', '?')}")
        
        # Track rewards
        info = self.locals.get('infos')[0]
        if info.get('terminal_observation') is not None:  # Episode ended
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)
            avg_reward = np.mean(self.episode_rewards[-100:])
            logger.info(f"[bold]Episode {self.episode_count} finished![/bold]")
            logger.info(f"Reward: {self.current_reward:.2f} | Last 100 Average: {avg_reward:.2f}")
            self.current_reward = 0
        else:
            self.current_reward += reward
        
        return True

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"[bold green]Using CUDA device: {device_name}[/bold green]")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("[bold yellow]Using Apple MPS (Metal) device[/bold yellow]")
    else:
        device = "cpu"
        logger.info("[bold red]Using CPU device[/bold red]")
    return torch.device(device)

def make_env(task="combat"):
    """Create and wrap the RuneScape environment"""
    logger.info(f"Creating environment for task: {task}")
    env = RuneScapeEnv(task=task)
    env = Monitor(env, "logs/train")
    return env

def train_combat_bot(total_timesteps=1000000):
    """Train the combat bot using PPO"""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/combat_bot_{timestamp}"
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
    
    logger.info(f"[bold]Starting training session: {timestamp}[/bold]")
    logger.info(f"Log directory: {log_dir}")
    
    # Create and vectorize environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8
    )
    
    device = get_device()
    
    # Initialize PPO model with enhanced parameters
    logger.info("Initializing PPO model...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"{log_dir}/tensorboard",
        device=device
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="combat_bot",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/eval",
        log_path=f"{log_dir}/eval",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    verbose_callback = VerboseCallback()
    
    try:
        logger.info("[bold green]Starting model training...[/bold green]")
        logger.info("Press Ctrl+C to stop training gracefully")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, verbose_callback],
            tb_log_name="combat_training",
            progress_bar=True
        )
        
        logger.info("[bold green]Training completed successfully![/bold green]")
        model.save(f"{log_dir}/final_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
        
    except KeyboardInterrupt:
        logger.warning("[bold yellow]Training interrupted by user[/bold yellow]")
        model.save(f"{log_dir}/interrupted_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
    except Exception as e:
        logger.error(f"[bold red]Training failed with error: {str(e)}[/bold red]", exc_info=True)
        raise
    finally:
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
    try:
        console.print("[bold blue]RuneScape AI Training Bot[/bold blue]")
        console.print("Make sure RuneLite is running with the RLBot plugin enabled")
        console.print("Waiting for WebSocket connection...\n")
        train_combat_bot(total_timesteps=1000000)
    except Exception as e:
        logger.error(f"[bold red]Program failed with error: {str(e)}[/bold red]", exc_info=True)
        raise 