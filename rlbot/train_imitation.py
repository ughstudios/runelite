from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
from runescape_env import RuneScapeEnv, Action
import torch
import json
import logging
from rich.logging import RichHandler
from rich.console import Console
from datetime import datetime
import base64
from PIL import Image
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import time

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

class ImitationCallback(BaseCallback):
    """Custom callback for processing imitation learning data"""
    
    def __init__(self, recording_dir: str = "recordings"):
        super().__init__(verbose=0)  # Disable base callback logging
        self.recording_dir = Path(recording_dir)
        self.recordings: List[Path] = []
        self.current_recording: Optional[Path] = None
        self.current_step: int = 0
        self.demonstration_data: Optional[Dict] = None
        self.last_log_time = time.time()
        self.log_interval = 60  # Only log every 60 seconds
        
    def _init_callback(self) -> None:
        """Load all recordings at the start of training."""
        # Get all recording directories
        if self.recording_dir.exists():
            self.recordings = [
                d for d in self.recording_dir.iterdir()
                if d.is_dir() and any(f.suffix == '.json' for f in d.iterdir())
            ]
            self.recordings.sort(key=str)  # Sort by string representation
            if self.recordings:
                logger.warning(f"Found {len(self.recordings)} recordings for imitation learning")
            else:
                logger.error("No recordings found for imitation learning")
                
    def _on_step(self) -> bool:
        """Load next demonstration if needed and compute imitation loss."""
        # Load next demonstration if needed
        if not self.demonstration_data:
            if not self.recordings:
                return True
                
            self.current_recording = self.recordings.pop(0)
            demo_file = next(self.current_recording.glob('*.json'))
            with open(demo_file) as f:
                self.demonstration_data = json.load(f)
                
        # Compute imitation loss
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            logger.warning(f"Training progress - Step: {self.num_timesteps}, Recordings left: {len(self.recordings)}")
            self.last_log_time = current_time
            
        return True

    def _process_actions(self, actions: List[Dict]) -> Optional[Action]:
        """Convert recorded actions into environment actions"""
        if not actions:
            return None
            
        # For now, just handle basic mouse clicks
        for action in actions:
            if action["type"] == "click":
                # Convert click coordinates to game action
                # This is a simplified version - you'll need to expand this
                # based on your game's mechanics and UI layout
                x, y = action["x"], action["y"]
                
                # Example: Convert click to game action based on position
                if y < 100:  # Top of screen
                    return Action.MOVE_FORWARD
                elif y > 400:  # Bottom of screen
                    return Action.MOVE_BACKWARD
                elif x < 200:  # Left side
                    return Action.MOVE_LEFT
                elif x > 600:  # Right side
                    return Action.MOVE_RIGHT
                else:
                    return Action.ATTACK
                    
        return None

    def _update_policy(self) -> None:
        """Update the policy using the collected demonstration data"""
        try:
            # Convert demonstrations to training data
            states = []
            actions = []
            
            for demo in self.demonstration_data:
                # Process state and screenshot
                observation = self.training_env.get_attr('env')[0]._process_state(demo["state"])
                if demo["screenshot"]:
                    screenshot = self._process_screenshot(demo["screenshot"])
                    observation["screenshot"] = screenshot
                    
                states.append(observation)
                actions.append(demo["action"].value)
            
            # Convert to numpy arrays
            states_array = np.array(states)
            actions_array = np.array(actions)
            
            # Update policy using behavioral cloning
            self.model.policy.train()
            self.model.policy.set_training_mode(True)
            
            # Get action probabilities from current policy
            with torch.no_grad():
                actions_pred, _ = self.model.policy.forward(states_array)
            
            # Compute loss (cross entropy between predicted and demonstrated actions)
            loss = torch.nn.functional.cross_entropy(
                torch.tensor(actions_pred),
                torch.tensor(actions_array, dtype=torch.long)
            )
            
            # Backward pass and optimization
            self.model.policy.optimizer.zero_grad()
            loss.backward()
            self.model.policy.optimizer.step()
            
            # Only log loss periodically
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                logger.warning(f"Training Status - Loss: {loss.item():.4f}, Steps: {self.num_timesteps}")
                self.last_log_time = current_time
            
        except Exception as e:
            logger.error(f"Error updating policy: {e}", exc_info=True)

    def _process_screenshot(self, screenshot_base64: str) -> np.ndarray:
        """Convert base64 screenshot to numpy array"""
        try:
            # Decode base64 string to image
            img_data = base64.b64decode(screenshot_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image to match environment's expected dimensions
            target_size = self.training_env.get_attr('env')[0].screenshot_shape[:2]
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            return np.array(img, dtype=np.uint8)
            
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            return np.zeros(self.training_env.get_attr('env')[0].screenshot_shape, dtype=np.uint8)

def train_with_imitation(total_timesteps=1000000, recordings_dir="recordings"):
    """Train the bot using imitation learning from recorded gameplay"""
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/imitation_{timestamp}"
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
    
    logger.info(f"Starting imitation learning session: {timestamp}")
    logger.info(f"Log directory: {log_dir}")
    
    # Create and wrap environment
    env = DummyVecEnv([lambda: Monitor(RuneScapeEnv(task="combat"), "logs/train")])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0
    )
    
    # Initialize PPO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=0.0001,  # Lower learning rate for imitation
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device=device,
        tensorboard_log=f"{log_dir}/tensorboard"
    )
    
    # Setup callbacks
    imitation_callback = ImitationCallback(recording_dir=recordings_dir)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="imitation_bot"
    )
    
    eval_env = DummyVecEnv([lambda: Monitor(RuneScapeEnv(task="combat"), "logs/eval")])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0
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
    
    try:
        logger.info("Starting imitation learning...")
        logger.info(f"View training progress with: tensorboard --logdir={log_dir}/tensorboard")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[imitation_callback, checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        logger.info("Training completed successfully!")
        model.save(f"{log_dir}/final_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        model.save(f"{log_dir}/interrupted_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    try:
        console.print("RuneScape Imitation Learning")
        console.print("Make sure you have recorded gameplay data in the recordings directory")
        train_with_imitation()
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}", exc_info=True)
        raise 