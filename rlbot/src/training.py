
"""
Training functions for the RuneScape RL agent.

This module provides functions for training and evaluating reinforcement learning
agents for the RuneScape environment.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict, Union

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from pydantic import BaseModel

from .environment import RuneScapeEnv
from .extractors import CombinedExtractor


class Location(BaseModel):
    """Location coordinates model."""

    x: int
    y: int
    plane: int


class Health(BaseModel):
    """Health status model."""

    current: int
    maximum: int


class Skill(BaseModel):
    """Skill level and experience model."""

    level: int
    realLevel: int
    experience: int


class PlayerInfo(BaseModel):
    """Player information model based on state schema."""

    name: Optional[str] = None
    combatLevel: Optional[int] = None
    location: Optional[Location] = None
    health: Optional[Health] = None
    inCombat: bool = False
    isRunning: bool = False
    runEnergy: float = 0.0
    skills: Dict[str, Skill] = {}
    prayer: Optional[int] = None


class NPC(BaseModel):
    """NPC model."""

    id: int
    name: str
    combatLevel: Optional[int] = None
    location: Optional[Location] = None
    health: Optional[Health] = None
    interacting: bool = False
    distance: float = 0.0


class GameObject(BaseModel):
    """Game object model."""

    id: int
    name: str
    location: Location
    actions: List[str] = []


class GroundItem(BaseModel):
    """Ground item model."""

    id: int
    name: str
    quantity: int
    location: Location


class Interface(BaseModel):
    """Interface element model."""

    id: int
    type: str
    text: str
    actions: List[str] = []


class Exploration(BaseModel):
    """Exploration data model."""

    currentChunk: Dict[str, int] = {}
    visitedChunks: int = 0


class GameState(BaseModel):
    """Game state model based on the state schema."""

    player: Optional[PlayerInfo] = None
    npcs: List[NPC] = []
    objects: List[GameObject] = []
    groundItems: List[GroundItem] = []
    interfaces: List[Interface] = []
    interfacesOpen: bool = False
    pathObstructed: bool = False
    exploration: Optional[Exploration] = None
    screenshot: Optional[str] = None


def create_ppo_agent(
    env: Union[gym.Env, DummyVecEnv],
    model_dir: str,
    log_dir: Optional[str] = None,
    **kwargs: Any,
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
    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=model_dir,
        name_prefix="runelite_bot",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )

    callbacks: List[BaseCallback] = [checkpoint_callback]
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=kwargs["learning_rate"] if "learning_rate" in kwargs else 3e-4,
        n_steps=kwargs["n_steps"] if "n_steps" in kwargs else 2048,
        batch_size=kwargs["batch_size"] if "batch_size" in kwargs else 64,
        n_epochs=kwargs["n_epochs"] if "n_epochs" in kwargs else 10,
        gamma=kwargs["gamma"] if "gamma" in kwargs else 0.99,
        gae_lambda=kwargs["gae_lambda"] if "gae_lambda" in kwargs else 0.95,
        clip_range=kwargs["clip_range"] if "clip_range" in kwargs else 0.2,
        clip_range_vf=kwargs["clip_range_vf"] if "clip_range_vf" in kwargs else None,
        normalize_advantage=kwargs["normalize_advantage"]
        if "normalize_advantage" in kwargs
        else True,
        ent_coef=kwargs["ent_coef"] if "ent_coef" in kwargs else 0.01,
        vf_coef=kwargs["vf_coef"] if "vf_coef" in kwargs else 0.5,
        max_grad_norm=kwargs["max_grad_norm"] if "max_grad_norm" in kwargs else 0.5,
        use_sde=kwargs["use_sde"] if "use_sde" in kwargs else False,
        sde_sample_freq=kwargs["sde_sample_freq"]
        if "sde_sample_freq" in kwargs
        else -1,
        target_kl=kwargs["target_kl"] if "target_kl" in kwargs else None,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=kwargs["seed"] if "seed" in kwargs else 0,
        device=kwargs["device"] if "device" in kwargs else "auto",
    )
    return model, callbacks


def train_bot(
    debug: bool = False, verbose: bool = False, timesteps: int = 1000000
) -> None:
    """Train a bot.
    Args:
        debug: Whether to enable debug logging
        verbose: Whether to enable verbose output
        timesteps: Number of timesteps to train for
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./rlbot/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug or verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"train_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("RuneScapeBotTrainer")
    checkpoint_dir = Path(f"./rlbot/checkpoints/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = Path("./rlbot/logs/tb_logs")
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    env = RuneScapeEnv(debug=debug)
    max_login_attempts = 5
    is_player_logged_in = False

    for attempt in range(max_login_attempts):
        state = env.reset()
        if not state:
            time.sleep(10)
            continue

        raw_state = getattr(env, "state", None)
        if not raw_state:
            time.sleep(10)
            continue

        try:
            game_state = GameState.model_validate(raw_state)
            if game_state.player and game_state.player.name:
                is_player_logged_in = True
                break
        except Exception as e:
            logger.error(f"Error parsing game state: {str(e)}")

        time.sleep(10)

    if not is_player_logged_in:
        logger.error(
            "Failed to detect logged in player. Please log into RuneScape and try again."
        )
        env.close()
        return

    monitor_path = f"./rlbot/logs/monitor_{timestamp}"
    env_monitored: Monitor = Monitor(env, monitor_path)
    env_vec = DummyVecEnv([lambda: env_monitored])

    model, callbacks = create_ppo_agent(
        env_vec,
        model_dir=str(checkpoint_dir),
        log_dir=str(tb_log_dir),
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    model.learn(
        total_timesteps=timesteps, callback=CallbackList(callbacks), progress_bar=True
    )

    final_model_path = f"./rlbot/models/runescape_bot_{timestamp}"
    model.save(final_model_path)


def test_connection(debug: bool = False) -> Dict[str, Any]:
    """Test the connection to the RuneLite client by attempting to reset the environment."""
    env = RuneScapeEnv(debug=debug)
    state = env.reset()
    env.close()
    if state:
        return {}
    return {"error": "No state returned"}
