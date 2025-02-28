#!/usr/bin/env python3
"""
RuneScape AI Training Bot using PPO with Stable Baselines3.

This module defines a custom Gymnasium environment for RuneScape and associated training/testing
utilities for training a combat bot. It leverages PPO from Stable Baselines3 and logs custom metrics
to TensorBoard.
"""

import gymnasium as gym
from gymnasium import spaces

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

# -----------------------------------------------------------------------------
# Logging & Console Setup
# -----------------------------------------------------------------------------
# Make sure the log directory exists
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.WARNING,  # Changed from ERROR to WARNING
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "rlbot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

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
    TOGGLE_RUN = 6  # Add new action
    
    def __int__(self):
        return self.value

class RuneScapeEnv(gym.Env):
    """Custom Environment for controlling a character in RuneScape."""

    def __init__(self, task: str = "combat"):
        super().__init__()
        
        self.task = task
        self.logger = logging.getLogger("RuneScapeEnv")
        
        # Define skills to track and create tracking variables for environment state
        self.tracked_skills = ["attack", "strength", "defence", "hitpoints", "ranged", "magic", "prayer"]
        self.state: Optional[Dict[str, Any]] = None
        self.last_action_time: float = 0.0
        self.actions_per_minute: float = MIN_ACTIONS_PER_MINUTE
        self.action_times: List[float] = []
        self.skill_xp: Dict[str, int] = {skill: 0 for skill in self.tracked_skills}
        self.explored_locations: Set[Tuple[int, int]] = set()
        self.last_position: Optional[Tuple[int, int]] = None
        self.timestep: int = 0
        self.current_target: Optional[str] = None
        self.is_in_combat: bool = False
        self.last_combat_time: float = 0
        self.player_health: int = 100
        self.max_player_health: int = 100
        self.initial_skill_xp: Dict[str, int] = {}
        
        # WebSocket connection setup
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.connection_event = threading.Event()
        
        # Asyncio setup
        self.loop = asyncio.new_event_loop()
        nest_asyncio.apply(self.loop)
        
        # Start websocket client in a separate thread
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()
        
        # Wait for initial connection
        self.logger.info("Waiting for WebSocket connection...")
        if not self.connection_event.wait(30):
            self.logger.error("Failed to connect within 30 seconds")
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Action))
        
        # Define observation space for multidimensional inputs
        self.observation_space = spaces.Dict({
            "screenshot": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "player_position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "player_combat_stats": spaces.Box(low=0, high=np.inf, shape=(len(self.tracked_skills),), dtype=np.float32),
            "player_health": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "nearby_npcs": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_NPCS, 4), dtype=np.float32),  # x, y, level, distance
            "in_combat": spaces.Discrete(2),
            "action_cooldown": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        
        # Attempt to get initial state
        self.state = self._get_state()

    def _run_websocket_loop(self) -> None:
        """Run the websocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_client())

    def _calculate_combat_level(self, skills: Dict[str, int]) -> int:
        """
        Calculate the player's combat level based on their skills.
        
        Args:
            skills: Dictionary of skill levels with keys as skill names
            
        Returns:
            int: The calculated combat level
        """
        # Get base skill levels with defaults if not found
        attack = skills.get("attack", 1)
        strength = skills.get("strength", 1)
        defence = skills.get("defence", 1)
        hitpoints = skills.get("hitpoints", 10)
        prayer = skills.get("prayer", 1)
        ranged = skills.get("ranged", 1)
        magic = skills.get("magic", 1)
        
        # Calculate base combat level
        base = 0.25 * (defence + hitpoints + math.floor(prayer / 2))
        melee = 0.325 * (attack + strength)
        range_cb = 0.325 * (math.floor(ranged * 1.5))
        magic_cb = 0.325 * (math.floor(magic * 1.5))
        
        # Calculate final level using maximum of combat types
        combat_level = math.floor(base + max(melee, range_cb, magic_cb))
        
        # Ensure level is at least 3 and at most 126
        return max(3, min(126, combat_level))
        
    async def _websocket_client(self) -> None:
        """Handle websocket connection and message processing."""
        while True:
            try:
                if not self.ws:
                    self.logger.error("WebSocket is not connected")
                    await asyncio.sleep(5)
                    continue
                    
                async with self.ws as websocket:
                    self.ws = websocket
                    self.logger.info("WebSocket connection established")
                    while True:
                        try:
                            message = await websocket.recv()
                            
                            # Handle different message types properly
                            if isinstance(message, bytes):
                                self.logger.debug(f"Received binary message of {len(message)} bytes")
                                try:
                                    message_str = message.decode('utf-8', errors='replace')
                                except Exception as e:
                                    self.logger.warning(f"Could not decode binary message: {e}")
                                    continue
                            else:
                                self.logger.debug(f"Received text message: {message[:100]}..." if len(message) > 100 else f"Received text message: {message}")
                                message_str = message
                                
                            # Process the message
                            try:
                                data = json.loads(message_str)
                                if isinstance(data, dict):
                                    if data.get("type") == "screenshot":
                                        if self.state is not None:
                                            self.state["screenshot"] = data.get("data")
                                        continue
                                    try:
                                        validate(instance=data, schema=self.state_schema)
                                        self.state = data
                                        self.interfaces_open = data.get("interfacesOpen", False)
                                        self.path_obstructed = data.get("pathObstructed", False)
                                    except ValidationError as ve:
                                        self.logger.warning(f"State validation error: {ve.message}")
                            except json.JSONDecodeError as je:
                                self.logger.warning(f"JSON decode error: {je}")
                        except websockets.exceptions.ConnectionClosed:
                            self.logger.warning("Connection closed during receive")
                            break
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
            except websockets.ConnectionClosed as e:
                self.logger.warning(f"WebSocket connection closed: {e}")
                self.ws = None
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                self.ws = None
                await asyncio.sleep(5)

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        """Helper method to log a scalar value to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def _log_interface_metrics(self, info: Dict) -> None:
        """Log interface-related metrics to TensorBoard."""
        if self.writer is None:
            return
            
        self._log_scalar("interfaces/open", float(self.interface_open), self.num_timesteps)
        self._log_scalar("interfaces/consecutive_frames", float(self.consecutive_interface_frames), self.num_timesteps)
        
        if self.interface_open:
            self._log_scalar("interfaces/time_spent", float(self.interface_check_cooldown), self.num_timesteps)

    def _action_to_command(self, action: Action) -> Optional[Dict]:
        """Convert a high-level action into a command following the schema."""
        if not self.state:
            return None
            
        if isinstance(action, (int, np.integer)):
            action = Action(action)
        
        player = self.state["player"] if "player" in self.state else {}
        location = player["location"] if "location" in player else {}
        current_x = location["x"] if "x" in location else 0
        current_y = location["y"] if "y" in location else 0
        
        if action == Action.ATTACK:
            # Find nearest attackable NPC
            npcs = self.state["npcs"] if "npcs" in self.state else []
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
                                 (1 + abs(x["combatLevel"] if "combatLevel" in x else 0 - player_combat_level) / 20)
                )
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
                    self.state["exploration"]["currentChunk"]["x"] if "exploration" in self.state and "currentChunk" in self.state["exploration"] and "x" in self.state["exploration"]["currentChunk"] else 0,
                    self.state["exploration"]["currentChunk"]["y"] if "exploration" in self.state and "currentChunk" in self.state["exploration"] and "y" in self.state["exploration"]["currentChunk"] else 0
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
            interfaces = self.state["interfaces"] if "interfaces" in self.state else []
            
            # Find interfaces with available options
            clickable_interfaces = [
                interface for interface in interfaces
                if "options" in interface and interface["options"]
            ]
            
            if clickable_interfaces:
                # Select the first option from the first interface
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
            
        elif action == Action.TOGGLE_RUN:
            current_time = time.time()
            if current_time - self.last_run_toggle < self.run_toggle_cooldown:
                return None  # Still on cooldown
                
            player = self.state["player"] if "player" in self.state else {}
            run_energy = player["runEnergy"] if "runEnergy" in player else 0.0
            is_running = player["isRunning"] if "isRunning" in player else False
            
            # Only toggle if it makes sense energy-wise
            if (is_running and run_energy < 5.0) or (not is_running and run_energy > 20.0):
                self.last_run_toggle = current_time
                return {
                    "action": "interfaceAction",
                    "data": {
                        "interfaceId": 10485787,  # MINIMAP_TOGGLE_RUN_ORB ID
                        "groupId": 160,           # Group ID from logs
                        "optionText": "Toggle Run"
                    }
                }
        
        return None 

def get_tensorboard_dir() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    tb_log_dir = os.path.join(base_dir, "tb_logs")
    return tb_log_dir

def make_env(task: str = "combat") -> gym.Env:
    """Create and wrap the RuneScape environment."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    env = RuneScapeEnv(task=task)
    # Type cast to satisfy mypy
    return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"), allow_early_resets=True)

def train_combat_bot(total_timesteps: int = 1_000_000, checkpoint: Optional[str] = None) -> None:
    """Train the combat bot using PPO."""
    console.print("[bold]Initializing training environment...[/bold]")
    
    # Create the environment
    env = DummyVecEnv([lambda: make_env()])
    
    # Set up model
    console.print("[bold]Setting up PPO model...[/bold]")
    try:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            tensorboard_log=get_tensorboard_dir(),
        )
        
        # Set up checkpointing
        checkpoint_dir = os.path.join(get_tensorboard_dir(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=checkpoint_dir,
            name_prefix="combat_bot",
            verbose=1
        )
        
        # Train the model
        console.print("[bold green]Starting training...[/bold green]")
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save the final model
        model.save(os.path.join(get_tensorboard_dir(), "final_model"))
        console.print("[bold green]Training completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during training: {str(e)}[/bold red]")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RuneScape AI Training Bot")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total timesteps to train for")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.INFO)
        console.print("[bold green]Verbose logging enabled[/bold green]")
    
    console.print("[bold blue]RuneScape AI Training Bot[/bold blue]")
    console.print("Make sure RuneLite is running with the RLBot plugin enabled")
    
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            console.print(f"[green]Resuming from checkpoint: {args.checkpoint}[/green]")
        else:
            console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
            console.print("Starting fresh training run...")
    
    console.print("Waiting for WebSocket connection...\n")
    
    try:
        # Create the logging directory for TensorBoard if it doesn't exist
        tensorboard_dir = get_tensorboard_dir()
        os.makedirs(tensorboard_dir, exist_ok=True)
        console.print(f"[green]TensorBoard logs will be saved to: {tensorboard_dir}[/green]")
        
        # Start the training process
        console.print("[yellow]Starting training process...[/yellow]")
        train_combat_bot(total_timesteps=args.timesteps, checkpoint=args.checkpoint)
        console.print("[green]Training completed successfully![/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error during training: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        console.print("[red]Training failed.[/red]") 