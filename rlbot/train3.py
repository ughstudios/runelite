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
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Dict, AsyncGenerator
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio  # type: ignore
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env

import traceback  # Add this import
import io
import uuid
from pathlib import Path
from pydantic import BaseModel, Field, validator
from stable_baselines3.common.evaluation import evaluate_policy

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

# Define MAX_NPCS constant for observation space
MAX_NPCS = 10  # Maximum number of NPCs to track in the observation space

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

MIN_ACTIONS_PER_MINUTE = 100

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

# Define Pydantic models based on the state schema
class Location(BaseModel):
    x: int
    y: int
    plane: int = 0

class Health(BaseModel):
    current: int
    maximum: int

class Skill(BaseModel):
    level: int
    realLevel: int = 0
    experience: int = 0

class Player(BaseModel):
    location: Location
    health: Health
    inCombat: bool = False
    isRunning: bool = False
    runEnergy: float = 0
    skills: Dict[str, Skill] = {}
    prayer: int = 0

class NPC(BaseModel):
    id: int
    name: str = ""
    combatLevel: int = 0
    location: Location
    health: Optional[Health] = None
    interacting: bool = False
    distance: float = 0

class GameObject(BaseModel):
    id: int
    name: str = ""
    location: Location
    actions: List[str] = []

class GroundItem(BaseModel):
    id: int
    name: str = ""
    quantity: int = 1
    location: Location

class InterfaceElement(BaseModel):
    id: int
    type: str = ""
    text: str = ""
    actions: List[str] = []

class Exploration(BaseModel):
    currentChunk: dict = Field(default_factory=dict)
    visitedChunks: int = 0

class GameState(BaseModel):
    player: Player
    npcs: List[NPC] = []
    objects: List[GameObject] = []
    groundItems: List[GroundItem] = []
    interfaces: List[InterfaceElement] = []
    interfacesOpen: bool = False
    pathObstructed: bool = False
    exploration: Optional[Exploration] = None
    screenshot: str = ""

# -----------------------------------------------------------------------------
# Action Definitions
# -----------------------------------------------------------------------------
class RuneScapeEnv(gym.Env):
    """Custom Environment for controlling a character in RuneScape."""

    def __init__(self, task: str = "combat", debug: bool = True):
        """Initialize the RuneScape environment."""
        # Configure logging
        self.debug = debug
        self.logger = logging.getLogger("RuneScapeEnv")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Ensure we have a handler to prevent "No handlers could be found" warnings
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize task
        self.task = task
        
        # Define observation and action spaces
        if task == "combat":
            # Combined observation space with both image and vector components
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                'vector': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(102,), dtype=np.float32)
            })
            # Action space includes movement, attack, interface actions
            self.action_space = gym.spaces.Discrete(len(Action))
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Initialize counters and state
        self.tracked_skills = ["attack", "strength", "defence", "ranged", "magic", "hitpoints", "prayer"]
        self.actions_per_minute: float = MIN_ACTIONS_PER_MINUTE
        self.action_times: List[float] = []
        self.skill_xp: Dict[str, int] = {skill: 0 for skill in self.tracked_skills}
        self.visited_areas: Set[Tuple[int, int]] = set()  # Initialize visited_areas
        self.last_position: Optional[Tuple[int, int]] = None
        self.timestep: int = 0
        self.current_target: Optional[str] = None
        self.is_in_combat: bool = False
        self.last_combat_time: float = 0
        self.player_health: int = 100
        self.max_player_health: int = 100
        self.initial_skill_xp: Dict[str, int] = {}
        self.max_steps: int = 2000  # Maximum number of steps per episode
        
        # Rate limiting for requests (100 per minute = 1 every 0.6 seconds)
        self.rate_limit: float = 0.6  # Minimum time between requests in seconds
        self.last_request_time: float = 0.0
        
        # Additional tracking variables
        self.num_timesteps: int = 0
        self.last_target_id: Optional[int] = None
        self.interfaces_open: bool = False
        self.path_obstructed: bool = False
        self.interface_open: bool = False
        self.consecutive_interface_frames: int = 0
        self.interface_check_cooldown: float = 0.0
        self.run_toggle_cooldown: float = 30.0  # Cooldown for run toggle (in seconds)
        self.last_run_toggle: float = 0.0
        self.writer: Optional[SummaryWriter] = None
        
        # WebSocket connection setup
        self.websocket_url = "ws://localhost:43595"  # URL to connect to the RuneLite plugin
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.connection_event = threading.Event()
        
        # Add synchronization and message handling with proper types
        self.ws_lock: asyncio.Lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[Union[str, bytes]] = asyncio.Queue()
        self.response_futures: Dict[int, asyncio.Future[str]] = {}
        self.next_request_id: int = 0
        self.state_updates: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
        # State cache and updates
        self.state: Optional[Dict[str, Any]] = None
        self.last_state_update: float = 0.0
        self.state_update_interval: float = 0.1  # Min seconds between state updates
        
        # Load state schema for validation
        schema_path = os.path.join(os.path.dirname(__file__), "state_schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                self.state_schema = json.load(f)
                if self.debug:
                    self.logger.info(f"Loaded state schema from {schema_path}")
        else:
            self.logger.warning("State schema file not found, validation will be skipped")
            self.state_schema = {}
        
        # Asyncio setup
        self.loop = asyncio.new_event_loop()
        nest_asyncio.apply(self.loop)
        
        # Start websocket client
        if self.debug:
            self.logger.info("Starting WebSocket client thread")
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()
        
        # Wait for initial connection
        self.logger.info("Waiting for WebSocket connection...")
        if not self.connection_event.wait(30):
            self.logger.error("Failed to connect within 30 seconds")
            self.logger.error("Is the RuneLite client running with the RLBot plugin?")
            raise RuntimeError("Failed to connect to RuneLite. Please ensure the client is running with the RLBot plugin.")
        
        # Attempt to get initial state
        if self.debug:
            self.logger.info("Attempting to get initial state...")
        self._refresh_state()
        if self.state is None:
            self.logger.error("Failed to get initial state. Is the player logged in?")
        elif self.debug:
            self.logger.info("Initial state retrieved successfully")

        # Add visited areas tracker
        self.visited_areas = set()

    def _run_websocket_loop(self) -> None:
        """Run the websocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_client())

    def _calculate_combat_level(self, skills: Dict[str, Any]) -> int:
        """
        Calculate the player's combat level based on their skills.
        
        Args:
            skills: Dictionary of skill levels with keys as skill names
            
        Returns:
            int: The calculated combat level
        """
        # Extract skill levels from skill objects
        def get_skill_level(skill_name: str) -> int:
            skill_data = skills.get(skill_name, {})
            if isinstance(skill_data, dict):
                return skill_data.get("level", 1)
            elif isinstance(skill_data, (int, float)):
                return int(skill_data)
            return 1  # Default level
        
        # Get base skill levels
        attack = get_skill_level("attack")
        strength = get_skill_level("strength")
        defence = get_skill_level("defence")
        hitpoints = get_skill_level("hitpoints")
        prayer = get_skill_level("prayer")
        ranged = get_skill_level("ranged")
        magic = get_skill_level("magic")
        
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
        # Establish a single persistent websocket connection
        async with websockets.connect(
            self.websocket_url, 
            ping_interval=None,  # Disable automatic pings
            ping_timeout=None,   # Disable ping timeout
            close_timeout=5      # Shorter close timeout
        ) as websocket:
            self.ws = websocket
            self.connected = True
            self.connection_event.set()
            self.logger.info("WebSocket connection established")
            
            # Start message processor
            message_processor = asyncio.create_task(self._process_messages())
            
            # Start receiving messages
            while True:
                # Receive each message and add it to the queue
                message = await websocket.recv()
                # Log the message properly based on its type
                if isinstance(message, bytes):
                    self.logger.debug(f"Received binary message of {len(message)} bytes")
                else:
                    self.logger.debug(f"Received text message: {message[:100]}...")
                await self.message_queue.put(message)
    
    async def _process_messages(self) -> None:
        """Process incoming messages from the websocket."""
        while True:
            # Get the next message from the queue
            message = await self.message_queue.get()
            
            # Decode binary messages
            if isinstance(message, bytes):
                self.logger.debug(f"Received binary message of {len(message)} bytes")
                message_str = message.decode('utf-8', errors='replace')
            else:
                message_str = message
            
            # Parse the message
            data = self._parse_json(message_str)
            if data is None:
                self.logger.warning("Received invalid JSON message")
                self.message_queue.task_done()
                continue  # Skip invalid JSON
                
            # Handle message based on its content
            self._handle_message(data, message_str)
            
            # Mark this message as processed
            self.message_queue.task_done()
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON safely without exceptions."""
        if not text:
            return None
            
        # Parse JSON without try/except
        result = json.loads(text)
        if not isinstance(result, dict):
            self.logger.warning(f"Received non-dict data: {result}")
            return None
            
        return result
    
    def _handle_message(self, data: Dict[str, Any], raw_message: str) -> None:
        """Route the message to the appropriate handler based on its content."""
        # Check if this is a response to a specific request
        request_id = data.get("request_id")
        if request_id is not None and request_id in self.response_futures:
            # This is a response to a specific request
            future = self.response_futures.pop(request_id)
            if not future.done():
                future.set_result(raw_message)
            return
                
        # Handle state updates or other general messages
        if "type" in data and data["type"] == "screenshot" and self.state is not None:
            self.state["screenshot"] = data.get("data")
            self.logger.debug("Updated screenshot data")
        elif "error" in data:
            self.logger.error(f"Received error from server: {data['error']}")
        elif self._is_valid_state(data):
            # Update the game state and timestamp
            self.state = data
            self.last_state_update = time.time()
            self.interfaces_open = data.get("interfacesOpen", False)
            self.path_obstructed = data.get("pathObstructed", False)
            self.logger.debug("Received valid state update")
            
            # Update position and other tracking variables if needed
            if "player" in data:
                player = data["player"]
                
                # Get player location/position
                location = player.get("location", {})
                if not location and "position" in player:
                    location = player.get("position", {})
                
                self.last_position = (location.get("x", 0), location.get("y", 0))
                
                # Track player health
                self.player_health = player.get("health", 0)
                
                # Store other state flags
                self.interfaces_open = data.get("interfacesOpen", False)
                self.path_obstructed = data.get("pathObstructed", False)
            
            # Also put it in the state updates queue
            asyncio.run_coroutine_threadsafe(
                self.state_updates.put(data),
                self.loop
            )
        else:
            self.logger.debug(f"Received unhandled message type: {data.get('type', 'unknown')}")
    
    def _is_valid_state(self, data: Dict[str, Any]) -> bool:
        """Check if the data is a valid game state."""
        required_keys = ["player", "npcs", "objects", "groundItems"]
        if not all(key in data for key in required_keys):
            missing = [key for key in required_keys if key not in data]
            self.logger.warning(f"Received incomplete state. Missing fields: {missing}")
            return False
            
        if not isinstance(data["player"], dict):
            self.logger.warning("Player data is not a dictionary")
            return False
            
        return True

    def _refresh_state(self) -> bool:
        """Refresh the state from the server if needed.
        
        Returns:
            bool: True if the state was refreshed, False otherwise
        """
        # If no state or state is older than refresh interval, get a new one
        current_time = time.time()
        if self.state is None or (current_time - self.last_state_update) >= self.state_update_interval:
            # If we just started up, wait to ensure WebSocket connection is established
            if not self.connected or not self.ws:
                # Wait for connection if it's still establishing
                if not self.connection_event.is_set():
                    self.logger.info("Waiting for WebSocket connection...")
                    # Check for connection
                    if not self.connection_event.wait(timeout=2.0):
                        self.logger.warning("WebSocket connection timeout")
                        return False
                else:
                    # Connection event is set but we're not connected? Something's wrong
                    self.logger.warning("Connection event is set but WebSocket is not connected")
                    return False
            
            # Now we can get the state
            new_state = None
            
            # Use the _send_command method directly which is more reliable
            command = {"type": "get_state"}
            sent = asyncio.run_coroutine_threadsafe(
                self._send_command(command),
                self.loop
            )
            
            # Wait for the command to be sent
            sent.result(timeout=1.0)
            
            # The state should be updated by the message handler
            # Check if we have a state now
            if self.state is not None:
                # Update timestamp
                self.last_state_update = current_time
                
                # Update tracking variables
                if "player" in self.state:
                    player = self.state["player"]
                    
                    # Get player location/position
                    location = {}
                    if "location" in player:
                        location = player["location"]
                    elif "position" in player:
                        location = player["position"]
                    
                    if "x" in location and "y" in location:
                        self.last_position = (location["x"], location["y"])
                    
                    # Track player health
                    if "health" in player:
                        self.player_health = player["health"]
                    
                    # Store other state flags
                    if "interfacesOpen" in self.state:
                        self.interfaces_open = self.state["interfacesOpen"]
                    if "pathObstructed" in self.state:
                        self.path_obstructed = self.state["pathObstructed"]
                
                return True
            else:
                self.logger.warning("No state received after request")
                return False
        
        # State is recent enough, no need to refresh
        return False

    def _get_state(self) -> Optional[Dict[str, Any]]:
        """Get the current game state, using cached state if recent enough.
        
        Returns:
            The game state as a dictionary, or None if unavailable
        """
        # Try to refresh state if needed
        self._refresh_state()
        
        # Return the state (cached or refreshed)
        return self.state

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

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to an initial state."""
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.action_space.seed(seed)
            
        # Clear state
        self.timestep = 0
        self.visited_areas.clear()
        
        # Force state refresh for reset
        self._refresh_state()
        
        # Reset player state
        if self.state is not None and "player" in self.state:
            player = self.state["player"]
            self.last_position = (player.get("x", 0), player.get("y", 0))
            self.player_health = player.get("health", 100)
            self.max_player_health = player.get("maxHealth", 100)
            
            # Record initial XP
            for skill, xp in player.get("skills", {}).items():
                if skill in self.tracked_skills:
                    self.skill_xp[skill] = xp
                    self.initial_skill_xp[skill] = xp
        else:
            # No valid state available
            self.logger.warning("Reset: Could not get valid state")
            # Provide a default observation with empty image and vector
            image_obs = np.zeros((84, 84, 3), dtype=np.uint8)
            vector_obs = np.zeros(102, dtype=np.float32)
            return {'image': image_obs, 'vector': vector_obs}, {}
        
        # Get initial observation
        observation = self._state_to_observation()
        
        # Return initial observation and empty info dict
        return observation, {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute action and advance environment by one step."""
        # Validate action
        if not self.action_space.contains(action):
            self.logger.warning(f"Invalid action: {action}")
            action = 0  # Default to WAIT action
        
        # Take the action
        cmd = self._action_to_command(Action(action))
        
        # Default values in case of failure
        old_state = self.state
        
        # Send the command if valid
        if cmd is not None:
            # Run command in the event loop
            fut = asyncio.run_coroutine_threadsafe(
                self._send_command(cmd),
                self.loop
            )
            # Wait for the result
            fut.result(timeout=2.0)
        
        # Get the new state after action
        self._refresh_state()
        
        # Handle case where we couldn't get state
        if self.state is None:
            self.logger.warning("Failed to get state after action")
            # Use old state if available
            if old_state is not None:
                self.state = old_state
            else:
                # Create empty observation with blank image and vector
                image_obs = np.zeros((84, 84, 3), dtype=np.uint8)
                vector_obs = np.zeros(102, dtype=np.float32)
                return {'image': image_obs, 'vector': vector_obs}, 0.0, False, True, {}
        
        # Update state
        self.state = self.state
        
        # Increment timestep
        self.timestep += 1
        self.num_timesteps += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Prepare observation
        observation = self._state_to_observation()
        
        # Determine if episode is done
        terminated = False
        truncated = self.timestep >= self.max_steps
        
        # Prepare info dict with diagnostics
        info = {
            "timestep": self.timestep,
            "health": self.player_health,
            "max_health": self.max_player_health,
            "action": action,
            "position": self.last_position,
            "visited_areas": len(self.visited_areas),
        }
        
        # Log metrics for this step
        if self.debug and self.timestep % 10 == 0:
            self.logger.debug(f"Step {self.timestep}: Action={Action(action).name}, Reward={reward:.4f}")
        
        return observation, reward, terminated, truncated, info
    
    def _state_to_observation(self) -> Dict[str, np.ndarray]:
        """Convert state to the observation format expected by the agent."""
        # Initialize vector observation with zeros
        vector_obs = np.zeros(102, dtype=np.float32)
        
        # Initialize image observation with zeros
        image_obs = np.zeros((84, 84, 3), dtype=np.uint8)
        
        # Return default observation if no state is available
        if self.state is None:
            if self.debug:
                self.logger.warning(f"No state available at timestep {self.timestep}, returning zero observation")
            return {'image': image_obs, 'vector': vector_obs}
        
        # First attempt to parse with Pydantic model
        game_state_valid = False
        game_state = None
        
        # Try to parse the state using our Pydantic model
        if self._is_valid_state(self.state):
            game_state = GameState.model_validate(self.state)
            game_state_valid = True
            
            # Process screenshot if available
            if game_state.screenshot:
                # Decode base64 screenshot
                image_data = base64.b64decode(game_state.screenshot)
                pil_image = Image.open(BytesIO(image_data))
                # Resize to 84x84
                resized_image = pil_image.resize((84, 84))
                # Convert to numpy array
                image_obs = np.array(resized_image, dtype=np.uint8)
            
            # Extract vector features
            features = {}
            
            # Extract player information
            player_location = Location(x=0, y=0, plane=0)
            if game_state.player and game_state.player.location:
                player_location = game_state.player.location
            
            # Player position
            features["player_x"] = float(player_location.x)
            features["player_y"] = float(player_location.y)
            
            # Player health (normalized)
            current_health: float = 0.0
            max_health: float = 1.0  # Default to avoid division by zero
            
            if game_state.player and game_state.player.health:
                player_health = game_state.player.health
                current_health = float(player_health.current)
                max_health = float(player_health.maximum)
            
            features["health"] = current_health / max(1.0, max_health)  # Normalize
            
            # Player combat status
            in_combat: bool = False
            if game_state.player:
                in_combat = game_state.player.inCombat
            
            features["in_combat"] = 1.0 if in_combat else 0.0
            
            # Player run energy
            run_energy: float = 0.0
            if game_state.player:
                run_energy = float(game_state.player.runEnergy)
            
            features["run_energy"] = run_energy / 100.0  # Normalize
            
            # Is player running
            is_running: bool = False
            if game_state.player:
                is_running = game_state.player.isRunning
            
            features["is_running"] = 1.0 if is_running else 0.0
            
            # Player skill levels
            skills: Dict[str, Skill] = {}
            if game_state.player and game_state.player.skills:
                skills = game_state.player.skills
            
            for i, skill_name in enumerate(self.tracked_skills):
                if i < 7:  # Ensure we stay within observation space bounds
                    skill_value: float = 1.0
                    if skill_name in skills:
                        skill_data = skills[skill_name]
                        if isinstance(skill_data, Skill) and hasattr(skill_data, "level"):
                            skill_value = float(skill_data.level)
                        elif isinstance(skill_data, dict) and "level" in skill_data:
                            skill_value = float(skill_data["level"])
                        elif isinstance(skill_data, (int, float)):
                            skill_value = float(skill_data)
                    features[f"{skill_name}_level"] = skill_value
            
            # Nearby NPCs (up to 5 closest)
            npcs_list: List[NPC] = []
            if self.state is not None and "npcs" in self.state:
                npcs_list = self.state["npcs"]
            
            for i in range(min(5, len(npcs_list))):
                current_npc: NPC = npcs_list[i]
                prefix = f"npc{i+1}_"
                
                # NPC position relative to player
                npc_location = Location(x=0, y=0)
                if hasattr(current_npc, "location"):
                    npc_location = current_npc.location
                
                npc_x = float(npc_location.x) if hasattr(npc_location, "x") else 0.0
                npc_y = float(npc_location.y) if hasattr(npc_location, "y") else 0.0
                rel_x = npc_x - features["player_x"]
                rel_y = npc_y - features["player_y"]
                
                features[f"{prefix}rel_x"] = rel_x
                features[f"{prefix}rel_y"] = rel_y
                
                distance = math.sqrt(rel_x**2 + rel_y**2)
                if hasattr(current_npc, "distance"):
                    distance = float(current_npc.distance)
                features[f"{prefix}distance"] = distance
                
                features[f"{prefix}in_combat"] = 1.0 if hasattr(current_npc, "interacting") and current_npc.interacting else 0.0
                
                # NPC health if available
                npc_health_percent = 1.0
                if hasattr(current_npc, "health") and current_npc.health is not None:
                    npc_health = current_npc.health
                    current = float(npc_health.current) if hasattr(npc_health, "current") else 0.0
                    maximum = float(npc_health.maximum) if hasattr(npc_health, "maximum") else 1.0
                    npc_health_percent = current / max(1, maximum)
                features[f"{prefix}health"] = npc_health_percent
                
                # NPC combat level
                combat_level = 0.0
                if hasattr(current_npc, "combatLevel"):
                    combat_level = float(current_npc.combatLevel)
                features[f"{prefix}level"] = combat_level / 100.0  # Normalize
        
        # If Pydantic model failed, fallback to manual parsing
        if not game_state_valid:
            self.logger.warning("Failed to parse state with Pydantic. Falling back to manual parsing.")
            
            # Extract values safely from the state
            # Get player data
            player = {}
            if "player" in self.state:
                player = self.state["player"]
            
            features = {}
            
            # Extract player information
            player_loc = {}
            if "location" in player:
                player_loc = player["location"]
            elif "position" in player:
                player_loc = player["position"]
            
            # Player position
            features["player_x"] = float(player_loc.get("x", 0))
            features["player_y"] = float(player_loc.get("y", 0))
            
            # Player health (normalized)
            curr_health: float = 0.0
            max_hp: float = 1.0  # Default to avoid division by zero
            
            if "health" in player:
                health_data = player["health"]
                if isinstance(health_data, dict):
                    curr_health = float(health_data.get("current", 0))
                    max_hp = float(health_data.get("maximum", 100))
            
            features["health"] = curr_health / max(1.0, max_hp)  # Normalize
            
            # Player combat status
            is_combat: bool = False
            if "inCombat" in player:
                is_combat = player["inCombat"]
            
            features["in_combat"] = 1.0 if is_combat else 0.0
            
            # Player run energy
            energy: float = 0.0
            if "runEnergy" in player:
                energy = float(player["runEnergy"])
            
            features["run_energy"] = energy / 100.0  # Normalize
            
            # Is player running
            running: bool = False
            if "isRunning" in player:
                running = player["isRunning"]
            
            features["is_running"] = 1.0 if running else 0.0
            
            # Player skill levels
            player_skills = {}
            if "skills" in player:
                player_skills = player["skills"]
            
            for i, skill_name in enumerate(self.tracked_skills):
                if i < 7:  # Ensure we stay within observation space bounds
                    skill_val: float = 1.0
                    if skill_name in player_skills:
                        skill_data = player_skills[skill_name]
                        if isinstance(skill_data, Skill) and hasattr(skill_data, "level"):
                            skill_val = float(skill_data.level)
                        elif isinstance(skill_data, dict) and "level" in skill_data:
                            skill_val = float(skill_data["level"])
                        elif isinstance(skill_data, (int, float)):
                            skill_val = float(skill_data)
                    features[f"{skill_name}_level"] = skill_val
            
            # Nearby NPCs (up to 5 closest)
            npcs_list: List[NPC] = []
            if self.state is not None and "npcs" in self.state:
                npcs_list = self.state["npcs"]
            
            for i in range(min(5, len(npcs_list))):
                current_npc: NPC = npcs_list[i]
                prefix = f"npc{i+1}_"
                
                # NPC position relative to player
                npc_location = Location(x=0, y=0)
                if hasattr(current_npc, "location"):
                    npc_location = current_npc.location
                
                npc_x = float(npc_location.x) if hasattr(npc_location, "x") else 0.0
                npc_y = float(npc_location.y) if hasattr(npc_location, "y") else 0.0
                rel_x = npc_x - features["player_x"]
                rel_y = npc_y - features["player_y"]
                
                features[f"{prefix}rel_x"] = rel_x
                features[f"{prefix}rel_y"] = rel_y
                
                distance = math.sqrt(rel_x**2 + rel_y**2)
                if hasattr(current_npc, "distance"):
                    distance = float(current_npc.distance)
                features[f"{prefix}distance"] = distance
                
                features[f"{prefix}in_combat"] = 1.0 if hasattr(current_npc, "interacting") and current_npc.interacting else 0.0
                
                # NPC health if available
                npc_health_percent = 1.0
                if hasattr(current_npc, "health") and current_npc.health is not None:
                    npc_health = current_npc.health
                    current = float(npc_health.current) if hasattr(npc_health, "current") else 0.0
                    maximum = float(npc_health.maximum) if hasattr(npc_health, "maximum") else 1.0
                    npc_health_percent = current / max(1, maximum)
                features[f"{prefix}health"] = npc_health_percent
                
                # NPC combat level
                combat_level = 0.0
                if hasattr(current_npc, "combatLevel"):
                    combat_level = float(current_npc.combatLevel)
                features[f"{prefix}level"] = combat_level / 100.0  # Normalize
        
        # Convert features to numpy array in fixed order
        feature_keys = sorted(features.keys())
        for i, key in enumerate(feature_keys):
            if i < vector_obs.shape[0]:  # Avoid index out of bounds
                vector_obs[i] = features[key]
        
        # Return both image and vector observations
        return {'image': image_obs, 'vector': vector_obs}
        
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        # Initialize reward
        reward = 0.0

        # Check if we have a previous or current state
        if self.state is None or self.last_position is None:
            self.logger.warning("Cannot calculate reward: No state or last position")
            return 0.0
        
        # Get the current player state
        player = {}
        if "player" in self.state:
            player = self.state["player"]
        else:
            self.logger.warning("Player data missing in state")
            return 0.0

        # Health change reward
        current_health: int = 0
        max_health: int = 100
        if "health" in player:
            health = player["health"]
            if isinstance(health, dict):
                if "current" in health:
                    current_health = int(health["current"])
                if "maximum" in health:
                    max_health = int(health["maximum"])
            elif isinstance(health, (int, float)):
                current_health = int(health)
                if "maxHealth" in player:
                    max_health = int(player["maxHealth"])
        
        # Calculate health difference based on tracked health value
        prev_health = 0
        if isinstance(self.player_health, dict):
            if hasattr(self.player_health, "get") and callable(getattr(self.player_health, "get")):
                if "current" in self.player_health:
                    prev_health = self.player_health["current"]
        elif isinstance(self.player_health, (int, float)):
            prev_health = self.player_health
        
        health_diff = current_health - prev_health
        
        # Update health tracking
        if health_diff != 0:
            # Negative reward for taking damage (-1 for each hp lost)
            if health_diff < 0:
                health_reward = health_diff * 0.5  # Penalty for taking damage
                self.logger.debug(f"Health decreased by {-health_diff}, reward: {health_reward}")
                reward += health_reward
            # Small positive reward for healing
            elif health_diff > 0:
                health_reward = health_diff * 0.1  # Small bonus for healing
                self.logger.debug(f"Health increased by {health_diff}, reward: {health_reward}")
                reward += health_reward
                
            # Update health tracking
            if "health" in player:
                self.player_health = player["health"]  # Store the entire health object/value
        
        # Track if player died and revived
        # Get previous health value for comparison
        prev_health_value = 0
        if isinstance(self.player_health, dict):
            if "current" in self.player_health:
                prev_health_value = self.player_health["current"]
        elif isinstance(self.player_health, (int, float)):
            prev_health_value = self.player_health
        
        if prev_health_value <= 0 and current_health > 0:
            self.logger.info("Player died and respawned")
            reward -= 50.0  # Large penalty for dying
        
        # Experience gains reward
        skills = {}
        if "skills" in player:
            skills = player["skills"]
        
        for skill_name, skill_data in skills.items():
            if skill_name not in self.tracked_skills:
                continue
            
            # Extract experience value from skill data
            current_xp: int = 0
            if isinstance(skill_data, dict):
                if "experience" in skill_data:
                    current_xp = int(skill_data["experience"])
            elif isinstance(skill_data, (int, float)):
                current_xp = int(skill_data)
            
            # Calculate XP gained since last check
            previous_xp = 0
            if skill_name in self.skill_xp:
                previous_xp = self.skill_xp[skill_name]
            
            # Ensure both current_xp and previous_xp are integers before subtraction
            if isinstance(current_xp, int) and isinstance(previous_xp, int):
                xp_gained = current_xp - previous_xp
            elif isinstance(current_xp, dict) and "experience" in current_xp:
                # If current_xp is a dictionary, extract the experience value
                current_xp_value = int(current_xp["experience"])
                xp_gained = current_xp_value - previous_xp
            else:
                # If we can't handle the type, assume no XP gain
                xp_gained = 0
            
            if xp_gained > 0:
                # Significant positive reward for gaining XP
                xp_reward = xp_gained * 0.01
                self.logger.debug(f"Gained {xp_gained} XP in {skill_name}, reward: {xp_reward}")
                reward += xp_reward
                
                # Update tracked XP - store as integer value
                if isinstance(current_xp, int):
                    self.skill_xp[skill_name] = current_xp
                elif isinstance(current_xp, dict) and "experience" in current_xp:
                    self.skill_xp[skill_name] = int(current_xp["experience"])
        
        # Movement and exploration reward
        location = {}
        if "location" in player:
            location = player["location"]
        elif "position" in player:
            location = player["position"]
        
        current_x = 0
        current_y = 0
        if "x" in location:
            current_x = location["x"]
        if "y" in location:
            current_y = location["y"]
        
        current_position = (current_x, current_y)
        if self.last_position is not None and current_position != self.last_position:
            # Small reward for moving
            movement_reward = 0.01
            self.logger.debug(f"Moved from {self.last_position} to {current_position}, reward: {movement_reward}")
            reward += movement_reward
            
        # Track position
        self.last_position = current_position
            
        # Check if we've visited this area before
        region_x = current_position[0] // 10
        region_y = current_position[1] // 10
        region = (region_x, region_y)
        
        # Exploration reward
        if region not in self.visited_areas:
            # Significant reward for exploring new areas
            exploration_reward = 1.0
            self.logger.debug(f"Explored new region {region}, reward: {exploration_reward}")
            reward += exploration_reward
            
            # Track visited areas
            self.visited_areas.add(region)
        
        # Combat reward
        is_in_combat = False
        if "inCombat" in player:
            is_in_combat = player["inCombat"]
        
        # Track time spent in combat
        if is_in_combat:
            # Small reward for being in combat
            combat_reward = 0.1
            self.logger.debug(f"In combat, reward: {combat_reward}")
            reward += combat_reward
            
            # Track combat time
            self.last_combat_time = time.time()
        
        # Return the summed reward
        return reward

    async def _send_command(self, command: Dict[str, Any]) -> bool:
        """Send a command to the RuneLite plugin via WebSocket."""
        if not self.ws or not self.connected:
            self.logger.warning("Cannot send command: WebSocket not connected")
            return False
        
        # Apply rate limiting - wait if we're sending too quickly
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit:
            # Sleep to enforce rate limit
            sleep_time = self.rate_limit - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Serialize the command
        command_json = json.dumps(command)
        
        # Send with the lock to prevent concurrent sends
        async with self.ws_lock:
            await self.ws.send(command_json)
        return True

    async def _request(self, command: str) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a response."""
        if not self.ws or not self.connected:
            self.logger.warning(f"Cannot send request: WebSocket not connected")
            return None
        
        # Apply rate limiting - wait if we're sending too quickly
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit:
            # Sleep to enforce rate limit
            sleep_time = self.rate_limit - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Create a unique request ID
        request_id = self.next_request_id
        self.next_request_id += 1
        
        # Create a future to receive the response
        future = self.loop.create_future()
        self.response_futures[request_id] = future
        
        # Prepare the request with an ID for tracking
        request = {
            "request_id": request_id,
            "command": command
        }
        request_json = json.dumps(request)
        
        # Send the request with the lock to prevent concurrent sends
        async with self.ws_lock:
            await self.ws.send(request_json)
        
        # Wait for the response with a timeout
        return await asyncio.wait_for(future, timeout=2.0)

    def _extract_features(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract features from game state for the observation space."""
        # Create empty vector observation array
        vector_obs = np.zeros(shape=(102,), dtype=np.float32)
        
        # Convert raw screenshot to image observation
        image_obs = np.zeros(shape=(84, 84, 3), dtype=np.uint8)
        
        if "screenshot" in state and state["screenshot"]:
            # Process screenshot to usable size
            image_data = base64.b64decode(state["screenshot"])
            pil_image = Image.open(BytesIO(image_data))
            resized_image = pil_image.resize((84, 84))
            image_obs = np.array(resized_image, dtype=np.uint8)
        
        # Create feature dictionary
        features = {}
        
        # Extract game state information
        # Parse state into a GameState object if it's not already
        game_state: GameState
        if isinstance(state, dict):
            try:
                # Use model_validate instead of deprecated parse_obj
                game_state = GameState.model_validate(state)
            except Exception as e:
                self.logger.error(f"Error parsing game state: {e}")
                # Return empty observations if we can't parse the state
                return {'image': image_obs, 'vector': vector_obs}
        else:
            game_state = state
        
        # Extract player information
        player_location = Location(x=0, y=0, plane=0)
        if game_state.player and game_state.player.location:
            player_location = game_state.player.location
        
        # Player position
        features["player_x"] = float(player_location.x)
        features["player_y"] = float(player_location.y)
        
        # Player health (normalized)
        current_health: float = 0.0
        max_health: float = 1.0  # Default to avoid division by zero
        
        if game_state.player and game_state.player.health:
            player_health = game_state.player.health
            current_health = float(player_health.current)
            max_health = float(player_health.maximum)
        
        features["health"] = current_health / max(1.0, max_health)  # Normalize
        
        # Player combat status
        in_combat: bool = False
        if game_state.player:
            in_combat = game_state.player.inCombat
        
        features["in_combat"] = 1.0 if in_combat else 0.0
        
        # Player run energy
        run_energy: float = 0.0
        if game_state.player:
            run_energy = float(game_state.player.runEnergy)
        
        features["run_energy"] = run_energy / 100.0  # Normalize
        
        # Is player running
        is_running: bool = False
        if game_state.player:
            is_running = game_state.player.isRunning
        
        features["is_running"] = 1.0 if is_running else 0.0
        
        # Player skill levels
        skills: Dict[str, Skill] = {}
        if game_state.player and game_state.player.skills:
            skills = game_state.player.skills
        
        for i, skill_name in enumerate(self.tracked_skills):
            if i < 7:  # Ensure we stay within observation space bounds
                skill_value: float = 1.0
                if skill_name in skills:
                    skill_data = skills[skill_name]
                    if isinstance(skill_data, Skill) and hasattr(skill_data, "level"):
                        skill_value = float(skill_data.level)
                    elif isinstance(skill_data, dict) and "level" in skill_data:
                        skill_value = float(skill_data["level"])
                    elif isinstance(skill_data, (int, float)):
                        skill_value = float(skill_data)
                features[f"{skill_name}_level"] = skill_value
        
        # Nearby NPCs (up to 5 closest)
        npcs: List[NPC] = []
        if "npcs" in self.state:
            npcs = self.state["npcs"]
        
        for i in range(min(5, len(npcs))):
            current_npc: NPC = npcs[i]
            prefix = f"npc{i+1}_"
            
            # NPC position relative to player
            npc_location = Location(x=0, y=0)
            if hasattr(current_npc, "location"):
                npc_location = current_npc.location
            
            npc_x = float(npc_location.x) if hasattr(npc_location, "x") else 0.0
            npc_y = float(npc_location.y) if hasattr(npc_location, "y") else 0.0
            rel_x = npc_x - features["player_x"]
            rel_y = npc_y - features["player_y"]
            
            features[f"{prefix}rel_x"] = rel_x
            features[f"{prefix}rel_y"] = rel_y
            
            distance = math.sqrt(rel_x**2 + rel_y**2)
            if hasattr(current_npc, "distance"):
                distance = float(current_npc.distance)
            features[f"{prefix}distance"] = distance
            
            features[f"{prefix}in_combat"] = 1.0 if hasattr(current_npc, "interacting") and current_npc.interacting else 0.0
            
            # NPC health if available
            npc_health_percent = 1.0
            if hasattr(current_npc, "health") and current_npc.health is not None:
                npc_health = current_npc.health
                current = float(npc_health.current) if hasattr(npc_health, "current") else 0.0
                maximum = float(npc_health.maximum) if hasattr(npc_health, "maximum") else 1.0
                npc_health_percent = current / max(1, maximum)
            features[f"{prefix}health"] = npc_health_percent
            
            # NPC combat level
            combat_level = 0.0
            if hasattr(current_npc, "combatLevel"):
                combat_level = float(current_npc.combatLevel)
            features[f"{prefix}level"] = combat_level / 100.0  # Normalize
        
        # Convert features to numpy array in fixed order
        feature_keys = sorted(features.keys())
        for i, key in enumerate(feature_keys):
            if i < vector_obs.shape[0]:  # Avoid index out of bounds
                vector_obs[i] = features[key]
        
        # Return both image and vector observations
        return {'image': image_obs, 'vector': vector_obs}

def get_tensorboard_dir() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    tb_log_dir = os.path.join(base_dir, "tb_logs")
    return tb_log_dir

def make_env(task: str = "combat", debug: bool = False, seed: Optional[int] = None) -> Callable[[], gym.Env]:
    """Factory function to create a RuneScape environment.
    
    Args:
        task: The task to perform ("combat", "fishing", etc.)
        debug: Whether to enable debug logging
        seed: Optional random seed
        
    Returns:
        A function that creates and returns a RuneScape environment
    """
    def _init() -> gym.Env:
        # Create the environment without retries
        env = RuneScapeEnv(task=task, debug=debug)
        
        # Set the seed if provided
        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset()
            
        return env
    
    return _init

def create_ppo_agent(env, model_dir: str, log_dir: Optional[str] = None, **kwargs):
    """Create a PPO agent with the specified parameters."""
    
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
    
    # Create evaluation callback
    eval_env = None
    eval_callback = None
    if kwargs.get("evaluate", False):
        # Create separate environment for evaluation
        eval_env = make_vec_env(
            lambda: RuneScapeEnv(task="combat", debug=False),
            n_envs=1,
            seed=kwargs.get("seed", 0),
            vec_env_cls=DummyVecEnv,
            monitor_dir=None if log_dir is None else os.path.join(log_dir, "eval")
        )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False
        )
    
    # Set up tensorboard logging if requested
    callbacks: List[BaseCallback] = []  # Use a more general type annotation
    callbacks.append(checkpoint_callback)
    if eval_callback is not None:
        callbacks.append(eval_callback)
    
    # Create the PPO agent with MultiInputPolicy for Dict observation spaces
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation spaces
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
    """Test the connection to RuneLite by creating an environment and getting initial state."""
    console = Console()
    console.print("[bold cyan]Testing connection to RuneLite[/bold cyan]")
    console.print("Initializing environment...")
    
    env = None
    
    # Initialize the environment
    env = RuneScapeEnv(task="combat", debug=debug)
    
    # Wait a bit for the connection to establish
    time.sleep(2.0)
    
    # No need to explicitly call _get_state as it was done during initialization
    if env.state is None:
        console.print("[bold red]Failed to get state from RuneLite[/bold red]")
        console.print("Please make sure:")
        console.print("1. RuneLite is running with the RLBot plugin")
        console.print("2. You are logged into the game")
    else:
        console.print("[bold green]Successfully connected to RuneLite![/bold green]")
        # Access player position correctly
        if "player" in env.state and "location" in env.state["player"]:
            player_x = env.state["player"]["location"]["x"]
            player_y = env.state["player"]["location"]["y"]
            console.print(f"Player position: ({player_x}, {player_y})")
        else:
            console.print("Player position: (N/A, N/A)")
    
    # Clean up
    if env is not None:
        env.close()

class CombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor that processes both image and vector observations."""
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract shapes from observation space
        image_shape = observation_space.spaces['image'].shape
        vector_shape = observation_space.spaces['vector'].shape
        
        # Calculate vector dimension safely
        vector_dim: int = 0
        if vector_shape is not None:
            if isinstance(vector_shape, tuple) and len(vector_shape) > 0:
                vector_dim = vector_shape[0]
        
        # Calculate image input channels safely
        channels: int = 3  # Default to 3 channels (RGB)
        if image_shape is not None:
            if isinstance(image_shape, tuple):
                if len(image_shape) == 3:
                    if image_shape[2] == 3:  # (H,W,C) format
                        channels = image_shape[2]
                    else:  # (C,H,W) format
                        channels = image_shape[0]
        
        # CNN for processing images
        cnn_output_dim = 128
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size by doing one forward pass
        with torch.no_grad():
            sample_img = torch.zeros(1, channels, 84, 84)
            n_flatten = self.cnn(sample_img).shape[1]
        
        # Linear layer after CNN
        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU(),
        )
        
        # MLP for processing vector data
        vector_output_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim, 256),  # Expanded first layer to handle larger input
            nn.ReLU(),
            nn.Linear(256, 128),         # Intermediate layer for better feature extraction
            nn.ReLU(),
            nn.Linear(128, vector_output_dim),
            nn.ReLU(),
        )
        
        # Final layer to combine both features
        self.final = nn.Linear(cnn_output_dim + vector_output_dim, features_dim)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process the observation dictionary."""
        # Process image
        # Ensure image is in right format [B, C, H, W] and normalized
        if observations['image'].shape[-1] == 3:  # If in format [B, H, W, C]
            img = observations['image'].permute(0, 3, 1, 2)
        else:
            img = observations['image']
        
        img = img.float() / 255.0  # Normalize to [0, 1]
        cnn_features = self.cnn_linear(self.cnn(img))
        
        # Process vector
        vector_features = self.mlp(observations['vector'])
        
        # Combine features
        combined = torch.cat([cnn_features, vector_features], dim=1)
        return self.final(combined)

def train_combat_bot(debug=False, verbose=False, timesteps=1000000):
    """Train a bot to fight NPCs."""
    # Create a structured logging path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./rlbot/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logging.basicConfig(
        level=logging.DEBUG if debug or verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"train_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("RuneScapeBotTrainer")
    
    # Let the user know what's happening
    print("Starting training session")
    
    # Create environment
    print("Creating environment")
    
    # Create and wrap environment - pass only the parameters that the constructor accepts
    env = RuneScapeEnv(task="combat", debug=debug)
    
    # Verify the environment is working by attempting to reset
    logger.info("Verifying environment initialization")
    
    # Reset the environment without try/except
    initial_obs = env.reset()[0]  # SB3 expects the new Gymnasium API
    
    # If we got here, the reset was successful
    logger.info("Initial state verification successful!")
    print("Initial state verification successful!")
    
    # Wrap the environment for stable-baselines3
    monitor_path = f"./rlbot/logs/monitor_{timestamp}"
    env = Monitor(env, monitor_path)
    env = DummyVecEnv([lambda: env])
    
    # Set up callbacks
    checkpoint_freq = 10000  # Save every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=f"./rlbot/checkpoints/{timestamp}/",
        name_prefix="runescape_bot",
        verbose=1,
    )
    
    # Set up evaluation callback (optional)
    eval_callback = None
    
    # Combine callbacks using a more general type annotation
    callbacks: List[BaseCallback] = []
    callbacks.append(checkpoint_callback)
    if eval_callback is not None:
        callbacks.append(eval_callback)
    
    # Create the PPO agent with MultiInputPolicy for Dict observation spaces
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"./rlbot/logs/tb_logs/",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={
            'net_arch': {
                'pi': [128, 128],
                'vf': [128, 128]
            }
        }
    )
    
    print(f"Starting training for {timesteps} timesteps...")
    
    # Learn without try/except - let any errors bubble up
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True
    )
    
    # Save the final model
    model.save(f"./rlbot/models/runescape_bot_{timestamp}")
    logger.info("Training completed successfully!")
    print("Training completed successfully!")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="RuneScape AI Training")
    parser.add_argument("--test", action="store_true", help="Test connection only without training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of timesteps to train for")
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Initialize console for prettier output
    console = Console()
    
    console.print("[bold cyan]RuneScape AI Training System[/bold cyan]")
    console.print(f"Debug mode: {'[green]Enabled[/green]' if args.debug or args.verbose else '[red]Disabled[/red]'}")
    console.print(f"Verbose logging: {'[green]Enabled[/green]' if args.verbose else '[red]Disabled[/red]'}")
    
    # Determine which function to run
    if args.test:
        console.print("[bold yellow]Running connection test...[/bold yellow]")
        test_connection(debug=args.debug or args.verbose)
    else:
        console.print(f"[bold yellow]Starting training for {args.timesteps} timesteps...[/bold yellow]")
        train_combat_bot(debug=args.debug or args.verbose, timesteps=args.timesteps) 
