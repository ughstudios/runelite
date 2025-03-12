#!/usr/bin/env python3
"""
RuneScape Environment Module

This module defines the RuneScapeEnv class, a custom Gymnasium environment
for controlling a character in RuneScape. The environment connects to the RuneLite client
via WebSocket and provides a reinforcement learning interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import asyncio
import json
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

# Import from our modules
from .models import GameState, Location, Health, NPC
from .websocket_client import WebSocketClient

# Configure logging
logger = logging.getLogger("RuneScapeEnv")

class Action(int):
    """Actions that can be performed in the RuneScape environment."""
    ATTACK = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    INTERFACE_ACTION = 5
    TOGGLE_RUN = 6
    INTERACT_OBJECT = 7
    INTERACT_NPC = 8
    CAMERA_ROTATE_LEFT = 9
    CAMERA_ROTATE_RIGHT = 10
    CAMERA_ROTATE_UP = 11
    CAMERA_ROTATE_DOWN = 12
    CAMERA_ZOOM_IN = 13
    CAMERA_ZOOM_OUT = 14

class RuneScapeEnv(gym.Env):
    """Custom Environment for controlling a character in RuneScape."""

    def __init__(self, task: str = "combat", debug: bool = True):
        """Initialize the RuneScape environment.
        
        Args:
            task: The task to perform ("combat", "fishing", etc.)
            debug: Whether to enable debug logging
        """
        # Configure logging
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Initialize task
        self.task = task
        
        # Define observation and action spaces
        self._setup_observation_space()
        self._setup_action_space()
        
        # Initialize tracking variables
        self._initialize_tracking_variables()
        
        # Create WebSocket client
        self.ws_client = WebSocketClient(debug=debug)
        
        # Wait for initial connection and get initial state
        logger.info("Waiting for WebSocket connection...")
        if not self.ws_client.wait_for_connection(30):
            logger.error("Failed to connect within 30 seconds")
            logger.error("Is the RuneLite client running with the RLBot plugin?")
            raise RuntimeError("Failed to connect to RuneLite")
        
        # Get initial state
        self._refresh_state()
        if self.state is None:
            logger.error("Failed to get initial state. Is the player logged in?")
    
    def _setup_observation_space(self):
        """Set up the observation space for the environment."""
        if self.task == "combat":
            # Combined observation space with both image and vector components
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                'vector': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(102,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _setup_action_space(self):
        """Set up the action space for the environment."""
        # Action space includes all possible actions
        self.action_space = gym.spaces.Discrete(15)  # Number of actions in Action enum
    
    def _initialize_tracking_variables(self):
        """Initialize tracking variables for the environment."""
        # Skills to track
        self.tracked_skills = ["attack", "strength", "defence", "ranged", "magic", "hitpoints", "prayer"]
        
        # Performance metrics
        self.actions_per_minute = 100  # Minimum actions per minute
        self.action_times = []
        
        # Skill experience tracking
        self.skill_xp = {skill: 0 for skill in self.tracked_skills}
        self.initial_skill_xp = {}
        
        # Exploration tracking
        self.visited_areas = set()
        self.last_position = None
        
        # Episode tracking
        self.timestep = 0
        self.max_steps = 2000
        
        # Combat tracking
        self.current_target = None
        self.is_in_combat = False
        self.last_combat_time = 0
        self.last_target_id = None
        
        # Health tracking
        self.player_health = 100
        self.max_player_health = 100
        
        # Interface tracking
        self.interfaces_open = False
        self.path_obstructed = False
        self.interface_open = False
        self.consecutive_interface_frames = 0
        self.interface_check_cooldown = 0.0
        
        # Run energy tracking
        self.run_toggle_cooldown = 30.0
        self.last_run_toggle = 0.0
        
        # State tracking
        self.state = None
        self.last_state_update = 0.0
        self.state_update_interval = 0.1
    
    def _refresh_state(self, force: bool = False) -> bool:
        """Refresh the state from the server if needed.
        
        Args:
            force: If True, force a state refresh regardless of when the last state was received
        
        Returns:
            bool: True if the state was refreshed, False otherwise
        """
        # If no state or state is older than refresh interval, or force is True, get a new one
        current_time = time.time()
        if force or self.state is None or (current_time - self.last_state_update) >= self.state_update_interval:
            # Send the get_state command
            command = {"type": "get_state"}
            
            # Use the websocket client to send the command
            self.ws_client.send_command(command)
            
            # Update the state from the client
            if self.ws_client.state is not None:
                self.state = self.ws_client.state
                self.last_state_update = current_time
                
                # Update tracking variables from state
                self._update_tracking_from_state()
                
                return True
        
        # State is recent enough, no need to refresh
        return False
    
    def _update_tracking_from_state(self):
        """Update tracking variables from the current state."""
        if self.state is None:
            return
        
        # Update player position
        if "player" in self.state:
            player = self.state["player"]
            
            # Get position
            location = player.get("location", player.get("position", {}))
            if location and "x" in location and "y" in location:
                self.last_position = (location["x"], location["y"])
            
            # Update health
            if "health" in player:
                health = player["health"]
                if isinstance(health, dict):
                    self.player_health = health.get("current", 0)
                    self.max_player_health = health.get("maximum", 100)
                else:
                    self.player_health = health
            
            # Update combat status
            self.is_in_combat = player.get("inCombat", False)
        
        # Update interface status
        self.interfaces_open = self.state.get("interfacesOpen", False)
        self.path_obstructed = self.state.get("pathObstructed", False)
    
    def _state_to_observation(self) -> Dict[str, np.ndarray]:
        """Convert state to the observation format expected by the agent."""
        # Initialize default observations
        image_obs = np.zeros((84, 84, 3), dtype=np.uint8)
        vector_obs = np.zeros(102, dtype=np.float32)
        
        # Return default observation if no state is available
        if self.state is None:
            if self.debug:
                logger.warning(f"No state available at timestep {self.timestep}")
            return {'image': image_obs, 'vector': vector_obs}
        
        # Process screenshot
        if "screenshot" in self.state:
            image_data = base64.b64decode(self.state["screenshot"])
            pil_image = Image.open(BytesIO(image_data))
            # Resize to 84x84
            resized_image = pil_image.resize((84, 84))
            # Convert to numpy array
            image_obs = np.array(resized_image, dtype=np.uint8)
        
        # Extract vector features
        features = self._extract_vector_features()
        
        # Convert features to numpy array in fixed order
        feature_keys = sorted(features.keys())
        for i, key in enumerate(feature_keys):
            if i < vector_obs.shape[0]:  # Avoid index out of bounds
                vector_obs[i] = features[key]
        
        # Return both image and vector observations
        return {'image': image_obs, 'vector': vector_obs}
    
    def _extract_vector_features(self) -> Dict[str, float]:
        """Extract vector features from the game state."""
        features = {}
        
        # Extract player information
        player = self.state.get("player", {})
        location = player.get("location", player.get("position", {}))
        
        # Player position
        features["player_x"] = float(location.get("x", 0))
        features["player_y"] = float(location.get("y", 0))
        
        # Player health
        health = player.get("health", {})
        current_health = 0
        max_health = 1  # Default to avoid division by zero
        
        if isinstance(health, dict):
            current_health = health.get("current", 0)
            max_health = health.get("maximum", 1)
        elif isinstance(health, (int, float)):
            current_health = health
        
        features["health"] = float(current_health) / max(1.0, float(max_health))
        
        # Combat status
        features["in_combat"] = 1.0 if player.get("inCombat", False) else 0.0
        
        # Run energy
        features["run_energy"] = float(player.get("runEnergy", 0)) / 100.0
        features["is_running"] = 1.0 if player.get("isRunning", False) else 0.0
        
        # Player skills
        skills = player.get("skills", {})
        for i, skill_name in enumerate(self.tracked_skills):
            if i < 7:  # Ensure we stay within observation space bounds
                skill_value = 1.0
                if skill_name in skills:
                    skill_data = skills[skill_name]
                    if isinstance(skill_data, dict):
                        skill_value = float(skill_data.get("level", 1))
                    elif isinstance(skill_data, (int, float)):
                        skill_value = float(skill_data)
                features[f"{skill_name}_level"] = skill_value
        
        # Nearby NPCs
        npcs = self.state.get("npcs", [])
        for i in range(min(5, len(npcs))):
            npc = npcs[i]
            prefix = f"npc{i+1}_"
            
            # NPC position relative to player
            npc_loc = npc.get("location", {})
            npc_x = float(npc_loc.get("x", 0))
            npc_y = float(npc_loc.get("y", 0))
            
            rel_x = npc_x - features["player_x"]
            rel_y = npc_y - features["player_y"]
            
            features[f"{prefix}rel_x"] = rel_x
            features[f"{prefix}rel_y"] = rel_y
            
            # Distance
            distance = math.sqrt(rel_x**2 + rel_y**2)
            if "distance" in npc:
                distance = float(npc["distance"])
            features[f"{prefix}distance"] = distance
            
            # Combat status
            features[f"{prefix}in_combat"] = 1.0 if npc.get("interacting", False) else 0.0
            
            # Health
            npc_health = npc.get("health", {})
            npc_health_percent = 1.0
            if npc_health:
                if isinstance(npc_health, dict):
                    current = float(npc_health.get("current", 0))
                    maximum = float(npc_health.get("maximum", 1))
                    npc_health_percent = current / max(1, maximum)
            features[f"{prefix}health"] = npc_health_percent
            
            # Combat level
            features[f"{prefix}level"] = float(npc.get("combatLevel", 0)) / 100.0
        
        return features
    
    def _action_to_command(self, action: int) -> Optional[Dict]:
        """Convert a high-level action into a command to send to the client."""
        if self.state is None:
            return None
        
        if action == Action.ATTACK:
            return self._create_attack_command()
        elif action == Action.MOVE_FORWARD:
            return self._create_move_command(0, 2)  # Move 2 tiles forward (y+2)
        elif action == Action.MOVE_BACKWARD:
            return self._create_move_command(0, -2)  # Move 2 tiles backward (y-2)
        elif action == Action.MOVE_LEFT:
            return self._create_move_command(-2, 0)  # Move 2 tiles left (x-2)
        elif action == Action.MOVE_RIGHT:
            return self._create_move_command(2, 0)  # Move 2 tiles right (x+2)
        elif action == Action.INTERFACE_ACTION:
            return self._create_interface_action_command()
        elif action == Action.TOGGLE_RUN:
            return self._create_toggle_run_command()
        elif action == Action.INTERACT_OBJECT:
            return self._create_interact_object_command()
        elif action == Action.INTERACT_NPC:
            return self._create_interact_npc_command()
        elif action == Action.CAMERA_ROTATE_LEFT:
            return {"action": "cameraRotate", "data": {"direction": "left"}}
        elif action == Action.CAMERA_ROTATE_RIGHT:
            return {"action": "cameraRotate", "data": {"direction": "right"}}
        elif action == Action.CAMERA_ROTATE_UP:
            return {"action": "cameraRotate", "data": {"direction": "up"}}
        elif action == Action.CAMERA_ROTATE_DOWN:
            return {"action": "cameraRotate", "data": {"direction": "down"}}
        elif action == Action.CAMERA_ZOOM_IN:
            return {"action": "cameraZoom", "data": {"direction": "in"}}
        elif action == Action.CAMERA_ZOOM_OUT:
            return {"action": "cameraZoom", "data": {"direction": "out"}}
        
        return None
    
    def _create_attack_command(self) -> Optional[Dict]:
        """Create a command to attack the nearest appropriate NPC."""
        npcs = self.state.get("npcs", [])
        player = self.state.get("player", {})
        
        # Calculate player's combat level
        player_combat_level = self._calculate_combat_level(player.get("skills", {}))
        
        # Filter for attackable NPCs
        attackable_npcs = [
            npc for npc in npcs
            if (npc.get("combatLevel", 0) > 0 and 
                abs(npc.get("combatLevel", 0) - player_combat_level) < 20 and
                not npc.get("interacting", False) and
                npc.get("id") != self.last_target_id)
        ]
        
        if not attackable_npcs:
            # Fallback to any non-interacting NPCs
            attackable_npcs = [
                npc for npc in npcs
                if (npc.get("combatLevel", 0) > 0 and 
                    not npc.get("interacting", False))
            ]
        
        if attackable_npcs:
            # Find nearest appropriate NPC
            nearest_npc = min(
                attackable_npcs,
                key=lambda x: (x.get("distance", float("inf"))) * 
                            (1 + abs(x.get("combatLevel", 0) - player_combat_level) / 20)
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
        
        # No suitable NPCs found, return movement to explore instead
        return self._create_exploration_command()
    
    def _create_exploration_command(self) -> Dict:
        """Create a command to explore when no suitable targets are found."""
        # Check if exploration data is available
        if "exploration" in self.state and "currentChunk" in self.state["exploration"]:
            current_chunk = (
                self.state["exploration"]["currentChunk"].get("x", 0),
                self.state["exploration"]["currentChunk"].get("y", 0)
            )
            
            # Get nearby chunks that haven't been visited
            nearby_chunks = [
                (current_chunk[0] + dx, current_chunk[1] + dy)
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                if (current_chunk[0] + dx, current_chunk[1] + dy) not in self.visited_areas
            ]
            
            if nearby_chunks:
                # Move towards nearest unvisited chunk
                target_chunk = min(
                    nearby_chunks, 
                    key=lambda c: abs(c[0] - current_chunk[0]) + abs(c[1] - current_chunk[1])
                )
                dx = target_chunk[0] - current_chunk[0]
                dy = target_chunk[1] - current_chunk[1]
                
                if abs(dx) > abs(dy):
                    return self._create_move_command(2 if dx > 0 else -2, 0)
                else:
                    return self._create_move_command(0, 2 if dy > 0 else -2)
        
        # Default to random movement
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        dx, dy = random.choice(directions)
        return self._create_move_command(dx, dy)
    
    def _create_move_command(self, dx: int, dy: int) -> Dict:
        """Create a command to move by the specified delta."""
        player = self.state.get("player", {})
        location = player.get("location", player.get("position", {}))
        
        current_x = location.get("x", 0)
        current_y = location.get("y", 0)
        
        return {
            "action": "moveAndClick",
            "data": {
                "targetType": "coordinates",
                "action": "Move",
                "x": current_x + dx,
                "y": current_y + dy
            }
        }
    
    def _create_interface_action_command(self) -> Optional[Dict]:
        """Create a command to interact with an interface element."""
        interfaces = self.state.get("interfaces", [])
        
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
        
        return None
    
    def _create_toggle_run_command(self) -> Optional[Dict]:
        """Create a command to toggle run status based on energy levels."""
        current_time = time.time()
        if current_time - self.last_run_toggle < self.run_toggle_cooldown:
            return None  # Still on cooldown
        
        player = self.state.get("player", {})
        run_energy = player.get("runEnergy", 0.0)
        is_running = player.get("isRunning", False)
        
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
    
    def _create_interact_object_command(self) -> Optional[Dict]:
        """Create a command to interact with the nearest game object."""
        objects = self.state.get("objects", [])
        
        interactable_objects = [
            obj for obj in objects
            if "actions" in obj and obj["actions"]
        ]
        
        if interactable_objects:
            # Sort by distance
            nearest_object = min(
                interactable_objects,
                key=lambda x: x.get("distance", float("inf"))
            )
            
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "object",
                    "action": nearest_object["actions"][0],
                    "objectId": nearest_object["id"]
                }
            }
        
        return None
    
    def _create_interact_npc_command(self) -> Optional[Dict]:
        """Create a command to interact (non-combat) with the nearest NPC."""
        npcs = self.state.get("npcs", [])
        
        interactable_npcs = [
            npc for npc in npcs
            if "actions" in npc and npc["actions"]
        ]
        
        if interactable_npcs:
            # Sort by distance
            nearest_npc = min(
                interactable_npcs,
                key=lambda x: x.get("distance", float("inf"))
            )
            
            return {
                "action": "moveAndClick",
                "data": {
                    "targetType": "npc",
                    "action": nearest_npc["actions"][0],
                    "npcId": nearest_npc["id"]
                }
            }
        
        return None
    
    def _calculate_combat_level(self, skills: Dict[str, Any]) -> int:
        """Calculate the player's combat level based on their skills."""
        # Extract skill levels
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
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        # Initialize reward
        reward = 0.0

        # Check if we have a previous or current state
        if self.state is None or self.last_position is None:
            logger.warning("Cannot calculate reward: No state or last position")
            return 0.0
        
        # Get the current player state
        player = self.state.get("player", {})
        if not player:
            logger.warning("Player data missing in state")
            return 0.0

        # Health change reward
        health = player.get("health", {})
        current_health = 0
        
        if isinstance(health, dict):
            current_health = health.get("current", 0)
        elif isinstance(health, (int, float)):
            current_health = int(health)
        
        # Calculate health difference
        health_diff = current_health - self.player_health
        
        if health_diff != 0:
            # Negative reward for taking damage (-0.5 for each hp lost)
            if health_diff < 0:
                health_reward = health_diff * 0.5
                reward += health_reward
            # Small positive reward for healing
            elif health_diff > 0:
                health_reward = health_diff * 0.1
                reward += health_reward
            
            # Update health tracking
            self.player_health = current_health
        
        # Large penalty for dying and respawning
        if self.player_health <= 0 and current_health > 0:
            logger.info("Player died and respawned")
            reward -= 50.0
        
        # Experience gains reward
        skills = player.get("skills", {})
        
        for skill_name, skill_data in skills.items():
            if skill_name not in self.tracked_skills:
                continue
            
            # Extract experience value
            current_xp = 0
            if isinstance(skill_data, dict):
                current_xp = skill_data.get("experience", 0)
            elif isinstance(skill_data, (int, float)):
                current_xp = int(skill_data)
            
            # Get previous XP for this skill
            previous_xp = self.skill_xp.get(skill_name, 0)
            
            # Calculate XP gained
            xp_gained = current_xp - previous_xp
            
            if xp_gained > 0:
                # Reward for gaining XP
                xp_reward = xp_gained * 0.01
                reward += xp_reward
                
                # Update tracked XP
                self.skill_xp[skill_name] = current_xp
        
        # Movement and exploration reward
        location = player.get("location", player.get("position", {}))
        current_x = location.get("x", 0)
        current_y = location.get("y", 0)
        current_position = (current_x, current_y)
        
        if current_position != self.last_position:
            # Small reward for moving
            movement_reward = 0.01
            reward += movement_reward
        
        # Track position
        self.last_position = current_position
        
        # Check if we've visited this area before
        region_x = current_position[0] // 10
        region_y = current_position[1] // 10
        region = (region_x, region_y)
        
        # Exploration reward
        if region not in self.visited_areas:
            # Reward for exploring new areas
            exploration_reward = 1.0
            reward += exploration_reward
            self.visited_areas.add(region)
        
        # Combat reward
        is_in_combat = player.get("inCombat", False)
        
        if is_in_combat:
            # Reward for being in combat
            combat_reward = 0.1
            reward += combat_reward
            self.last_combat_time = time.time()
        
        return reward
    
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
        self._refresh_state(force=True)
        
        # Reset player state
        if self.state is not None and "player" in self.state:
            player = self.state["player"]
            location = player.get("location", player.get("position", {}))
            self.last_position = (location.get("x", 0), location.get("y", 0))
            
            # Get health information
            health = player.get("health", {})
            if isinstance(health, dict):
                self.player_health = health.get("current", 100)
                self.max_player_health = health.get("maximum", 100)
            else:
                self.player_health = health if isinstance(health, (int, float)) else 100
            
            # Record initial XP
            skills = player.get("skills", {})
            for skill, data in skills.items():
                if skill in self.tracked_skills:
                    xp = 0
                    if isinstance(data, dict):
                        xp = data.get("experience", 0)
                    elif isinstance(data, (int, float)):
                        xp = int(data)
                    
                    self.skill_xp[skill] = xp
                    self.initial_skill_xp[skill] = xp
        else:
            # No valid state available
            logger.warning("Reset: Could not get valid state")
            # Provide default observation
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
            logger.warning(f"Invalid action: {action}")
            action = 0
        
        # Store the timestamp of the last state update before taking action
        pre_action_state_timestamp = self.last_state_update
        
        # Convert action to command
        cmd = self._action_to_command(action)
        
        # Store the current state
        old_state = self.state
        
        # Send the command if valid
        if cmd is not None:
            # Send command
            self.ws_client.send_command(cmd)
            
            # Get the new state after action
            max_wait_time = 2.0
            wait_start = time.time()
            
            # Wait for a state update
            while (self.last_state_update <= pre_action_state_timestamp and 
                   time.time() - wait_start < max_wait_time):
                # Force a state refresh
                self._refresh_state(force=True)
                # If we still don't have a new state, wait a bit
                if self.last_state_update <= pre_action_state_timestamp:
                    time.sleep(0.05)
            
            if self.last_state_update <= pre_action_state_timestamp:
                logger.warning(f"Timed out waiting for state update after action {action}")
        
        # Handle case where we couldn't get state
        if self.state is None:
            logger.warning("Failed to get state after action")
            # Use old state if available
            if old_state is not None:
                self.state = old_state
            else:
                # Create empty observation
                image_obs = np.zeros((84, 84, 3), dtype=np.uint8)
                vector_obs = np.zeros(102, dtype=np.float32)
                return {'image': image_obs, 'vector': vector_obs}, 0.0, False, True, {}
        
        # Increment timestep
        self.timestep += 1
        
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
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self, 'ws_client'):
            self.ws_client.close() 