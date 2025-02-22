try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

import numpy as np
import json
import logging
import asyncio
import websockets
import nest_asyncio
import threading
import time
from typing import Optional, Dict, Any, Tuple, List, Set, Union, TYPE_CHECKING, cast
from enum import Enum
from dataclasses import dataclass
from jsonschema import validate, ValidationError
import os
import base64
from io import BytesIO
from PIL import Image
from PIL.ImageFile import ImageFile
from gymnasium.spaces import Dict as GymDict

if TYPE_CHECKING:
    from gymnasium.spaces.dict import Dict as GymDict
    from gymnasium.spaces.space import Space

# Type aliases
StateDict = Dict[str, Any]
FloatArray = np.ndarray

nest_asyncio.apply()

# Configure logging to be less verbose
logging.basicConfig(
    level=logging.WARNING,  # Change from INFO to WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Action(Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    DO_NOTHING = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6
    ZOOM_IN = 7
    ZOOM_OUT = 8
    ATTACK = 9

class RuneScapeEnv(gym.Env):
    """
    Gymnasium Environment for RuneScape focused on combat training.
    """

    def __init__(self, websocket_url: str = "ws://localhost:43595", task: str = "combat"):
        super().__init__()
        self.logger = logging.getLogger("RuneScapeEnv")
        self.logger.setLevel(logging.INFO)  # Keep important env logs at INFO
        
        # Add a handler that only shows WARNING and above for websockets
        logging.getLogger('websockets').setLevel(logging.WARNING)
        
        self.websocket_url = websocket_url
        self.task = task
        self.screenshot_shape = (480, 640, 3)  # Default size, will be updated on first screenshot

        # Load JSON schemas
        schema_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(schema_dir, 'command_schema.json')) as f:
            self.command_schema = json.load(f)
        with open(os.path.join(schema_dir, 'state_schema.json')) as f:
            self.state_schema = json.load(f)

        # Environment state variables
        self.current_state: Optional[Dict] = None
        self.last_combat_exp = 0
        self.visited_areas: Set[Tuple[int, int]] = set()
        self.interfaces_open = False
        self.path_obstructed = False
        self.last_action: Optional[Action] = None
        self.last_action_time: float = 0.0
        self.action_cooldown = 0.2
        self.last_position = None
        self.consecutive_same_pos = 0
        self.last_command: Optional[str] = None
        self.command_time: float = 0.0
        self.min_command_interval = 0.1
        self.last_target_id = None
        
        # Websocket setup
        self.ws = None
        self.loop = asyncio.new_event_loop()
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()
        
        # Wait for initial connection
        timeout = 30
        start_time = time.time()
        while not self.ws and time.time() - start_time < timeout:
            time.sleep(0.1)
        if not self.ws:
            self.logger.error(f"Failed to connect within {timeout} seconds")
            
        # Initialize observation and action spaces
        self.observation_space = gym.spaces.Dict({
            'screenshot': gym.spaces.Box(low=0, high=255, shape=self.screenshot_shape, dtype=np.uint8),
            'player_position': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'player_combat_stats': gym.spaces.Box(low=1, high=99, shape=(7,), dtype=np.int32),
            'player_health': gym.spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            'player_prayer': gym.spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            'player_run_energy': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'skills': gym.spaces.Box(low=1, high=99, shape=(23,), dtype=np.int32),
            'npcs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 6), dtype=np.float32),  # 10 nearest NPCs, 6 features each
            'in_combat': gym.spaces.Discrete(2),
            'interfaces_open': gym.spaces.Discrete(2),
            'path_obstructed': gym.spaces.Discrete(2),
            'current_chunk': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32),
            'visited_chunks_count': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'nearby_areas': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 5), dtype=np.float32),
            'exploration_score': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Discrete(len(Action))

    def _run_websocket_loop(self):
        """Run the websocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_client())
        
    async def _websocket_client(self):
        """Handle websocket connection and message processing."""
        while True:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.ws = websocket
                    self.logger.info("Connected to RuneLite")
                    
                    while True:
                        try:
                            message = await websocket.recv()
                            state = json.loads(message)
                            # Validate state against schema
                            validate(instance=state, schema=self.state_schema)
                            self.current_state = state
                            
                            # Update environment flags based on state
                            self.interfaces_open = state.get('interfacesOpen', False)
                            self.path_obstructed = state.get('pathObstructed', False)
                            
                        except ValidationError as e:
                            self.logger.error(f"Invalid state received: {e}")
                        except websockets.ConnectionClosed:
                            self.logger.warning("WebSocket connection closed")
                            break
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse message: {e}")
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
                            
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                self.ws = None
                await asyncio.sleep(5)
                
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and wait for an initial state."""
        self.last_combat_exp = 0
        self.visited_areas.clear()
        
        # Wait for a valid state
        timeout = 10
        start_time = time.time()
        while not self.current_state and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.current_state:
            self.logger.warning("No state received during reset")
            
        return self._state_to_observation(self.current_state), {}

    async def _execute_command(self, command: Dict) -> None:
        """
        Asynchronously send a command via the websocket with rate limiting and retry logic.
        """
        if self.ws is None:
            self.logger.error("WebSocket not connected")
            return

        max_retries = 3
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # Validate command against schema
                validate(instance=command, schema=self.command_schema)
                
                # Convert command to JSON for comparison
                cmd_json = json.dumps(command)
                
                # Check if this is a duplicate command
                current_time = time.time()
                if (self.last_command == cmd_json and 
                    current_time - self.command_time < self.action_cooldown):
                    self.logger.debug(f"Skipping duplicate command: {cmd_json}")
                    return
                    
                # Rate limit commands
                if current_time - self.command_time < self.min_command_interval:
                    wait_time = self.min_command_interval - (current_time - self.command_time)
                    self.logger.debug(f"Rate limiting - waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    
                # Update command tracking
                self.last_command = cmd_json
                self.command_time = current_time
                
                # Send command
                if not self.ws.open:
                    raise Exception("WebSocket connection is closed")
                    
                await self.ws.send(cmd_json)
                self.logger.debug(f"Sent command: {cmd_json}")
                
                # Wait longer after sending command
                await asyncio.sleep(0.2)  # Increased from 0.1
                return  # Success, exit retry loop
                
            except ValidationError as e:
                self.logger.error(f"Invalid command schema: {e}")
                return  # Don't retry schema validation errors
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.warning(f"Command failed (attempt {retry_count}/{max_retries}): {e}")
                    await asyncio.sleep(0.5 * retry_count)  # Exponential backoff
                else:
                    self.logger.error(f"Command failed after {max_retries} attempts. Last error: {e}")

    def step(self, action: Action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Enforce action cooldown
        current_time = time.time()
        if self.last_action_time and current_time - self.last_action_time < self.action_cooldown:
            sleep_time = self.action_cooldown - (current_time - self.last_action_time)
            self.logger.debug(f"Waiting {sleep_time:.1f}s for action cooldown")
            time.sleep(sleep_time)
        
        # Convert and execute action
        command = self._action_to_command(action)
        if command is not None:
            if self.loop and not self.loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(self._execute_command(command), self.loop)
                try:
                    # Increased timeout and added more detailed error handling
                    future.result(timeout=2.0)  # Increased from 1.0
                    self.last_action = action
                    self.last_action_time = time.time()
                except asyncio.TimeoutError:
                    self.logger.error("Command execution timed out after 2 seconds")
                except Exception as e:
                    self.logger.error(f"Error executing command: {str(e)}", exc_info=True)
            else:
                self.logger.error("Event loop not available for sending command")
        
        # Wait for state update with exponential backoff
        max_wait = 0.5
        wait_time = 0.1
        start_time = time.time()
        
        while not self.current_state and time.time() - start_time < max_wait:
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, max_wait)
        
        # Get updated state and compute rewards
        observation = self._state_to_observation(self.current_state)
        reward = self._calculate_reward(self.current_state)
        done = self._is_episode_done(self.current_state)
        truncated = False
        info = {
            'action': action.name if isinstance(action, Action) else Action(action).name,
            'state': self.current_state,
            'command_sent': command is not None
        }
        return observation, reward, done, truncated, info

    def render(self, mode: str = "human"):
        """
        Rendering is handled externally by the client.
        """
        pass

    def close(self):
        """Clean up resources."""
        if self.ws is not None and self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        self.logger.info("Environment closed")

    def _get_empty_observation(self) -> Dict[str, np.ndarray]:
        """
        Return an empty observation (zero-filled) for when no state is available.
        """
        return {
            'screenshot': np.zeros(self.screenshot_shape, dtype=np.uint8),
            'player_position': np.zeros(3, dtype=np.float32),
            'player_combat_stats': np.ones(7, dtype=np.int32),
            'player_health': np.ones(1, dtype=np.int32),
            'player_prayer': np.zeros(1, dtype=np.int32),
            'player_run_energy': np.zeros(1, dtype=np.float32),
            'skills': np.ones(23, dtype=np.int32),
            'npcs': np.zeros((10, 6), dtype=np.float32),
            'in_combat': np.zeros(1, dtype=np.int32),
            'interfaces_open': np.zeros(1, dtype=np.int32),
            'path_obstructed': np.zeros(1, dtype=np.int32),
            'current_chunk': np.zeros(2, dtype=np.int32),
            'visited_chunks_count': np.zeros(1, dtype=np.int32),
            'nearby_areas': np.zeros((9, 5), dtype=np.float32),
            'exploration_score': np.zeros(1, dtype=np.float32)
        }

    def _process_screenshot(self, screenshot_base64: str) -> np.ndarray:
        """Convert base64 screenshot to numpy array and resize if needed."""
        try:
            # Decode base64 string to image
            img_data = base64.b64decode(screenshot_base64)
            img: ImageFile = cast(ImageFile, Image.open(BytesIO(img_data)))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image to match expected dimensions
            if img.size != (self.screenshot_shape[1], self.screenshot_shape[0]):
                img = img.resize((self.screenshot_shape[1], self.screenshot_shape[0]), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            self.logger.error(f"Error processing screenshot: {e}")
            return np.zeros(self.screenshot_shape, dtype=np.uint8)

    def _state_to_observation(self, state: Optional[Dict]) -> Dict[str, np.ndarray]:
        """Convert the raw state from RuneLite into a gym observation."""
        if not state:
            return self._get_empty_observation()

        # Process screenshot if available
        screenshot = np.zeros(self.screenshot_shape, dtype=np.uint8)
        if state.get('screenshot'):
            screenshot = self._process_screenshot(state['screenshot'])

        # Remove debug logging of raw state
        # self.logger.info(f"Raw state: {json.dumps(state, indent=2)}")
        # self.logger.info(f"Player health from state: {state.get('playerHealth', '?')}")
        # self.logger.info(f"Player max health from state: {state.get('playerMaxHealth', '?')}")

        def safe_get(d: Dict, *keys, default=0):
            for key in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(key, default)
            return d if d is not None else default

        def normalize(value, min_val, max_val):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            return np.clip((value - min_val) / (max_val - min_val), 0, 1)

        player = state.get('player', {})
        
        # Extract player position
        location = player.get('location', {})
        position = np.array([
            location.get('x', 0),
            location.get('y', 0),
            location.get('plane', 0)
        ], dtype=np.float32)

        # Extract combat stats
        skills = player.get('skills', {})
        combat_stats = np.array([
            skills.get('ATTACK', {}).get('level', 1),
            skills.get('STRENGTH', {}).get('level', 1),
            skills.get('DEFENCE', {}).get('level', 1),
            skills.get('RANGED', {}).get('level', 1),
            skills.get('MAGIC', {}).get('level', 1),
            skills.get('HITPOINTS', {}).get('level', 1),
            skills.get('PRAYER', {}).get('level', 1)
        ], dtype=np.int32)

        # Extract health and prayer
        health = player.get('health', {})
        health_array = np.array([health.get('current', 1)], dtype=np.int32)
        prayer = np.array([player.get('prayer', 0)], dtype=np.int32)
        run_energy = np.array([player.get('runEnergy', 0.0)], dtype=np.float32)

        # Process NPCs (take 10 nearest)
        npcs = state.get('npcs', [])
        npcs.sort(key=lambda x: x.get('distance', float('inf')))
        npc_features = np.zeros((10, 6), dtype=np.float32)
        for i, npc in enumerate(npcs[:10]):
            npc_stats = [
                npc.get('id', 0),
                npc.get('level', 0),
                npc.get('distance', 0),
                npc.get('interacting', False),
                npc.get('health', {}).get('current', 0) / max(npc.get('health', {}).get('maximum', 1), 1),
                1.0  # NPC exists flag
            ]
            npc_features[i] = npc_stats

        # Create skills array
        skills_array = np.ones(23, dtype=np.int32)
        for i, skill in enumerate(skills.values()):
            if i < 23:
                skills_array[i] = skill.get('level', 1)

        return {
            'screenshot': screenshot,
            'player_position': position,
            'player_combat_stats': combat_stats,
            'player_health': health_array,
            'player_prayer': prayer,
            'player_run_energy': run_energy,
            'skills': skills_array,
            'npcs': npc_features,
            'in_combat': np.array([player.get('inCombat', False)], dtype=np.int32),
            'interfaces_open': np.array([1 if self.interfaces_open else 0], dtype=np.int32),
            'path_obstructed': np.array([1 if self.path_obstructed else 0], dtype=np.int32),
            'current_chunk': np.array([
                safe_get(state, 'exploration', 'currentChunk', 'x', default=0),
                safe_get(state, 'exploration', 'currentChunk', 'y', default=0)
            ], dtype=np.int32),
            'visited_chunks_count': np.array([
                safe_get(state, 'exploration', 'visitedChunks', default=0)
            ], dtype=np.int32),
            'nearby_areas': np.zeros((9, 5), dtype=np.float32),
            'exploration_score': np.array([0.0], dtype=np.float32)
        }

    def _calculate_reward(self, state: Optional[Dict]) -> float:
        """
        Compute reward based on combat experience gains.
        The primary goal is to maximize experience gain through combat.
        """
        if not state:
            return 0.0

        reward = 0.0
        player = state.get('player', {})
        skills = player.get('skills', {})

        # Calculate total combat experience
        combat_skills = ['ATTACK', 'STRENGTH', 'DEFENCE', 'RANGED', 'MAGIC', 'HITPOINTS']
        current_combat_exp = sum(
            skills.get(skill, {}).get('experience', 0)
            for skill in combat_skills
        )
        
        # Primary reward: Combat experience gains
        if current_combat_exp > self.last_combat_exp:
            exp_gain = current_combat_exp - self.last_combat_exp
            reward += float(exp_gain) * 0.5
            self.logger.info(f"Gained {exp_gain} combat exp")  # Simplified reward logging
        self.last_combat_exp = current_combat_exp

        # Health management during combat
        if player.get('inCombat', False):
            health = player.get('health', {})
            health_ratio = health.get('current', 0) / max(health.get('maximum', 1), 1)
            
            if health_ratio < 0.3:
                reward -= 5.0
                self.logger.warning("Low health during combat")  # Changed from info to warning

        # Penalty for being stuck
        if self.consecutive_same_pos > 2 and not player.get('inCombat', False):
            reward -= 0.5
            self.logger.debug("Penalty for inactivity")  # Changed from info to debug

        return reward

    def _is_episode_done(self, state: Optional[Dict]) -> bool:
        """Episode ends if player dies or no NPCs remain."""
        if not state:
            return False
            
        player = state.get('player', {})
        health = player.get('health', {})
        return health.get('current', 0) <= 0 or len(state.get('npcs', [])) == 0

    def _action_to_command(self, action: Action) -> Optional[Dict]:
        """Convert a high-level action into a command that follows the schema."""
        if not self.current_state:
            self.logger.warning("No current state available")
            return None
            
        try:
            if isinstance(action, (int, np.integer)):
                action = Action(action)
            
            player = self.current_state.get('player', {})
            location = player.get('location', {})
            current_x = location.get('x', 0)
            current_y = location.get('y', 0)
            
            if action == Action.ATTACK:
                # Find nearest attackable NPC
                npcs = self.current_state.get('npcs', [])
                attackable_npcs = [
                    npc for npc in npcs 
                    if npc.get('combatLevel', 0) > 0
                    and not npc.get('interacting', False)
                    and npc.get('id') != self.last_target_id
                ]
                
                if not attackable_npcs:
                    attackable_npcs = [
                        npc for npc in npcs 
                        if npc.get('combatLevel', 0) > 0
                        and npc.get('id') != self.last_target_id
                    ]
                
                if attackable_npcs:
                    nearest_npc = min(attackable_npcs, key=lambda x: x.get('distance', float('inf')))
                    self.last_target_id = nearest_npc.get('id')
                    
                    return {
                        "action": "moveAndClick",
                        "data": {
                            "targetType": "npc",
                            "action": "Attack",
                            "npcId": nearest_npc['id']
                        }
                    }
                else:
                    # Move randomly to find NPCs
                    import random
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
            elif action == Action.ROTATE_LEFT:
                return {
                    "action": "camera_rotate",
                    "data": {
                        "right": False
                    }
                }
            elif action == Action.ROTATE_RIGHT:
                return {
                    "action": "camera_rotate",
                    "data": {
                        "right": True
                    }
                }
            elif action == Action.ZOOM_IN:
                return {
                    "action": "camera_zoom",
                    "data": {
                        "in": True
                    }
                }
            elif action == Action.ZOOM_OUT:
                return {
                    "action": "camera_zoom",
                    "data": {
                        "in": False
                    }
                }
            elif action == Action.DO_NOTHING:
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting action to command: {e}")
            return None