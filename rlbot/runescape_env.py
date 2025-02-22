try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

import numpy as np
import json
import logging
import asyncio
import websockets
try:
    import nest_asyncio
except ImportError:
    import nest_asyncio  # type: ignore
import threading
import time
from typing import Optional, Dict, Any, Tuple, List, Set, TypeVar, cast, Union, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
from command_schema import Command, ActionType, TargetType, MoveAndClickData, CameraRotateData, CameraZoomData

if TYPE_CHECKING:
    from gymnasium.spaces.dict import Dict as GymDict
    from gymnasium.spaces.space import Space

# Type aliases
StateDict = Dict[str, Any]
SpaceDict = Dict[str, 'Space']
FloatArray = np.ndarray[Any, np.dtype[np.float32]]

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Action(Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    ATTACK_NEAREST = 4
    TAKE_NEAREST = 5
    ROTATE_LEFT = 6
    ROTATE_RIGHT = 7
    ZOOM_IN = 8
    ZOOM_OUT = 9
    DO_NOTHING = 10

class RuneScapeEnv(gym.Env):
    """
    Gymnasium Environment for RuneScape with enhanced capabilities
    """
    
    def __init__(self, websocket_url: str = "ws://localhost:43594", task: str = "combat"):
        super().__init__()
        
        # Set up logger first thing
        self.logger = logging.getLogger("rlbot.env")
        self.logger.setLevel(logging.INFO)
        
        # Ensure we have a handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        self.logger.info("Initializing RuneScape Environment")
        self.logger.info(f"WebSocket URL: {websocket_url}")
        self.logger.info(f"Task: {task}")
        
        # Action rate limiting
        self.GAME_TICK = 0.6  # RuneScape game tick in seconds
        self.MIN_ACTION_DELAY = self.GAME_TICK  # Minimum time between actions
        self.MAX_ACTIONS_PER_MINUTE = 30  # More reasonable action limit (1 action per 2 seconds on average)
        self.action_timestamps: List[float] = []  # Track recent action times
        self.last_action_time = 0.0
        self.action_window = 60.0  # Time window for rate limiting in seconds
        
        # WebSocket setup
        self.websocket_url = websocket_url
        self.ws = None
        self.loop = None
        self.task = task
        self.ws_task = None
        self.ws_connected = threading.Event()
        self.ws_should_close = threading.Event()  # New flag to control WebSocket shutdown
        
        # Initialize observation and action spaces
        self.observation_space = gym.spaces.Dict({
            'player_position': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'player_combat_stats': gym.spaces.Box(low=1, high=99, shape=(7,), dtype=np.int32),
            'player_health': gym.spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            'player_prayer': gym.spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            'player_run_energy': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'npcs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 8), dtype=np.float32),
            'inventory': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28, 3), dtype=np.float32),
            'skills': gym.spaces.Box(low=1, high=99, shape=(23,), dtype=np.int32),
            'ground_items': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20, 8), dtype=np.float32),
            'ground_items_total_ge_value': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'ground_items_total_ha_value': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'ground_items_nearest_distance': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'interfaces_open': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'path_obstructed': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'current_chunk': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32),
            'visited_chunks_count': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'nearby_areas': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 5), dtype=np.float32),
            'exploration_score': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Discrete(len(Action))
        
        # Initialize state variables
        self.last_observation: Optional[np.ndarray] = None
        self.last_reward: float = 0
        self.last_combat_exp: int = 0
        self.last_inventory: List[Dict[str, Any]] = []
        self.last_npcs: List[Dict[str, Any]] = []
        self.last_objects: List[Dict[str, Any]] = []
        self.current_state: Optional[StateDict] = None
        self.path_obstructed: bool = False
        self.interfaces_open: bool = False
        self.visited_areas: Set[Tuple[int, int]] = set()
        
        # Set up WebSocket connection
        self._setup_websocket()
        
        # Wait for WebSocket connection
        if not self.ws_connected.wait(timeout=10):
            self.logger.error("Failed to connect to RuneLite WebSocket after 10 seconds")
            raise ConnectionError("Failed to connect to RuneLite WebSocket")
        
        # Exploration tracking
        self.area_scores: Dict[Tuple[int, int], float] = {}
        self.last_exploration_time: Dict[Tuple[int, int], float] = {}
        self.exploration_cooldown = 300  # 5 minutes cooldown for revisiting areas
        
        # Movement tracking
        self.is_moving = False
        self.last_position = None
        self.movement_timeout = 10  # Maximum seconds to wait for movement
        self.position_check_interval = 0.1  # How often to check position
        self.stationary_threshold = 3  # How many position checks must be same to consider stopped
        self.stationary_count = 0
    
    def _setup_websocket(self):
        """Set up WebSocket connection to RuneLite"""
        try:
            def run_websocket():
                try:
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self.loop = loop
                    
                    # Run the websocket handler
                    loop.run_until_complete(self.websocket_handler())
                except Exception as e:
                    self.logger.error("WebSocket handler error: " + str(e))
                finally:
                    try:
                        # Only close the loop if it's not already closed and we're shutting down
                        if not loop.is_closed() and self.ws_should_close.is_set():
                            pending = asyncio.all_tasks(loop)
                            loop.run_until_complete(asyncio.gather(*pending))
                            loop.close()
                    except Exception as e:
                        self.logger.error("Error closing event loop: " + str(e))
            
            self.ws_task = threading.Thread(target=run_websocket)
            self.ws_task.daemon = True
            self.ws_task.start()
            
        except Exception as e:
            self.logger.error("Failed to setup WebSocket: " + str(e))
            raise

    async def websocket_handler(self):
        """Handle WebSocket connection and messages"""
        self.logger.info("Attempting to connect to RuneLite...")
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.ws = websocket
                    self.logger.info("Connected to RuneLite successfully!")
                    self.ws_connected.set()  # Signal that we're connected
                    
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            # Handle status updates
                            if isinstance(data, dict) and 'status' in data:
                                if data['status'] == 'obstructed':
                                    self.path_obstructed = True
                                    self.logger.warning(f"Path obstructed to {data.get('target', 'unknown location')}")
                                continue
                            
                            # Handle regular state updates
                            self.current_state = data
                            
                            # Update interface state
                            self.interfaces_open = data.get('interfacesOpen', False)
                            
                            # Log significant state changes
                            if self._has_significant_changes(data):
                                self._log_state_changes(data)
                            
                        except websockets.exceptions.ConnectionClosed:
                            self.logger.error("WebSocket connection closed")
                            break
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse message: {e}")
                        except Exception as e:
                            self.logger.error(f"WebSocket error: {e}")
                            break
                            
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Failed to connect to RuneLite (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(5)  # Wait before retrying
                else:
                    self.logger.error("Max retries reached, giving up on WebSocket connection")
                    raise ConnectionError("Failed to establish WebSocket connection after max retries")

    def _has_significant_changes(self, new_state: Dict) -> bool:
        """Check if there are significant changes in the state"""
        if not self.current_state:
            return True
            
        significant_changes = [
            new_state.get('playerHealth') != self.current_state.get('playerHealth'),
            new_state.get('playerPrayer') != self.current_state.get('playerPrayer'),
            len(new_state.get('npcs', [])) != len(self.current_state.get('npcs', [])),
            new_state.get('playerAnimation') != self.current_state.get('playerAnimation'),
            new_state.get('playerInteracting') != self.current_state.get('playerInteracting')
        ]
        
        return any(significant_changes)
    
    def _log_state_changes(self, state: Dict):
        """Log significant state changes"""
        if not self.current_state:
            self.logger.info("Initial state received")
            return
            
        # Log health changes
        if state.get('playerHealth') != self.current_state.get('playerHealth'):
            self.logger.info("Health changed: " + str(self.current_state.get('playerHealth')) + " -> " + str(state.get('playerHealth')))
        
        # Log prayer changes
        if state.get('playerPrayer') != self.current_state.get('playerPrayer'):
            self.logger.info("Prayer changed: " + str(self.current_state.get('playerPrayer')) + " -> " + str(state.get('playerPrayer')))
        
        # Log NPC changes
        new_npcs = len(state.get('npcs', []))
        old_npcs = len(self.current_state.get('npcs', []))
        if new_npcs != old_npcs:
            self.logger.info("NPCs in range changed: " + str(old_npcs) + " -> " + str(new_npcs))
        
        # Log animation changes
        if state.get('playerAnimation') != self.current_state.get('playerAnimation'):
            self.logger.info("Animation changed: " + str(self.current_state.get('playerAnimation')) + " -> " + str(state.get('playerAnimation')))
    
    def connect_websocket(self):
        """Initialize WebSocket connection to RuneLite"""
        def run_websocket():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.websocket_handler())
        
        self.ws_task = threading.Thread(target=run_websocket)
        self.ws_task.daemon = True
        self.ws_task.start()
        
        # Wait for WebSocket to connect with timeout
        start_time = time.time()
        while not self.ws and time.time() - start_time < 10:  # Wait up to 10 seconds
            time.sleep(0.1)
        
        if not self.ws:
            self.logger.error("Failed to connect to RuneLite WebSocket after 10 seconds")
            raise ConnectionError("Failed to connect to RuneLite WebSocket")
    
    async def _send_command(self, command: Command) -> None:
        """Send a command to the RuneLite client via WebSocket"""
        if not self.ws_connected.is_set():
            self.logger.error("Cannot send command - WebSocket not connected")
            return
        
        try:
            cmd_dict = command.to_dict()
            if self.ws is not None:
                self.logger.info("Sending command: " + json.dumps(cmd_dict))
                await self.ws.send(json.dumps(cmd_dict))
                self.logger.debug("Command sent successfully: " + json.dumps(cmd_dict))
            else:
                self.logger.error("WebSocket connection not established")
                raise ConnectionError("WebSocket connection not established")
        except websockets.exceptions.ConnectionClosed:
            self.logger.error("WebSocket connection closed unexpectedly")
            self.ws_connected.clear()
            # Attempt to reconnect
            self._setup_websocket()
        except Exception as e:
            self.logger.error("Error sending command: " + str(e))
            raise

    def _move_and_click(self, target_type: TargetType, **kwargs) -> None:
        """Send a move and click command with validation"""
        if not self.ws_connected.is_set():
            self.logger.error("Cannot send move and click - WebSocket not connected")
            return
        
        try:
            data = MoveAndClickData(targetType=target_type, **kwargs)
            data.validate()  # This will raise ValueError if data is invalid
            cmd = Command(action=ActionType.MOVE_AND_CLICK, data=data)
            
            if self.loop is not None and not self.loop.is_closed():
                self.logger.info("Executing move and click: " + str(kwargs))
                future = asyncio.run_coroutine_threadsafe(self._send_command(cmd), self.loop)
                # Wait for the command to complete with a timeout
                future.result(timeout=5)
            else:
                self.logger.error("Event loop not initialized or closed")
                raise RuntimeError("Event loop not initialized or closed")
        except asyncio.TimeoutError:
            self.logger.error("Move and click command timed out")
        except ValueError as e:
            self.logger.error("Invalid move and click data: " + str(e))
            raise
        except Exception as e:
            self.logger.error("Error in move and click: " + str(e))
            raise

    def _rotate_camera(self, right: bool) -> None:
        """Rotate the camera left or right"""
        if not self.ws_connected.is_set():
            self.logger.error("Cannot rotate camera - WebSocket not connected")
            return
        
        try:
            cmd = Command(
                action=ActionType.CAMERA_ROTATE,
                data=CameraRotateData(right=right)
            )
            if self.loop is not None and not self.loop.is_closed():
                self.logger.info("Rotating camera " + ("right" if right else "left"))
                future = asyncio.run_coroutine_threadsafe(self._send_command(cmd), self.loop)
                future.result(timeout=5)
            else:
                self.logger.error("Event loop not initialized or closed")
                raise RuntimeError("Event loop not initialized or closed")
        except asyncio.TimeoutError:
            self.logger.error("Camera rotation command timed out")
        except Exception as e:
            self.logger.error("Error rotating camera: " + str(e))
            raise

    def _zoom_camera(self, zoom_in: bool) -> None:
        """Zoom the camera in or out"""
        if not self.ws_connected.is_set():
            self.logger.error("Cannot zoom camera - WebSocket not connected")
            return
        
        try:
            cmd = Command(
                action=ActionType.CAMERA_ZOOM,
                data=CameraZoomData(in_=zoom_in)
            )
            if self.loop is not None and not self.loop.is_closed():
                self.logger.info("Zooming camera " + ("in" if zoom_in else "out"))
                future = asyncio.run_coroutine_threadsafe(self._send_command(cmd), self.loop)
                future.result(timeout=5)
            else:
                self.logger.error("Event loop not initialized or closed")
                raise RuntimeError("Event loop not initialized or closed")
        except asyncio.TimeoutError:
            self.logger.error("Camera zoom command timed out")
        except Exception as e:
            self.logger.error("Error zooming camera: " + str(e))
            raise

    def _execute_action(self, action: Action):
        """Execute the given action in the game with rate limiting"""
        current_time = time.time()
        
        # Clean up old timestamps (older than our window)
        self.action_timestamps = [t for t in self.action_timestamps if current_time - t < self.action_window]
        
        # Calculate current action rate
        if self.action_timestamps:
            window_duration = current_time - min(self.action_timestamps)
            current_rate = len(self.action_timestamps) / window_duration if window_duration > 0 else float('inf')
            target_rate = self.MAX_ACTIONS_PER_MINUTE / 60.0  # Convert to actions per second
            
            # If we're acting too fast, calculate delay needed to maintain target rate
            if current_rate > target_rate:
                ideal_delay = 1.0 / target_rate
                required_delay = max(
                    ideal_delay - (current_time - self.last_action_time),
                    self.MIN_ACTION_DELAY
                )
                self.logger.debug("Rate limiting: current rate " + str(current_rate) + "/s, target " + str(target_rate) + "/s, waiting " + str(required_delay) + "s")
                time.sleep(required_delay)
        
        # Ensure minimum delay between actions
        time_since_last_action = current_time - self.last_action_time
        if time_since_last_action < self.MIN_ACTION_DELAY:
            sleep_time = self.MIN_ACTION_DELAY - time_since_last_action
            self.logger.debug("Enforcing minimum delay: waiting " + str(sleep_time) + "s")
            time.sleep(sleep_time)
        
        # Record this action
        self.last_action_time = time.time()
        self.action_timestamps.append(self.last_action_time)
        
        if not self.current_state:
            self.logger.warning("No state available, skipping action")
            return

        if action == Action.DO_NOTHING:
            if self.current_state.get('inCombat', False):
                self.logger.info("In combat, waiting...")
            else:
                self.logger.info("Doing nothing...")
            return

        try:
            # Movement actions
            if action in [Action.MOVE_FORWARD, Action.MOVE_BACKWARD, Action.MOVE_LEFT, Action.MOVE_RIGHT]:
                if not all(key in self.current_state for key in ['playerLocation']):
                    self.logger.warning("Missing player location data")
                    return
                    
                current_x = self.current_state['playerLocation']['x']
                current_y = self.current_state['playerLocation']['y']
                
                dx, dy = {
                    Action.MOVE_FORWARD: (0, 3),
                    Action.MOVE_BACKWARD: (0, -3),
                    Action.MOVE_LEFT: (-3, 0),
                    Action.MOVE_RIGHT: (3, 0)
                }[action]
                
                target_x = current_x + dx
                target_y = current_y + dy
                
                self.logger.info("Moving " + action.name.split('_')[1].lower() + " from (" + str(current_x) + ", " + str(current_y) + ") to (" + str(target_x) + ", " + str(target_y) + ")")
                
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._send_command(Command(
                            action=ActionType.MOVE_AND_CLICK,
                            data=MoveAndClickData(
                                targetType=TargetType.COORDINATES,
                                action="Move",
                                x=target_x,
                                y=target_y
                            )
                        )),
                        self.loop
                    )
                else:
                    self.logger.error("Event loop not available for sending command")
                
            elif action == Action.ATTACK_NEAREST:
                target = self._find_nearest_goblin()
                if target:
                    self.logger.info("Attacking " + target.get('name', 'Unknown') + " at (" + str(target['location']['x']) + ", " + str(target['location']['y']) + ")")
                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._send_command(Command(
                                action=ActionType.MOVE_AND_CLICK,
                                data=MoveAndClickData(
                                    targetType=TargetType.NPC,
                                    action="Attack",
                                    npcId=target['id']
                                )
                            )),
                            self.loop
                        )
                    else:
                        self.logger.error("Event loop not available for sending command")
                else:
                    self.logger.warning("No valid target found for attack")
                    
            elif action == Action.ROTATE_LEFT:
                self.logger.info("Rotating camera left")
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._send_command(Command(
                            action=ActionType.CAMERA_ROTATE,
                            data=CameraRotateData(right=False)
                        )),
                        self.loop
                    )
                else:
                    self.logger.error("Event loop not available for sending command")
                
            elif action == Action.ROTATE_RIGHT:
                self.logger.info("Rotating camera right")
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._send_command(Command(
                            action=ActionType.CAMERA_ROTATE,
                            data=CameraRotateData(right=True)
                        )),
                        self.loop
                    )
                else:
                    self.logger.error("Event loop not available for sending command")
                
            elif action == Action.ZOOM_IN:
                self.logger.info("Zooming camera in")
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._send_command(Command(
                            action=ActionType.CAMERA_ZOOM,
                            data=CameraZoomData(in_=True)
                        )),
                        self.loop
                    )
                else:
                    self.logger.error("Event loop not available for sending command")
                
            elif action == Action.ZOOM_OUT:
                self.logger.info("Zooming camera out")
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self._send_command(Command(
                            action=ActionType.CAMERA_ZOOM,
                            data=CameraZoomData(in_=False)
                        )),
                        self.loop
                    )
                else:
                    self.logger.error("Event loop not available for sending command")

            elif action == Action.TAKE_NEAREST:
                best_item = self._find_best_ground_item()
                if best_item:
                    self.logger.info("Picking up " + best_item['name'] + " worth " + str(best_item['gePrice']) + " gp at (" + str(best_item['location']['x']) + ", " + str(best_item['location']['y']) + ")")
                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._send_command(Command(
                                action=ActionType.MOVE_AND_CLICK,
                                data=MoveAndClickData(
                                    targetType=TargetType.OBJECT,
                                    action="Take",
                                    name=best_item['name'],
                                    x=best_item['location']['x'],
                                    y=best_item['location']['y']
                                )
                            )),
                            self.loop
                        )
                    else:
                        self.logger.error("Event loop not available for sending command")
                else:
                    self.logger.warning("No valuable items found nearby")
                    
        except Exception as e:
            self.logger.error("Error executing action " + str(action) + ": " + str(e))
    
    def _find_nearest_goblin(self) -> Optional[Dict]:
        """Find the nearest goblin that can be attacked"""
        if not self.current_state or 'npcs' not in self.current_state:
            return None
            
        player_pos = np.array([
            self.current_state['playerLocation']['x'],
            self.current_state['playerLocation']['y']
        ])
        
        nearest_goblin = None
        min_distance = float('inf')
        
        for npc in self.current_state['npcs']:
            if (npc.get('name', '').lower() == 'goblin' and 
                not npc.get('interacting') and 
                npc.get('combatLevel', 0) > 0):
                
                npc_pos = np.array([npc['location']['x'], npc['location']['y']])
                distance = float(np.linalg.norm(player_pos - npc_pos))
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_goblin = npc
        
        return nearest_goblin
    
    def _find_consumable_item(self) -> Optional[Dict]:
        """Find a food or potion item in the inventory"""
        if not self.current_state or 'inventory' not in self.current_state:
            return None
            
        consumables = [
            'Lobster', 'Swordfish', 'Shark', 'Monkfish',  # Foods
            'Prayer potion', 'Super restore', 'Saradomin brew'  # Potions
        ]
        
        for item in self.current_state['inventory']:
            if any(consumable.lower() in item['name'].lower() for consumable in consumables):
                return item
        
        return None
    
    def _find_best_ground_item(self) -> Optional[Dict]:
        """Find the most valuable nearby ground item that's worth picking up"""
        if not self.current_state or 'groundItems' not in self.current_state:
            return None

        # Define minimum value thresholds
        MIN_GE_VALUE = 100  # Minimum GE value to bother picking up
        MIN_HA_VALUE = 50   # Minimum HA value to bother picking up

        best_item = None
        best_value = 0

        for item in self.current_state['groundItems']:
            # Calculate total value
            total_ge_value = item['gePrice'] * item['quantity']
            total_ha_value = item['haPrice'] * item['quantity']
            
            # Skip if not worth picking up
            if total_ge_value < MIN_GE_VALUE and total_ha_value < MIN_HA_VALUE:
                continue

            # Calculate a score based on value and distance
            value = max(total_ge_value, total_ha_value)
            distance_penalty = item['distance'] * 10  # 10gp penalty per tile of distance
            score = value - distance_penalty

            if score > best_value:
                best_value = score
                best_item = item

        return best_item
    
    def _state_to_observation(self, state: Dict) -> Dict:
        """Convert RuneLite state to gym observation"""
        if not state:
            # Return zero-filled observation if state is None
            return self._get_empty_observation()
            
        # Helper function to safely get nested values
        def safe_get(d: Dict, *keys, default=0):
            for key in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(key, default)
            return d if d is not None else default

        # Helper function to normalize values
        def normalize(value, min_val, max_val):
            if not isinstance(value, (int, float)) or np.isnan(value):
                return 0.0
            return np.clip((value - min_val) / (max_val - min_val), 0, 1)

        observation = {
            'player_position': np.array([
                safe_get(state, 'playerLocation', 'x', default=0),
                safe_get(state, 'playerLocation', 'y', default=0),
                safe_get(state, 'playerLocation', 'plane', default=0)
            ], dtype=np.float32),
            
            'player_combat_stats': np.array([
                safe_get(state, 'skills', 'ATTACK', 'level', default=1),
                safe_get(state, 'skills', 'STRENGTH', 'level', default=1),
                safe_get(state, 'skills', 'DEFENCE', 'level', default=1),
                safe_get(state, 'skills', 'RANGED', 'level', default=1),
                safe_get(state, 'skills', 'MAGIC', 'level', default=1),
                safe_get(state, 'skills', 'HITPOINTS', 'level', default=1),
                safe_get(state, 'skills', 'PRAYER', 'level', default=1)
            ], dtype=np.int32),
            
            'player_health': np.array([max(1, safe_get(state, 'playerHealth', default=1))], dtype=np.int32),
            'player_prayer': np.array([max(0, safe_get(state, 'playerPrayer', default=0))], dtype=np.int32),
            'player_run_energy': np.array([
                normalize(safe_get(state, 'playerRunEnergy', default=100.0), 0, 100)
            ], dtype=np.float32),
            
            'npcs': np.zeros((10, 8), dtype=np.float32),
            'inventory': np.zeros((28, 3), dtype=np.float32),
            'skills': np.ones(23, dtype=np.int32),
            'ground_items': np.zeros((20, 8), dtype=np.float32),
            'ground_items_total_ge_value': np.array([0], dtype=np.float32),
            'ground_items_total_ha_value': np.array([0], dtype=np.float32),
            'ground_items_nearest_distance': np.array([0], dtype=np.float32),
            'interfaces_open': np.array([1 if self.interfaces_open else 0], dtype=np.int32),
            'path_obstructed': np.array([1 if self.path_obstructed else 0], dtype=np.int32),
            'current_chunk': np.array([0, 0], dtype=np.int32),
            'visited_chunks_count': np.array([0], dtype=np.int32),
            'nearby_areas': np.zeros((9, 5), dtype=np.float32),
            'exploration_score': np.array([0.0], dtype=np.float32)
        }
        
        # Process NPCs with bounds checking
        for i, npc in enumerate(state.get('npcs', [])[:10]):
            if not isinstance(npc, dict):
                continue
            observation['npcs'][i] = np.array([
                normalize(safe_get(npc, 'localX', default=0), -2048, 2048),
                normalize(safe_get(npc, 'localY', default=0), -2048, 2048),
                normalize(safe_get(npc, 'location', 'plane', default=0), 0, 3),
                normalize(safe_get(npc, 'id', default=0), 0, 32767),
                normalize(safe_get(npc, 'animation', default=0), 0, 32767),
                normalize(safe_get(npc, 'health', default=0), 0, 100),
                normalize(safe_get(npc, 'combatLevel', default=0), 0, 1000),
                1.0 if safe_get(npc, 'interacting', default=False) else 0.0
            ], dtype=np.float32)
        
        # Process inventory with bounds checking
        for i, item in enumerate(state.get('inventory', [])[:28]):
            if not isinstance(item, dict):
                continue
            observation['inventory'][i] = np.array([
                normalize(safe_get(item, 'id', default=0), 0, 32767),
                normalize(safe_get(item, 'quantity', default=0), 0, 2147483647),
                0.0  # Reserved for future use
            ], dtype=np.float32)
        
        # Process skills with bounds checking
        for i, (_, skill) in enumerate(state.get('skills', {}).items()):
            if i >= 23 or not isinstance(skill, dict):
                continue
            observation['skills'][i] = max(1, min(99, safe_get(skill, 'level', default=1)))
        
        # Process ground items with bounds checking
        nearest_distance = float('inf')
        for i, item in enumerate(state.get('groundItems', [])[:20]):
            if not isinstance(item, dict):
                continue
            observation['ground_items'][i] = np.array([
                normalize(safe_get(item, 'id', default=0), 0, 32767),
                normalize(safe_get(item, 'quantity', default=0), 0, 2147483647),
                normalize(safe_get(item, 'gePrice', default=0), 0, 2147483647),
                normalize(safe_get(item, 'haPrice', default=0), 0, 2147483647),
                normalize(safe_get(item, 'distance', default=0), 0, 100),
                1.0 if safe_get(item, 'tradeable', default=False) else 0.0,
                1.0 if safe_get(item, 'stackable', default=False) else 0.0,
                1.0 if safe_get(item, 'isNote', default=False) else 0.0
            ], dtype=np.float32)
            
            distance = safe_get(item, 'distance', default=float('inf'))
            if distance < nearest_distance:
                nearest_distance = distance
        
        observation['ground_items_nearest_distance'] = np.array([
            normalize(nearest_distance, 0, 100)
        ], dtype=np.float32)
        
        # Calculate total values with bounds checking
        total_ge_value = sum(
            safe_get(item, 'gePrice', default=0) * safe_get(item, 'quantity', default=0)
            for item in state.get('groundItems', [])
        )
        total_ha_value = sum(
            safe_get(item, 'haPrice', default=0) * safe_get(item, 'quantity', default=0)
            for item in state.get('groundItems', [])
        )
        
        observation['ground_items_total_ge_value'] = np.array([
            normalize(total_ge_value, 0, 2147483647)
        ], dtype=np.float32)
        observation['ground_items_total_ha_value'] = np.array([
            normalize(total_ha_value, 0, 2147483647)
        ], dtype=np.float32)
        
        # Process exploration data with bounds checking
        exploration_data = state.get('exploration', {})
        current_chunk = safe_get(exploration_data, 'currentChunk', default={})
        observation['current_chunk'] = np.array([
            safe_get(current_chunk, 'x', default=0),
            safe_get(current_chunk, 'y', default=0)
        ], dtype=np.int32)
        
        observation['visited_chunks_count'] = np.array([
            safe_get(exploration_data, 'visitedChunks', default=0)
        ], dtype=np.int32)
        
        # Process nearby areas with bounds checking
        for i, dx in enumerate(range(-1, 2)):
            for j, dy in enumerate(range(-1, 2)):
                area_key = f"{dx},{dy}"
                area_data = safe_get(exploration_data, 'nearbyAreas', area_key, default={})
                if area_data:
                    idx = i * 3 + j
                    observation['nearby_areas'][idx] = np.array([
                        normalize(safe_get(area_data, 'score', default=0), 0, 1),
                        normalize(safe_get(area_data, 'npcDensity', default=0), 0, 10),
                        normalize(safe_get(area_data, 'resourceDensity', default=0), 0, 5),
                        normalize(safe_get(area_data, 'averageNpcLevel', default=0), 0, 100),
                        normalize(safe_get(area_data, 'averageItemValue', default=0), 0, 10000)
                    ], dtype=np.float32)

        # Calculate exploration score
        observation['exploration_score'] = np.array([
            normalize(len(self.visited_areas), 0, 100)
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, state: Optional[StateDict]) -> float:
        """Calculate the reward based on the current state"""
        reward: float = 0.0
        
        if not state:
            return reward
            
        # Combat rewards
        if self.task == "combat":
            # Experience rewards
            current_combat_exp = int(state.get('totalCombatExp', 0))
            if current_combat_exp > self.last_combat_exp:
                exp_gain = current_combat_exp - self.last_combat_exp
                exp_reward = float(exp_gain) * 0.01  # Scale the experience to a reasonable reward
                reward += exp_reward
                self.logger.info(f"Reward +{exp_reward:.1f} for gaining {exp_gain} combat experience")
            self.last_combat_exp = current_combat_exp
            
            # Damage rewards
            if state.get('lastHitsplat', 0) > 0:
                damage_reward = float(state['lastHitsplat']) * 0.5
                reward += damage_reward
                self.logger.info(f"Reward +{damage_reward:.1f} for dealing {state['lastHitsplat']} damage")
            
            # Reward for killing NPCs
            if state.get('npcKilled', False):
                kill_reward = 10.0
                reward += kill_reward
                self.logger.info(f"Reward +{kill_reward:.1f} for killing NPC")
            
            # Penalty for taking damage
            if state.get('playerLastHit', 0) > 0:
                damage_penalty = float(-state['playerLastHit']) * 0.3
                reward += damage_penalty
                self.logger.info(f"Reward {damage_penalty:.1f} for taking {state['playerLastHit']} damage")
            
            # Small reward for being in combat
            if state.get('inCombat', False):
                combat_reward = 0.1
                reward += combat_reward
                self.logger.info(f"Reward +{combat_reward:.1f} for being in combat")
                
                # Encourage doing nothing while in combat
                if state.get('lastAction') == Action.DO_NOTHING:
                    patience_reward = 0.2
                    reward += patience_reward
                    self.logger.info(f"Reward +{patience_reward:.1f} for being patient in combat")
            
            # Penalty for low health
            health_ratio = float(state.get('playerHealth', 0)) / float(state.get('playerMaxHealth', 1))
            if health_ratio < 0.3:
                health_penalty = -0.5
                reward += health_penalty
                self.logger.info(f"Reward {health_penalty:.1f} for low health ({health_ratio:.1%})")

            # Penalty for path obstruction
            if self.path_obstructed:
                obstruction_penalty = -0.3
                reward += obstruction_penalty
                self.logger.info(f"Reward {obstruction_penalty:.1f} for path obstruction")

            # Penalty for having interfaces open
            if self.interfaces_open:
                interface_penalty = -0.2
                reward += interface_penalty
                self.logger.info(f"Reward {interface_penalty:.1f} for having interfaces open")
            
            # Reward for valuable drops
            if 'groundItems' in state:
                for item in state['groundItems']:
                    if item.get('distance', float('inf')) <= 1:  # Item is at our feet
                        value = max(
                            float(item.get('gePrice', 0)), 
                            float(item.get('haPrice', 0))
                        ) * float(item.get('quantity', 0))
                        if value > 1000:  # Significant value threshold
                            drop_reward = float(value) * 0.001  # 0.1% of item value as reward
                            reward += drop_reward
                            self.logger.info(f"Reward +{drop_reward:.1f} for valuable drop: {item['name']} ({value}gp)")
            
            # Log total reward
            if reward != 0:
                self.logger.info(f"Total reward this step: {reward:.2f}")
        
        # Add exploration rewards
        exploration_data = state.get('exploration', {})
        current_chunk = exploration_data.get('currentChunk', {})
        chunk_coords = (
            int(current_chunk.get('x', 0)), 
            int(current_chunk.get('y', 0))
        )
        
        # Reward for discovering new chunks
        if chunk_coords not in self.visited_areas:
            chunk_reward = 5.0
            reward += chunk_reward
            self.visited_areas.add(chunk_coords)
            self.logger.info(f"Reward +{chunk_reward:.1f} for discovering new chunk")
        
        # Reward based on area quality
        nearby_areas = exploration_data.get('nearbyAreas', {})
        current_area = nearby_areas.get("0,0", {})
        if current_area:
            area_score = float(current_area.get('score', 0))
            if area_score > 0.7:
                quality_reward = 10.0
                reward += quality_reward
                self.logger.info(f"Reward +{quality_reward:.1f} for being in high-value area")
            elif area_score > 0.4:
                quality_reward = 5.0
                reward += quality_reward
                self.logger.info(f"Reward +{quality_reward:.1f} for being in decent area")
        
        return reward
    
    def _is_episode_done(self, state: Dict) -> bool:
        """Determine if the episode should end"""
        if state is None:
            return False
            
        if self.task == "combat":
            return (
                state.get('playerHealth', 0) <= 0 or  # Player died
                len(state.get('npcs', [])) == 0  # No NPCs nearby
            )
        return False
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_state = None
        self.path_obstructed = False
        self.interfaces_open = False
        self.last_combat_exp = 0
        
        # Send reset command to RuneLite with proper format
        self._move_and_click(
            target_type=TargetType.COORDINATES,
            action="Move",
            x=0,
            y=0
        )
        
        # Wait for initial state
        timeout = 0
        while not self.current_state and timeout < 50:  # Wait up to 5 seconds
            time.sleep(0.1)
            timeout += 1
            
        if not self.current_state:
            self.logger.warning("No initial state received")
            return self._get_empty_observation(), {}
            
        return self._state_to_observation(self.current_state), {}
    
    def render(self):
        """Render is handled by RuneLite client"""
        pass
    
    def close(self):
        """Clean up resources"""
        try:
            # Signal WebSocket to close
            self.ws_should_close.set()
            
            # Close WebSocket connection
            if self.ws is not None:
                if self.loop and not self.loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            
            # Wait for WebSocket thread to finish
            if self.ws_task and self.ws_task.is_alive():
                self.ws_task.join(timeout=5)
            
        except Exception as e:
            self.logger.error("Error during cleanup: " + str(e))
        finally:
            # Reset state
            self.ws = None
            self.loop = None
            self.ws_task = None
            self.ws_connected.clear()
            self.ws_should_close.clear()
    
    def _find_best_unexplored_area(self) -> Tuple[int, int]:
        """Find the best unexplored area in the 3x3 grid around the player"""
        if not self.current_state:
            return 0, 0

        exploration_data = self.current_state.get('exploration', {})
        nearby_areas = exploration_data.get('nearbyAreas', {})
        
        best_score = -1
        best_direction = (0, 0)
        current_time = time.time()

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                area_key = f"{dx},{dy}"
                area_data = nearby_areas.get(area_key, {})
                if area_data:
                    area_score = area_data.get('score', 0)
                    area_coord = (dx, dy)
                    
                    # Apply cooldown penalty
                    last_visit = self.last_exploration_time.get(area_coord, 0)
                    if current_time - last_visit < self.exploration_cooldown:
                        area_score *= 0.5  # Reduce score for recently visited areas
                    
                    # Bonus for unexplored areas
                    if area_coord not in self.visited_areas:
                        area_score *= 1.5
                    
                    if area_score > best_score:
                        best_score = area_score
                        best_direction = area_coord

        return best_direction

    def _get_empty_observation(self) -> Dict[str, np.ndarray]:
        """Return an empty observation with zeros"""
        return {
            'player_position': np.zeros(3, dtype=np.float32),
            'player_combat_stats': np.ones(7, dtype=np.int32),  # Stats start at 1
            'player_health': np.ones(1, dtype=np.int32),  # Health starts at 1
            'player_prayer': np.zeros(1, dtype=np.int32),
            'player_run_energy': np.zeros(1, dtype=np.float32),
            'npcs': np.zeros((10, 8), dtype=np.float32),
            'inventory': np.zeros((28, 3), dtype=np.float32),
            'skills': np.ones(23, dtype=np.int32),  # Skills start at 1
            'ground_items': np.zeros((20, 8), dtype=np.float32),
            'ground_items_total_ge_value': np.zeros(1, dtype=np.float32),
            'ground_items_total_ha_value': np.zeros(1, dtype=np.float32),
            'ground_items_nearest_distance': np.zeros(1, dtype=np.float32),
            'interfaces_open': np.zeros(1, dtype=np.int32),
            'path_obstructed': np.zeros(1, dtype=np.int32),
            'current_chunk': np.zeros(2, dtype=np.int32),
            'visited_chunks_count': np.zeros(1, dtype=np.int32),
            'nearby_areas': np.zeros((9, 5), dtype=np.float32),
            'exploration_score': np.zeros(1, dtype=np.float32)
        }

    def step(self, action: Action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        # Rate limit actions
        self._rate_limit_action()
        
        # Execute the action
        self._execute_action(action)
        
        # Get the new state
        observation = self._get_observation()
        
        # Calculate reward
        reward: float = 0.0
        if self.current_state is not None:
            reward = self._calculate_reward(self.current_state)
        
        # Check if episode is done
        terminated = False  # We don't terminate episodes in this environment
        truncated = False  # We don't truncate episodes
        
        # Get additional info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation (game state)"""
        if not self.current_state:
            # Return zero-filled observation if no state
            return self._get_empty_observation()
            
        return self._state_to_observation(self.current_state)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment"""
        return {
            'inventory': self.last_inventory if self.last_inventory is not None else [],
            'npcs': self.last_npcs if self.last_npcs is not None else [],
            'objects': self.last_objects if self.last_objects is not None else [],
            'combat_exp': self.last_combat_exp if self.last_combat_exp is not None else 0
        }

    def _rate_limit_action(self) -> None:
        """Rate limit actions to prevent overwhelming the client"""
        current_time = time.time()
        
        # Remove old timestamps outside the window
        self.action_timestamps = [t for t in self.action_timestamps if current_time - t <= self.action_window]
        
        # Calculate current action rate
        if self.action_timestamps:
            window_duration = current_time - min(self.action_timestamps)
            current_rate = len(self.action_timestamps) / window_duration if window_duration > 0 else float('inf')
            target_rate = self.MAX_ACTIONS_PER_MINUTE / 60.0
            
            if current_rate > target_rate:
                required_delay = (1.0 / target_rate) - (1.0 / current_rate)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Rate limiting: current rate {current_rate:.2f}/s, target {target_rate:.2f}/s, waiting {required_delay:.2f}s")
                time.sleep(required_delay)
        
        # Enforce minimum delay between actions
        time_since_last_action = current_time - self.last_action_time
        if time_since_last_action < self.MIN_ACTION_DELAY:
            sleep_time = self.MIN_ACTION_DELAY - time_since_last_action
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Enforcing minimum delay: waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Update timestamps
        self.action_timestamps.append(current_time)
        self.last_action_time = current_time 