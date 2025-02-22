import gymnasium as gym
from gymnasium import spaces
import numpy as np
import websockets
import json
import threading
import time
import asyncio
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum, auto
import nest_asyncio
import logging
from rich.logging import RichHandler
import numpy.typing as npt

# Set up logging
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler('rlbot.log', mode='a')
    ]
)
logger = logging.getLogger("rlbot.env")
logger.info("[bold]Starting RLBot Environment Logger[/bold]")

# Allow nested event loops
nest_asyncio.apply()

class Action(Enum):
    MOVE_NORTH = auto()
    MOVE_SOUTH = auto()
    MOVE_EAST = auto()
    MOVE_WEST = auto()
    ATTACK = auto()
    CAMERA_ROTATE_LEFT = auto()
    CAMERA_ROTATE_RIGHT = auto()
    CAMERA_ZOOM_IN = auto()
    CAMERA_ZOOM_OUT = auto()
    DO_NOTHING = auto()
    CLOSE_INTERFACE = auto()  # New action for closing interfaces
    PICKUP_ITEM = auto()  # New action for picking up items
    EXPLORE_BEST_AREA = auto()  # New action for exploration

class RuneScapeEnv(gym.Env):
    """
    Gymnasium Environment for RuneScape with enhanced capabilities
    """
    
    def __init__(self, websocket_url: str = "ws://localhost:43594", task: str = "combat"):
        super().__init__()
        
        logger.info(f"[bold]Initializing RuneScape Environment[/bold]")
        logger.info(f"WebSocket URL: {websocket_url}")
        logger.info(f"Task: {task}")
        
        # Action rate limiting
        self.GAME_TICK = 0.6  # RuneScape game tick in seconds
        self.MIN_ACTION_DELAY = self.GAME_TICK  # Minimum time between actions
        self.MAX_ACTIONS_PER_MINUTE = 60  # Maximum actions per minute (roughly 1 action per second)
        self.action_timestamps = []  # Track recent action times
        self.last_action_time = 0
        
        # Define action spaces based on task
        self.task = task
        if task == "combat":
            self.action_space = spaces.Discrete(len(Action))
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Initialize combat tracking
        self.last_combat_exp: int = 0
        
        # Enhanced observation space
        self.observation_space = spaces.Dict({
            'player_position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'player_combat_stats': spaces.Box(low=0, high=99, shape=(7,), dtype=np.int32),
            'player_health': spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            'player_prayer': spaces.Box(low=0, high=99, shape=(1,), dtype=np.int32),
            'player_run_energy': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'npcs': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 8), dtype=np.float32),
            'inventory': spaces.Box(low=0, high=np.inf, shape=(28, 3), dtype=np.float32),
            'skills': spaces.Box(low=1, high=99, shape=(23,), dtype=np.int32),
            'ground_items': spaces.Box(low=-np.inf, high=np.inf, shape=(20, 8), dtype=np.float32),
            'ground_items_total_ge_value': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'ground_items_total_ha_value': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'ground_items_nearest_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'interfaces_open': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'path_obstructed': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'current_chunk': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32),
            'visited_chunks_count': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'nearby_areas': spaces.Box(low=0, high=1, shape=(9, 5), dtype=np.float32),
            'exploration_score': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # WebSocket setup
        self.ws = None
        self.ws_url = websocket_url
        self.current_state = None
        self.ws_task = None
        self.path_obstructed = False
        self.interfaces_open = False
        
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        self.connect_websocket()
        
        # Exploration tracking
        self.visited_areas: Set[Tuple[int, int]] = set()
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
    
    async def websocket_handler(self):
        """Handle WebSocket connection and messages"""
        logger.info("[yellow]Attempting to connect to RuneLite...[/yellow]")
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self.ws = websocket
                    logger.info("[green]Connected to RuneLite successfully![/green]")
                    
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            # Handle status updates
                            if isinstance(data, dict) and 'status' in data:
                                if data['status'] == 'obstructed':
                                    self.path_obstructed = True
                                    logger.warning(f"[yellow]Path obstructed to {data.get('target', 'unknown location')}[/yellow]")
                                continue

                            # Handle regular state updates
                            self.current_state = data
                            
                            # Update interface state
                            self.interfaces_open = data.get('interfacesOpen', False)
                            
                            # Log significant state changes
                            if self._has_significant_changes(data):
                                self._log_state_changes(data)
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.error("[red]WebSocket connection closed[/red]")
                            break
                        except json.JSONDecodeError as e:
                            logger.error(f"[red]Failed to parse message: {e}[/red]")
                        except Exception as e:
                            logger.error(f"[red]WebSocket error: {e}[/red]")
                            break
                            
            except Exception as e:
                retry_count += 1
                logger.error(f"[red]Failed to connect to RuneLite (attempt {retry_count}/{max_retries}): {e}[/red]")
                if retry_count < max_retries:
                    await asyncio.sleep(5)  # Wait before retrying
                else:
                    logger.error("[red]Max retries reached, giving up on WebSocket connection[/red]")
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
            logger.info("[bold]Initial state received[/bold]")
            return
            
        # Log health changes
        if state.get('playerHealth') != self.current_state.get('playerHealth'):
            logger.info(f"Health changed: {self.current_state.get('playerHealth')} -> {state.get('playerHealth')}")
        
        # Log prayer changes
        if state.get('playerPrayer') != self.current_state.get('playerPrayer'):
            logger.info(f"Prayer changed: {self.current_state.get('playerPrayer')} -> {state.get('playerPrayer')}")
        
        # Log NPC changes
        new_npcs = len(state.get('npcs', []))
        old_npcs = len(self.current_state.get('npcs', []))
        if new_npcs != old_npcs:
            logger.info(f"[yellow]NPCs in range changed: {old_npcs} -> {new_npcs}[/yellow]")
        
        # Log animation changes
        if state.get('playerAnimation') != self.current_state.get('playerAnimation'):
            logger.info(f"Animation changed: {self.current_state.get('playerAnimation')} -> {state.get('playerAnimation')}")
    
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
            logger.error("[red]Failed to connect to RuneLite WebSocket after 10 seconds[/red]")
            raise ConnectionError("Failed to connect to RuneLite WebSocket")
    
    def send_command(self, command):
        """Send command through WebSocket synchronously"""
        if not self.ws:
            logger.error("[red]WebSocket not connected, cannot send command[/red]")
            return
        
        logger.debug(f"[blue]Sending command: {json.dumps(command)}[/blue]")
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(command)),
                self.loop
            )
            future.result(timeout=5)  # Wait up to 5 seconds for the command to complete
            logger.debug("[green]Command sent successfully[/green]")
        except Exception as e:
            logger.error(f"[red]Failed to send command: {e}[/red]")
            # Try to reconnect if connection was lost
            if not self.ws.open:
                logger.info("[yellow]WebSocket connection lost, attempting to reconnect...[/yellow]")
                self.connect_websocket()
    
    def _wait_for_movement_completion(self) -> bool:
        """Wait for the player to stop moving or timeout"""
        start_time = time.time()
        self.stationary_count = 0
        
        while time.time() - start_time < self.movement_timeout:
            if not self.current_state:
                time.sleep(self.position_check_interval)
                continue
                
            current_pos = (
                self.current_state.get('playerLocation', {}).get('x', 0),
                self.current_state.get('playerLocation', {}).get('y', 0)
            )
            
            # Check if position hasn't changed
            if self.last_position == current_pos:
                self.stationary_count += 1
                if self.stationary_count >= self.stationary_threshold:
                    logger.info("[green]Movement completed[/green]")
                    self.is_moving = False
                    return True
            else:
                self.stationary_count = 0
                self.last_position = current_pos
            
            time.sleep(self.position_check_interval)
        
        logger.warning("[yellow]Movement timed out[/yellow]")
        self.is_moving = False
        return False

    def _execute_action(self, action: Action):
        """Execute the given action in the game with rate limiting"""
        current_time = time.time()
        
        # Clean up old timestamps (older than 1 minute)
        self.action_timestamps = [t for t in self.action_timestamps if current_time - t < 60]
        
        # Check if we've exceeded our actions per minute limit
        if len(self.action_timestamps) >= self.MAX_ACTIONS_PER_MINUTE:
            logger.info("[yellow]Action rate limit reached, waiting...[/yellow]")
            time.sleep(self.MIN_ACTION_DELAY)
            return
        
        # Ensure minimum delay between actions
        time_since_last_action = current_time - self.last_action_time
        if time_since_last_action < self.MIN_ACTION_DELAY:
            sleep_time = self.MIN_ACTION_DELAY - time_since_last_action
            logger.debug(f"[blue]Waiting {sleep_time:.2f}s between actions[/blue]")
            time.sleep(sleep_time)
        
        # Record this action
        self.last_action_time = time.time()
        self.action_timestamps.append(self.last_action_time)
        
        if not self.current_state:
            logger.warning("[yellow]No state available, skipping action[/yellow]")
            return

        # Don't allow new movement commands while still moving
        if self.is_moving and action in [
            Action.MOVE_NORTH, Action.MOVE_SOUTH, 
            Action.MOVE_EAST, Action.MOVE_WEST,
            Action.EXPLORE_BEST_AREA
        ]:
            logger.info("[yellow]Still moving, skipping movement command[/yellow]")
            return

        if action == Action.DO_NOTHING:
            # Just log that we're waiting
            if self.current_state.get('inCombat', False):
                logger.info("[blue]In combat, waiting...[/blue]")
            else:
                logger.info("[blue]Doing nothing...[/blue]")
            return

        if action == Action.CLOSE_INTERFACE:
            if self.interfaces_open:
                self.send_command({
                    "action": "close_interface"
                })
                logger.info("[blue]Closing interface[/blue]")
            return

        # Check if interfaces are open before executing movement or combat actions
        if self.interfaces_open and action not in [Action.DO_NOTHING, Action.CLOSE_INTERFACE]:
            logger.info("[yellow]Interface open, should close first[/yellow]")
            return

        if action in [Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST]:
            # Calculate movement coordinates
            if not all(key in self.current_state for key in ['playerLocation']):
                return
                
            current_x = self.current_state['playerLocation']['x']
            current_y = self.current_state['playerLocation']['y']
            
            dx, dy = {
                Action.MOVE_NORTH: (0, 3),
                Action.MOVE_SOUTH: (0, -3),
                Action.MOVE_EAST: (3, 0),
                Action.MOVE_WEST: (-3, 0)
            }[action]
            
            target_x = current_x + dx
            target_y = current_y + dy
            
            self.is_moving = True
            self.last_position = (current_x, current_y)
            
            self.send_command({
                "action": "move",
                "data": {
                    "x": target_x,
                    "y": target_y
                }
            })
            logger.info(f"[blue]Moving {action.name.split('_')[1].lower()}[/blue]")
            
            # Wait for movement to complete
            self._wait_for_movement_completion()
            
        elif action == Action.ATTACK:
            # Find nearest attackable NPC
            target = self._find_nearest_goblin()
            if target:
                self.send_command({
                    "action": "attack",
                    "data": {
                        "targetId": target['id']
                    }
                })
                logger.info(f"[red]Attacking {target.get('name', 'Unknown')}[/red]")
            else:
                logger.warning("[yellow]No valid target found for attack[/yellow]")
                
        elif action == Action.CAMERA_ROTATE_LEFT:
            self.send_command({
                "action": "camera_rotate",
                "data": {
                    "right": False
                }
            })
            logger.info("[blue]Rotating camera left[/blue]")
            
        elif action == Action.CAMERA_ROTATE_RIGHT:
            self.send_command({
                "action": "camera_rotate",
                "data": {
                    "right": True
                }
            })
            logger.info("[blue]Rotating camera right[/blue]")
            
        elif action == Action.CAMERA_ZOOM_IN:
            self.send_command({
                "action": "camera_zoom",
                "data": {
                    "in": True
                }
            })
            logger.info("[blue]Zooming camera in[/blue]")
            
        elif action == Action.CAMERA_ZOOM_OUT:
            self.send_command({
                "action": "camera_zoom",
                "data": {
                    "in": False
                }
            })
            logger.info("[blue]Zooming camera out[/blue]")

        elif action == Action.PICKUP_ITEM:
            # Find most valuable nearby item
            best_item = self._find_best_ground_item()
            if best_item:
                self.send_command({
                    "action": "interact",
                    "data": {
                        "x": best_item['location']['x'],
                        "y": best_item['location']['y'],
                        "target": best_item['name'],
                        "action": "Take"
                    }
                })
                logger.info(f"[blue]Picking up {best_item['name']} worth {best_item['gePrice']} gp[/blue]")

        elif action == Action.EXPLORE_BEST_AREA:
            # Find the best direction to explore
            dx, dy = self._find_best_unexplored_area()
            
            if dx == 0 and dy == 0:
                logger.info("[yellow]No good exploration targets found, staying put[/yellow]")
                return
            
            # Calculate target position
            if not self.current_state or 'playerLocation' not in self.current_state:
                return
                
            current_x = self.current_state['playerLocation']['x']
            current_y = self.current_state['playerLocation']['y']
            
            # Move towards the chosen area
            target_x = current_x + (dx * 8)  # 8 tiles per chunk
            target_y = current_y + (dy * 8)
            
            self.is_moving = True
            self.last_position = (current_x, current_y)
            
            self.send_command({
                "action": "move",
                "data": {
                    "x": target_x,
                    "y": target_y
                }
            })
            
            # Update exploration tracking
            current_chunk = (dx, dy)
            self.visited_areas.add(current_chunk)
            self.last_exploration_time[current_chunk] = time.time()
            
            logger.info(f"[blue]Exploring new area in direction ({dx}, {dy})[/blue]")
            
            # Wait for movement to complete
            self._wait_for_movement_completion()
    
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
                distance = np.linalg.norm(player_pos - npc_pos)
                
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
        observation = {
            'player_position': np.array([
                state.get('playerLocation', {}).get('x', 0),
                state.get('playerLocation', {}).get('y', 0),
                state.get('playerLocation', {}).get('plane', 0)
            ], dtype=np.float32),
            'player_combat_stats': np.array([
                state.get('skills', {}).get('ATTACK', {}).get('level', 1),
                state.get('skills', {}).get('STRENGTH', {}).get('level', 1),
                state.get('skills', {}).get('DEFENCE', {}).get('level', 1),
                state.get('skills', {}).get('RANGED', {}).get('level', 1),
                state.get('skills', {}).get('MAGIC', {}).get('level', 1),
                state.get('skills', {}).get('HITPOINTS', {}).get('level', 1),
                state.get('skills', {}).get('PRAYER', {}).get('level', 1)
            ], dtype=np.int32),
            'player_health': np.array([state.get('playerHealth', 1)], dtype=np.int32),
            'player_prayer': np.array([state.get('playerPrayer', 0)], dtype=np.int32),
            'player_run_energy': np.array([state.get('playerRunEnergy', 100.0)], dtype=np.float32),
            'npcs': np.zeros((10, 8), dtype=np.float32),
            'inventory': np.zeros((28, 3), dtype=np.float32),
            'skills': np.ones(23, dtype=np.int32),
            'ground_items': np.zeros((20, 8), dtype=np.float32),
            'ground_items_total_ge_value': np.array([0], dtype=np.float32),
            'ground_items_total_ha_value': np.array([0], dtype=np.float32),
            'ground_items_nearest_distance': np.array([float('inf')], dtype=np.float32),
            'interfaces_open': np.array([1 if self.interfaces_open else 0], dtype=np.int32),
            'path_obstructed': np.array([1 if self.path_obstructed else 0], dtype=np.int32),
            'current_chunk': np.array([0, 0], dtype=np.int32),
            'visited_chunks_count': np.array([0], dtype=np.int32),
            'nearby_areas': np.zeros((9, 5), dtype=np.float32),
            'exploration_score': np.array([0.0], dtype=np.float32)
        }
        
        # Process NPCs
        for i, npc in enumerate(state.get('npcs', [])[:10]):
            observation['npcs'][i] = np.array([
                npc.get('localX', 0),
                npc.get('localY', 0),
                npc.get('location', {}).get('plane', 0),
                npc.get('id', 0),
                npc.get('animation', 0),
                npc.get('health', 0),
                npc.get('combatLevel', 0),
                1 if npc.get('interacting') else 0
            ], dtype=np.float32)
        
        # Process inventory
        for i, item in enumerate(state.get('inventory', [])[:28]):
            observation['inventory'][i] = np.array([
                item.get('id', 0),
                item.get('quantity', 0),
                0  # Reserved for future use
            ], dtype=np.float32)
        
        # Process all skills
        for i, (_, skill) in enumerate(state.get('skills', {}).items()):
            if i < 23:
                observation['skills'][i] = skill.get('level', 1)
        
        # Process ground items
        nearest_distance = float('inf')
        for i, item in enumerate(state.get('groundItems', [])[:20]):
            observation['ground_items'][i] = np.array([
                item.get('id', 0),
                item.get('quantity', 0),
                item.get('gePrice', 0),
                item.get('haPrice', 0),
                item.get('distance', 0),
                1 if item.get('tradeable', False) else 0,
                1 if item.get('stackable', False) else 0,
                1 if item.get('isNote', False) else 0
            ], dtype=np.float32)
            
            # Update nearest item distance
            distance = item.get('distance', float('inf'))
            if distance < nearest_distance:
                nearest_distance = distance
        
        observation['ground_items_nearest_distance'] = np.array([nearest_distance], dtype=np.float32)
        
        # Calculate total value
        total_ge_value = sum(item['gePrice'] * item['quantity'] for item in state.get('groundItems', []))
        total_ha_value = sum(item['haPrice'] * item['quantity'] for item in state.get('groundItems', []))
        
        observation['ground_items_total_ge_value'] = np.array([total_ge_value], dtype=np.float32)
        observation['ground_items_total_ha_value'] = np.array([total_ha_value], dtype=np.float32)
        
        # Process exploration data
        exploration_data = state.get('exploration', {})
        current_chunk = exploration_data.get('currentChunk', {})
        observation['current_chunk'] = np.array([
            int(current_chunk.get('x', 0)),
            int(current_chunk.get('y', 0))
        ], dtype=np.int32)
        
        observation['visited_chunks_count'] = np.array([
            exploration_data.get('visitedChunks', 0)
        ], dtype=np.int32)
        
        # Process nearby areas
        for i, dx in enumerate(range(-1, 2)):
            for j, dy in enumerate(range(-1, 2)):
                area_key = f"{dx},{dy}"
                area_data = exploration_data.get('nearbyAreas', {}).get(area_key, {})
                if area_data:
                    idx = i * 3 + j
                    observation['nearby_areas'][idx] = np.array([
                        float(area_data.get('score', 0)),
                        float(area_data.get('npcDensity', 0)) / 10.0,
                        float(area_data.get('resourceDensity', 0)) / 5.0,
                        float(area_data.get('averageNpcLevel', 0)) / 100.0,
                        float(area_data.get('averageItemValue', 0)) / 10000.0
                    ], dtype=np.float32)

        # Calculate exploration score
        observation['exploration_score'] = np.array([
            float(len(self.visited_areas)) / 100.0
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, state: Dict) -> float:
        """Calculate the reward based on the current state"""
        if state is None:
            return 0.0
            
        reward = 0.0
        
        # Combat rewards
        if self.task == "combat":
            # Track experience gains
            current_combat_exp = sum(
                state.get('skills', {}).get(skill, {}).get('experience', 0)
                for skill in ['ATTACK', 'STRENGTH', 'DEFENCE', 'HITPOINTS']
            )
            exp_gain = current_combat_exp - self.last_combat_exp
            if exp_gain > 0:
                exp_reward = exp_gain * 0.01  # Scale the experience to a reasonable reward
                reward += exp_reward
                logger.info(f"[green]Reward +{exp_reward:.1f} for gaining {exp_gain} combat experience[/green]")
            self.last_combat_exp = current_combat_exp
            
            # Reward for dealing damage
            if state.get('lastHitsplat', 0) > 0:
                damage_reward = state['lastHitsplat'] * 0.5
                reward += damage_reward
                logger.info(f"[green]Reward +{damage_reward:.1f} for dealing {state['lastHitsplat']} damage[/green]")
            
            # Reward for killing NPCs
            if state.get('npcKilled', False):
                kill_reward = 10.0
                reward += kill_reward
                logger.info(f"[green]Reward +{kill_reward:.1f} for killing NPC[/green]")
            
            # Penalty for taking damage
            if state.get('playerLastHit', 0) > 0:
                damage_penalty = -state['playerLastHit'] * 0.3
                reward += damage_penalty
                logger.info(f"[red]Reward {damage_penalty:.1f} for taking {state['playerLastHit']} damage[/red]")
            
            # Small reward for being in combat
            if state.get('inCombat', False):
                combat_reward = 0.1
                reward += combat_reward
                logger.info(f"[green]Reward +{combat_reward:.1f} for being in combat[/green]")
                
                # Encourage doing nothing while in combat
                if state.get('lastAction') == Action.DO_NOTHING:
                    patience_reward = 0.2
                    reward += patience_reward
                    logger.info(f"[green]Reward +{patience_reward:.1f} for being patient in combat[/green]")
            
            # Penalty for low health
            health_ratio = state.get('playerHealth', 0) / state.get('playerMaxHealth', 1)
            if health_ratio < 0.3:
                health_penalty = -0.5
                reward += health_penalty
                logger.info(f"[red]Reward {health_penalty:.1f} for low health ({health_ratio:.1%})[/red]")

            # Penalty for path obstruction
            if self.path_obstructed:
                obstruction_penalty = -0.3
                reward += obstruction_penalty
                logger.info(f"[red]Reward {obstruction_penalty:.1f} for path obstruction[/red]")

            # Penalty for having interfaces open
            if self.interfaces_open:
                interface_penalty = -0.2
                reward += interface_penalty
                logger.info(f"[red]Reward {interface_penalty:.1f} for having interfaces open[/red]")
            
            # Reward for valuable drops
            if 'groundItems' in state:
                for item in state['groundItems']:
                    if item.get('distance', float('inf')) <= 1:  # Item is at our feet
                        value = max(item.get('gePrice', 0), item.get('haPrice', 0)) * item.get('quantity', 0)
                        if value > 1000:  # Significant value threshold
                            drop_reward = value * 0.001  # 0.1% of item value as reward
                            reward += drop_reward
                            logger.info(f"[green]Reward +{drop_reward:.1f} for valuable drop: {item['name']} ({value}gp)[/green]")
            
            # Log total reward
            if reward != 0:
                logger.info(f"[bold]Total reward this step: {reward:.2f}[/bold]")
        
        # Add exploration rewards
        exploration_data = state.get('exploration', {})
        current_chunk = exploration_data.get('currentChunk', {})
        chunk_coords = (int(current_chunk.get('x', 0)), int(current_chunk.get('y', 0)))
        
        # Reward for discovering new chunks
        if chunk_coords not in self.visited_areas:
            chunk_reward = 5.0
            reward += chunk_reward
            self.visited_areas.add(chunk_coords)
            logger.info(f"[green]Reward +{chunk_reward:.1f} for discovering new chunk[/green]")
        
        # Reward based on area quality
        nearby_areas = exploration_data.get('nearbyAreas', {})
        current_area = nearby_areas.get("0,0", {})
        if current_area:
            area_score = current_area.get('score', 0)
            if area_score > 0.7:
                quality_reward = 10.0
                reward += quality_reward
                logger.info(f"[green]Reward +{quality_reward:.1f} for being in high-value area[/green]")
            elif area_score > 0.4:
                quality_reward = 5.0
                reward += quality_reward
                logger.info(f"[green]Reward +{quality_reward:.1f} for being in decent area[/green]")
        
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
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_state = None
        self.path_obstructed = False
        self.interfaces_open = False
        
        # Send reset command to RuneLite
        self.send_command({"action": "reset"})
        
        # Wait for initial state
        timeout = 0
        while not self.current_state and timeout < 50:  # Wait up to 5 seconds
            time.sleep(0.1)
            timeout += 1
            
        if not self.current_state:
            logger.warning("[yellow]No initial state received[/yellow]")
            return self._get_empty_observation(), {}
            
        return self._state_to_observation(self.current_state), {}
    
    def render(self):
        """Render is handled by RuneLite client"""
        pass
    
    def close(self):
        """Clean up resources"""
        if self.ws:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.ws.close(),
                    self.loop
                )
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"[red]Error closing WebSocket: {e}[/red]")
        
        if self.ws_task and self.ws_task.is_alive():
            try:
                self.ws_task.join(timeout=5)
            except Exception as e:
                logger.error(f"[red]Error joining WebSocket thread: {e}[/red]")

        if self.loop and self.loop.is_running():
            try:
                # Cancel all running tasks
                for task in asyncio.all_tasks(self.loop):
                    task.cancel()
                
                # Run the event loop one last time to process cancellations
                self.loop.run_until_complete(asyncio.sleep(0))
                
                self.loop.stop()
                
                # Close the loop if it's not already closed
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception as e:
                logger.error(f"[red]Error closing event loop: {e}[/red]")
                # If we can't close cleanly, just suppress the error
                pass
    
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

    def _get_empty_observation(self) -> Dict:
        """Return an empty observation with zeros"""
        return {key: np.zeros(space.shape, dtype=space.dtype) 
                for key, space in self.observation_space.spaces.items()} 

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        # Convert action index to Action enum
        try:
            action_enum = list(Action)[action]
        except IndexError:
            logger.error(f"[red]Invalid action index: {action}[/red]")
            action_enum = Action.DO_NOTHING

        # Execute the action
        self._execute_action(action_enum)

        # Calculate reward
        reward = self._calculate_reward(self.current_state) if self.current_state else 0.0

        # Check if episode is done
        terminated = self._is_episode_done(self.current_state) if self.current_state else False
        truncated = False  # We don't use truncation

        # Get observation
        observation = self._state_to_observation(self.current_state) if self.current_state else self._get_empty_observation()

        # Additional info
        info: Dict[str, Any] = {}
        if self.current_state:
            info['player_health'] = self.current_state.get('playerHealth', 0)
            info['in_combat'] = self.current_state.get('inCombat', False)
            info['npcs_nearby'] = len(self.current_state.get('npcs', []))

        return observation, reward, terminated, truncated, info 