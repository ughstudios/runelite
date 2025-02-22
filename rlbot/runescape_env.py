import gymnasium as gym
from gymnasium import spaces
import numpy as np
import websockets
import json
import threading
import time
import asyncio
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
import nest_asyncio
import logging
from rich.logging import RichHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rlbot.env")

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

class RuneScapeEnv(gym.Env):
    """
    Gymnasium Environment for RuneScape with enhanced capabilities
    """
    
    def __init__(self, websocket_url: str = "ws://localhost:43594", task: str = "combat"):
        super().__init__()
        
        logger.info(f"[bold]Initializing RuneScape Environment[/bold]")
        logger.info(f"WebSocket URL: {websocket_url}")
        logger.info(f"Task: {task}")
        
        # Define action spaces based on task
        self.task = task
        if task == "combat":
            self.action_space = spaces.Discrete(len(Action))
        else:
            raise ValueError(f"Unknown task: {task}")
        
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
            'ground_items': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 4), dtype=np.float32),
            'interfaces_open': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            'path_obstructed': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32)
        })
        
        # WebSocket setup
        self.ws = None
        self.ws_url = websocket_url
        self.current_state = None
        self.ws_task = None
        self.last_action_time = 0.0
        self.action_delay = 0.6  # Minimum delay between actions in seconds
        self.path_obstructed = False
        self.interfaces_open = False
        
        # Create event loop for async operations
        self.loop = asyncio.new_event_loop()
        self.connect_websocket()
    
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
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        # Rate limit actions
        current_time = time.time()
        if current_time - self.last_action_time < self.action_delay:
            time.sleep(self.action_delay - (current_time - self.last_action_time))
        
        # Convert action to game command - fix the off-by-one error
        self._execute_action(Action(action + 1))  # Actions are 1-based in the enum
        self.last_action_time = time.time()
        
        # Wait for next state update with timeout
        timeout = 0
        while not self.current_state and timeout < 50:  # Wait up to 5 seconds
            time.sleep(0.1)
            timeout += 1
            
        if not self.current_state:
            logger.warning("[yellow]No state update received[/yellow]")
            return self._get_empty_observation(), 0.0, True, False, {}
            
        # Convert state to observation
        observation = self._state_to_observation(self.current_state)
        
        # Calculate reward based on task
        reward = self._calculate_reward(self.current_state)
        
        # Check if episode is done
        terminated = self._is_episode_done(self.current_state)
        truncated = False  # We don't truncate episodes
        
        # Additional info
        info = {
            'player_health': observation['player_health'][0],
            'player_prayer': observation['player_prayer'][0],
            'nearby_npcs': len(self.current_state.get('npcs', []))
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_empty_observation(self) -> Dict:
        """Return an empty observation with default values"""
        return {
            'player_position': np.zeros(3, dtype=np.float32),
            'player_combat_stats': np.ones(7, dtype=np.int32),
            'player_health': np.array([1], dtype=np.int32),
            'player_prayer': np.array([0], dtype=np.int32),
            'player_run_energy': np.array([100.0], dtype=np.float32),
            'npcs': np.zeros((10, 8), dtype=np.float32),
            'inventory': np.zeros((28, 3), dtype=np.float32),
            'skills': np.ones(23, dtype=np.int32),
            'ground_items': np.zeros((10, 4), dtype=np.float32),
            'interfaces_open': np.array([0], dtype=np.int32),
            'path_obstructed': np.array([0], dtype=np.int32)
        }
    
    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """Linear interpolation between two values"""
        return start + (end - start) * progress

    def _execute_action(self, action: Action):
        """Execute the given action in the game"""
        if not self.current_state:
            logger.warning("[yellow]No state available, skipping action[/yellow]")
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
            
            self.send_command({
                "action": "move",
                "data": {
                    "x": target_x,
                    "y": target_y
                }
            })
            logger.info(f"[blue]Moving {action.name.split('_')[1].lower()}[/blue]")
            
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
            'ground_items': np.zeros((10, 4), dtype=np.float32),
            'interfaces_open': np.array([1 if self.interfaces_open else 0], dtype=np.int32),
            'path_obstructed': np.array([1 if self.path_obstructed else 0], dtype=np.int32)
        }
        
        # Process NPCs
        for i, npc in enumerate(state.get('npcs', [])[:10]):
            observation['npcs'][i] = [
                npc.get('localX', 0),
                npc.get('localY', 0),
                npc.get('location', {}).get('plane', 0),
                npc.get('id', 0),
                npc.get('animation', 0),
                npc.get('health', 0),
                npc.get('combatLevel', 0),
                1 if npc.get('interacting') else 0
            ]
        
        # Process inventory
        for i, item in enumerate(state.get('inventory', [])[:28]):
            observation['inventory'][i] = [
                item.get('id', 0),
                item.get('quantity', 0),
                0  # Reserved for future use
            ]
        
        # Process all skills
        for i, (_, skill) in enumerate(state.get('skills', {}).items()):
            if i < 23:
                observation['skills'][i] = skill.get('level', 1)
        
        return observation
    
    def _calculate_reward(self, state: Dict) -> float:
        """Calculate the reward based on the current state"""
        reward = 0.0
        
        # Combat rewards
        if self.task == "combat":
            # Track experience gains
            if hasattr(self, 'last_combat_exp'):
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
            else:
                self.last_combat_exp = sum(
                    state.get('skills', {}).get(skill, {}).get('experience', 0)
                    for skill in ['ATTACK', 'STRENGTH', 'DEFENCE', 'HITPOINTS']
                )
            
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
            
            # Log total reward
            if reward != 0:
                logger.info(f"[bold]Total reward this step: {reward:.2f}[/bold]")
        
        return reward
    
    def _is_episode_done(self, state: Dict) -> bool:
        """Determine if the episode should end"""
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
        self.last_action_time = 0.0
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
            self.ws_task.join(timeout=5)
        
        if self.loop and self.loop.is_running():
            self.loop.stop()
            self.loop.close() 