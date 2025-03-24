from dotenv import load_dotenv
load_dotenv()

import os
import requests
import pyautogui
import asyncio
import time
import base64
import importlib.util
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from agents import Runner, Agent, ModelSettings, function_tool
import random


class Location(BaseModel):
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")
    plane: Optional[int] = Field(None, description="Plane/level")

class Health(BaseModel):
    current: int = Field(..., description="Current health points")
    maximum: int = Field(..., description="Maximum health points")

class Skill(BaseModel):
    level: int = Field(..., description="Current level")
    realLevel: int = Field(..., description="Base level without boosts")
    experience: int = Field(..., description="Total experience points")

class Player(BaseModel):
    name: Optional[str] = Field(None, description="Player's name")
    location: Optional[Location] = Field(None, description="Player's location")
    health: Optional[Health] = Field(None, description="Player's health")
    maxHealth: Optional[int] = Field(None, description="Maximum health")
    inCombat: Optional[bool] = Field(None, description="Whether the player is in combat")
    running: Optional[bool] = Field(None, description="Whether the player is running")
    runEnergy: Optional[int] = Field(None, description="Current run energy")
    skills: Optional[Dict[str, Skill]] = Field(None, description="Player's skills")
    prayer: Optional[int] = Field(None, description="Player's prayer points")

class NPC(BaseModel):
    id: int = Field(..., description="NPC ID")
    name: str = Field(..., description="NPC name")
    location: Location = Field(..., description="NPC location")
    distance: Optional[int] = Field(None, description="Distance to NPC")
    interacting: Optional[bool] = Field(None, description="Whether the NPC is interacting")

class GameObject(BaseModel):
    id: int = Field(..., description="Object ID")
    name: str = Field(..., description="Object name")
    location: Location = Field(..., description="Object location")
    actions: List[str] = Field(default_factory=list, description="Available actions")

class GroundItem(BaseModel):
    id: int = Field(..., description="Item ID")
    name: str = Field(..., description="Item name")
    location: Location = Field(..., description="Item location")
    quantity: int = Field(..., description="Item quantity")

class Interface(BaseModel):
    id: int = Field(..., description="Interface ID")
    type: Optional[str] = Field(None, description="Interface type")
    text: Optional[str] = Field(None, description="Interface text")
    actions: Optional[List[str]] = Field(None, description="Available actions")

class ChunkLocation(BaseModel):
    x: int = Field(..., description="Chunk x coordinate")
    y: int = Field(..., description="Chunk y coordinate")

class Exploration(BaseModel):
    currentChunk: ChunkLocation = Field(..., description="Current chunk coordinates")
    visitedChunks: int = Field(..., description="Number of chunks visited")

class Skills(BaseModel):
    Attack: Optional[Skill] = Field(None, description="Attack skill")
    Defence: Optional[Skill] = Field(None, description="Defence skill")
    Strength: Optional[Skill] = Field(None, description="Strength skill")
    Hitpoints: Optional[Skill] = Field(None, description="Hitpoints skill")
    Ranged: Optional[Skill] = Field(None, description="Ranged skill")
    Prayer: Optional[Skill] = Field(None, description="Prayer skill")
    Magic: Optional[Skill] = Field(None, description="Magic skill")
    Cooking: Optional[Skill] = Field(None, description="Cooking skill")
    Woodcutting: Optional[Skill] = Field(None, description="Woodcutting skill")
    Fletching: Optional[Skill] = Field(None, description="Fletching skill")
    Fishing: Optional[Skill] = Field(None, description="Fishing skill")
    Firemaking: Optional[Skill] = Field(None, description="Firemaking skill")
    Crafting: Optional[Skill] = Field(None, description="Crafting skill")
    Smithing: Optional[Skill] = Field(None, description="Smithing skill")
    Mining: Optional[Skill] = Field(None, description="Mining skill")
    Herblore: Optional[Skill] = Field(None, description="Herblore skill")
    Agility: Optional[Skill] = Field(None, description="Agility skill")
    Thieving: Optional[Skill] = Field(None, description="Thieving skill")
    Slayer: Optional[Skill] = Field(None, description="Slayer skill")
    Farming: Optional[Skill] = Field(None, description="Farming skill")
    Runecraft: Optional[Skill] = Field(None, description="Runecraft skill")
    Hunter: Optional[Skill] = Field(None, description="Hunter skill")
    Construction: Optional[Skill] = Field(None, description="Construction skill")

class World(BaseModel):
    number: Optional[int] = Field(None, description="World number")
    type: Optional[str] = Field(None, description="World type")
    location: Optional[str] = Field(None, description="World location")
    population: Optional[int] = Field(None, description="World population")

class GameState(BaseModel):
    player: Optional[Player] = Field(None, description="Player information")
    npcs: Optional[List[NPC]] = Field(None, description="List of nearby NPCs")
    objects: List[GameObject] = Field(default_factory=list, description="List of nearby objects")
    groundItems: Optional[List[GroundItem]] = Field(None, description="List of items on the ground")
    interfaces: Optional[List[Interface]] = Field(None, description="List of visible interfaces")
    exploration: Optional[Exploration] = Field(None, description="Exploration information")
    screenshot_path: str = Field(..., description="Path to the saved screenshot image")
    skills: Optional[Skills] = Field(None, description="Player's skill levels")
    world: Optional[World] = Field(None, description="World information")

class GameContext(BaseModel):
    last_state: Optional[GameState] = Field(None, description="Last retrieved game state")
    base_url: str = Field(..., description="Base URL for the REST API")
    screenshot_dir: str = Field(..., description="Directory to save screenshots")

    def __init__(self, **data):
        super().__init__(**data)
        self.screenshot_dir = os.path.abspath(os.path.join(os.getcwd(), self.screenshot_dir))
        os.makedirs(self.screenshot_dir, exist_ok=True)

class MouseMoveParams(BaseModel):
    x: int = Field(..., description="The x coordinate to move to")
    y: int = Field(..., description="The y coordinate to move to")

class CameraMoveParams(BaseModel):
    key: str = Field(..., description="The key to press (w/a/s/d)")

class ObjectClickParams(BaseModel):
    object_id: int = Field(..., description="The ID of the object to click")

@function_tool
async def move_mouse(context: GameContext, params: MouseMoveParams) -> str:
    """Move the mouse to the specified coordinates"""
    pyautogui.moveTo(params.x, params.y)
    return f"Moved mouse to ({params.x}, {params.y})"

@function_tool
async def click(context: GameContext) -> str:
    """Click at the current mouse position"""
    pyautogui.click()
    return "Clicked at current position"

@function_tool
async def move_camera(context: GameContext) -> str:
    """Move the camera to a random position"""
    key = random.choice(['up', 'down', 'left', 'right'])
    duration = random.uniform(0.1, 0.5)
    pyautogui.keyDown(key)
    await asyncio.sleep(duration)
    pyautogui.keyUp(key)
    return f"Moved camera {key} for {duration:.2f} seconds"

@function_tool
async def get_game_state(context: GameContext) -> GameState:
    """Get the current game state from the REST API"""
    url = "http://localhost:8080/state"
    print(f"Using URL: {url}")
    print(f"Screenshot directory: {context.screenshot_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print("Making request...")
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: requests.get(url, timeout=5)
    )
    print(f"Response status code: {response.status_code}")
    response.raise_for_status()
    data = response.json()
    print(f"Successfully parsed JSON data with keys: {list(data.keys())}")
    screenshot_data = data.pop("screenshot", "")
    screenshot_path = "no_screenshot_available"
    if screenshot_data:
        timestamp = int(time.time())
        screenshot_path = os.path.join(context.screenshot_dir, f"screenshot_{timestamp}.png")
        print(f"Attempting to save screenshot to: {screenshot_path}")
        if not os.path.exists(context.screenshot_dir):
            print(f"Screenshot directory does not exist: {context.screenshot_dir}")
            os.makedirs(context.screenshot_dir, exist_ok=True)
            print(f"Created screenshot directory: {context.screenshot_dir}")
        test_file = os.path.join(context.screenshot_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("Successfully verified write permissions")
        screenshot_bytes = base64.b64decode(screenshot_data)
        print(f"Decoded screenshot data length: {len(screenshot_bytes)} bytes")
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
        print(f"Successfully saved screenshot to: {screenshot_path}")
    data["screenshot_path"] = screenshot_path
    if "player" in data and "health" in data["player"]:
        if isinstance(data["player"]["health"], dict):
            health_data = data["player"]["health"]
        else:
            health_data = {"current": data["player"]["health"], "maximum": data["player"]["maxHealth"]}
        data["player"]["health"] = health_data
    if "skills" in data:
        skills_data = {}
        for skill_name, skill_data in data["skills"].items():
            if isinstance(skill_data, dict):
                skills_data[skill_name] = skill_data
            else:
                skills_data[skill_name] = {
                    "level": skill_data,
                    "realLevel": skill_data,
                    "experience": 0
                }
        data["skills"] = skills_data
    if "exploration" in data and "currentChunk" in data["exploration"]:
        data["exploration"] = {
            "currentChunk": data["exploration"]["currentChunk"],
            "visitedChunks": data["exploration"].get("visitedChunks", 0)
        }
    if "objects" in data:
        data["objects"] = data["objects"][:10]
        print(f"Truncated objects to {len(data['objects'])} items")
    if "npcs" in data:
        data["npcs"] = data["npcs"][:10]
        print(f"Truncated NPCs to {len(data['npcs'])} items")
    if "groundItems" in data:
        data["groundItems"] = data["groundItems"][:10]
        print(f"Truncated ground items to {len(data['groundItems'])} items")
    if "interfaces" in data:
        data["interfaces"] = data["interfaces"][:5]
        print(f"Truncated interfaces to {len(data['interfaces'])} items")
    state = GameState(**data)
    context.last_state = state
    return state

def convert_game_to_screen_coords(game_x: int, game_y: int) -> tuple[int, int]:
    """Convert game coordinates to screen coordinates using OS-specific window positions"""
    import sys
    if sys.platform.startswith("darwin") and importlib.util.find_spec("Quartz") is not None:
        import Quartz
        window_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
        game_window = None
        for window in window_list:
            owner_name = window.get("kCGWindowOwnerName", "")
            window_name = window.get("kCGWindowName", "")
            if "RuneScape" in owner_name or "RuneScape" in window_name:
                game_window = window
                break
        if game_window:
            bounds = game_window.get("kCGWindowBounds", {})
            left = int(bounds.get("X", 0))
            top = int(bounds.get("Y", 0))
            return left + game_x, top + game_y
        return game_x, game_y
    elif sys.platform.startswith("win") and importlib.util.find_spec("win32gui") is not None:
        import win32gui
        hwnd = win32gui.FindWindow(None, "RuneScape")
        if hwnd:
            rect = win32gui.GetWindowRect(hwnd)
            left, top, _, _ = rect
            return left + game_x, top + game_y
        return game_x, game_y
    else:
        return int(game_x), int(game_y)

@function_tool
async def click_game_object(context: GameContext, params: ObjectClickParams) -> str:
    """Click on a game object with the specified ID"""
    state = await get_game_state(context)
    if not state.objects:
        return "No objects found in the game state"
    for obj in state.objects:
        if obj.id == params.object_id:
            screen_x, screen_y = convert_game_to_screen_coords(obj.location.x, obj.location.y)
            await move_mouse(context, MouseMoveParams(x=screen_x, y=screen_y))
            await click(context)
            return f"Clicked on object {obj.name} at ({screen_x}, {screen_y})"
    return f"Object with ID {params.object_id} not found"

async def main():
    print("Starting RuneScape bot...")
    base_url = os.getenv("REST_API_URL", "http://localhost:8080")
    print(f"Using base URL from environment: {base_url}")
    screenshot_dir = "screenshots"
    print(f"Using screenshot directory: {screenshot_dir}")
    context = GameContext(
        base_url=base_url,
        screenshot_dir=screenshot_dir
    )
    print(f"Created game context with base_url: {context.base_url}")
    agent = Agent(
        name="RuneScape Bot",
        instructions="""You are a RuneScape bot that helps players with various tasks in the game.
            You can move the mouse, click, and move the camera to help the player.
            You should try to be as helpful as possible while staying within the rules of the game.""",
        tools=[
            move_mouse,
            click,
            move_camera,
            get_game_state,
            click_game_object
        ],
        model="gpt-4o",
        model_settings=ModelSettings(
            temperature=0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            tool_choice="required"
        )
    )
    print("Created agent")
    
    print("Testing game state access...")
    result = await Runner.run(
        agent,
        "Get the current game state",
        context=context,
        max_turns=50
    )
    print("Successfully accessed game state")
    print(f"Test result: {result}")
    if hasattr(result, 'error'):
        print(f"Error from test: {result.error}")
    if hasattr(result, 'errors'):
        for error in result.errors:
            print(f"Error from test: {error}")
    if hasattr(result, 'raw_responses'):
        for response in result.raw_responses:
            if hasattr(response, 'error'):
                print(f"Error from raw response: {response.error}")
            if hasattr(response, 'content'):
                print(f"Raw response content: {response.content}")
    
    result = await Runner.run(
        agent,
        "Look for nearby trees or resources",
        context=context,
        max_turns=50
    )

    print("Got result from agent")
    print(f"Bot result: {result}")
    if hasattr(result, 'error'):
        print(f"Error from main task: {result.error}")
    if hasattr(result, 'errors'):
        for error in result.errors:
            print(f"Error from main task: {error}")
    if hasattr(result, 'raw_responses'):
        for response in result.raw_responses:
            if hasattr(response, 'error'):
                print(f"Error from raw response: {response.error}")
            if hasattr(response, 'content'):
                print(f"Raw response content: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
