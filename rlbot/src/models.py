
"""
Data models for the RuneScape environment.

This module defines Pydantic models for game state data structures.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Location(BaseModel):
    """Location of an entity in the game world."""

    x: int
    y: int
    plane: int = 0


class Health(BaseModel):
    """Health of an entity."""

    current: int
    maximum: int


class Skill(BaseModel):
    """Skill with level and experience."""

    level: int
    realLevel: int = 0
    experience: int = 0


class Player(BaseModel):
    """Player character data."""

    location: Location
    health: Health
    inCombat: bool = False
    isRunning: bool = False
    runEnergy: float = 0
    skills: Dict[str, Skill] = {}
    prayer: int = 0


class NPC(BaseModel):
    """Non-player character data."""

    id: int
    name: str = ""
    combatLevel: int = 0
    location: Location
    health: Optional[Health] = None
    interacting: bool = False
    distance: float = 0
    actions: List[str] = []


class GameObject(BaseModel):
    """Game object data (e.g., trees, doors)."""

    id: int
    name: str = ""
    location: Location
    actions: List[str] = []
    distance: float = 0


class GroundItem(BaseModel):
    """Item on the ground."""

    id: int
    name: str = ""
    quantity: int = 1
    location: Location


class InterfaceOption(BaseModel):
    """Option for an interface element."""

    text: str = ""
    type: str = ""


class InterfaceElement(BaseModel):
    """Interface element (menu, dialog, etc.)."""

    id: int
    groupId: int = 0
    type: str = ""
    text: str = ""
    options: List[InterfaceOption] = []
    actions: List[str] = []


class ChunkCoordinate(BaseModel):
    """Coordinate of a map chunk."""

    x: int
    y: int


class Exploration(BaseModel):
    """Player's exploration data."""

    currentChunk: ChunkCoordinate = Field(
        default_factory=lambda: ChunkCoordinate(x=0, y=0)
    )
    visitedChunks: int = 0


class GameState(BaseModel):
    """Complete game state."""

    player: Player
    npcs: List[NPC] = []
    objects: List[GameObject] = []
    groundItems: List[GroundItem] = []
    interfaces: List[InterfaceElement] = []
    interfacesOpen: bool = False
    pathObstructed: bool = False
    exploration: Optional[Exploration] = None
    screenshot: str
