"""
RuneScape AI Training Bot - Source Package

This package contains the components for training a reinforcement learning
agent to play RuneScape via the RuneLite client.
"""

from .environment import RuneScapeEnv, Action
from .models import GameState, Player, NPC
from .extractors import CombinedExtractor
from .training import test_connection, train_bot, create_ppo_agent
from .websocket_client import WebSocketClient

__all__ = [
    "RuneScapeEnv",
    "Action",
    "GameState",
    "Player",
    "NPC",
    "CombinedExtractor",
    "test_connection",
    "train_bot",
    "create_ppo_agent",
    "WebSocketClient",
]
