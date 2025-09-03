#!/usr/bin/env python3
"""
Player walking detection logic moved from Java to Python for easier editing and testing.
This script can be called from Java using Python Fire.
"""

import fire
import time
from typing import Optional


def is_player_walking(
    last_move_ms: int,
    animation: int,
    pose_animation: int,
    graphic: int,
    is_interacting: bool,
    is_running: bool,
    current_time_ms: Optional[int] = None
) -> bool:
    """
    Check if the player is currently walking/moving.
    
    Args:
        last_move_ms: Timestamp of last movement in milliseconds
        animation: Current animation ID
        pose_animation: Current pose animation ID
        graphic: Current graphic ID
        is_interacting: Whether player is interacting with something
        is_running: Whether player is running (varp 173 == 1)
        current_time_ms: Current time in milliseconds (optional, uses system time if not provided)
    
    Returns:
        bool: True if player is walking/moving, False otherwise
    """
    if current_time_ms is None:
        current_time_ms = int(time.time() * 1000)
    
    # Method 1: Position change detection (most reliable)
    position_changed = current_time_ms - last_move_ms < 2000  # 2 second threshold
    
    # Method 2: Check for specific walking/running animation IDs
    is_walking_anim = animation in [819, 820, 821, 822, 824]
    
    # Method 3: Check pose animation (might be different from main animation)
    has_pose_animation = pose_animation != -1
    
    # Method 4: Check graphic effects (some movement might show graphics)
    has_graphic = graphic != -1
    
    # Method 5: Check if player is interacting with something (might indicate movement)
    # is_interacting is already passed as boolean
    
    # Method 6: Check if player is running (from RunHelper)
    # is_running is already passed as boolean
    
    # Combine all methods - if any indicate movement, consider player as moving
    return position_changed or is_walking_anim or has_pose_animation or has_graphic or is_interacting or is_running


def get_walking_debug_info(
    last_move_ms: int,
    animation: int,
    pose_animation: int,
    graphic: int,
    is_interacting: bool,
    is_running: bool,
    world_location: str,
    current_time_ms: Optional[int] = None
) -> str:
    """
    Get debug info for walking detection.
    
    Args:
        last_move_ms: Timestamp of last movement in milliseconds
        animation: Current animation ID
        pose_animation: Current pose animation ID
        graphic: Current graphic ID
        is_interacting: Whether player is interacting with something
        is_running: Whether player is running
        world_location: Player's world location as string
        current_time_ms: Current time in milliseconds (optional)
    
    Returns:
        str: Debug information string
    """
    if current_time_ms is None:
        current_time_ms = int(time.time() * 1000)
    
    time_since_move = current_time_ms - last_move_ms
    position_changed = time_since_move < 2000
    
    return f"Pos:{world_location} Time:{time_since_move}ms A:{animation} P:{pose_animation} G:{graphic} Int:{'Y' if is_interacting else 'N'} Run:{'Y' if is_running else 'N'}"


if __name__ == "__main__":
    fire.Fire({
        "is_player_walking": is_player_walking,
        "get_walking_debug_info": get_walking_debug_info
    })
