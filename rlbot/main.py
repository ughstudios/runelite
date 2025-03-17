"""
RuneScape AI Training Bot - Main Entry Point

This script provides a command-line interface for the RuneScape AI training system.
It allows for testing the connection and training an agent using the custom environment.
"""

import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

from src.logging_utils import get_logger
from src.training import test_connection, train_bot
from src.environment import RuneScapeEnv
from src.websocket_client import WebSocketClient


def check_game_logged_in(logger):
    """Attempt to verify if the game is ready and player is logged in."""
    logger.info("Checking if RuneLite client is running and player is logged in...")
    response = test_connection(debug=True)

    if response and not response.get("error"):
        logger.info("✓ RuneLite client is ready and player is logged in!")
        return True
    else:
        error_msg = "Unknown error"
        if response and response.get("error"):
            error_msg = response.get("error")

        logger.warning(f"Game state not available: {error_msg}")
        logger.warning("Please make sure:")
        logger.warning("  1. The RuneLite client is running")
        logger.warning("  2. The RLBot plugin is enabled")
        logger.warning("  3. Your character is logged into the game")
        logger.warning("  4. You are fully loaded into the game world")
        logger.info("The bot will continue checking until you're ready.")
        return False


def random_actions(env, logger, debug=False):
    """Execute random actions in the environment to test functionality.

    Args:
        env: The RuneScape environment
        logger: The logger instance
        debug: Whether to enable debug logging
    """
    logger.info("Executing random actions...")
    
    state = env.reset()

    try:
        for i in range(1, 101):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            
            if i % 10 == 0 or done:
                logger.info(f"Step {i}: Action={action}, Reward={reward:.2f}")
                
                if (
                    state
                    and "player" in env.state
                    and "position" in env.state["player"]
                ):
                    pos = env.state["player"]["position"]
                    logger.info(f"Position: X={pos.get('x', 0)}, Y={pos.get('y', 0)}")

            if done:
                logger.info("Episode finished!")
                break
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Random actions interrupted by user")
    except Exception as e:
        logger.error(f"Error during random actions: {e}", exc_info=debug)
        if debug:
            import traceback
            logger.error(traceback.format_exc())

    logger.info("Random actions completed!")


def setup_arg_parser():
    """Create and configure the argument parser for the CLI."""
    
    parser = argparse.ArgumentParser(
        description="RuneScape AI Training Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute", required=True)

    # Common arguments that apply to all subcommands
    common_args = {
        "--debug": {
            "action": "store_true",
            "help": "Enable debug logging"
        },
        "--verbose": {
            "action": "store_true",
            "help": "Enable verbose output"
        },
        "--wait": {
            "action": "store_true",
            "help": "Wait for player login before proceeding"
        }
    }

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test the connection to RuneLite")
    for arg, kwargs in common_args.items():
        test_parser.add_argument(arg, **kwargs)
    test_parser.add_argument(
        "--task",
        type=str,
        default="general",
        help="Task type to use for environment testing",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train a reinforcement learning agent"
    )
    for arg, kwargs in common_args.items():
        train_parser.add_argument(arg, **kwargs)
    train_parser.add_argument(
        "--task", type=str, default="general", help="Task to train the agent for"
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Number of timesteps to train for",
    )

    # Random command
    random_parser = subparsers.add_parser(
        "random", help="Execute random actions for testing"
    )
    for arg, kwargs in common_args.items():
        random_parser.add_argument(arg, **kwargs)
    random_parser.add_argument(
        "--task",
        type=str,
        default="general",
        help="Task type to use for the environment",
    )
    random_parser.add_argument(
        "--steps", type=int, default=100, help="Number of random steps to take"
    )

    return parser


def main():
    """Main entry point for the application."""
    load_dotenv()
    parser = setup_arg_parser()
    args = parser.parse_args()
    logger = get_logger(debug=args.debug, verbose=args.verbose)
    
    logger.info("=====================================")
    logger.info("   RuneScape AI Training System    ")
    logger.info("=====================================")

    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Verbose logging: {args.verbose}")
    logger.info(f"Auto-wait for login: {args.wait}")

    # Create necessary directories
    Path("./rlbot/logs").mkdir(parents=True, exist_ok=True)
    Path("./rlbot/models").mkdir(parents=True, exist_ok=True)
    Path("./rlbot/checkpoints").mkdir(parents=True, exist_ok=True)

    # Check game login status
    if args.wait:
        attempts = 0
        while not check_game_logged_in(logger):
            attempts += 1
            logger.info(f"Waiting for player to log in (attempt {attempts})")
            time.sleep(5)
    else:
        game_ready = check_game_logged_in(logger)
        if not game_ready:
            logger.warning("Proceeding even though game may not be ready")
            logger.warning("Use --wait flag to automatically wait for player login")

    # Execute the requested command
    if args.command == "test":
        logger.info("Testing connection to RuneLite")
        test_connection(debug=args.debug)
    elif args.command == "train":
        logger.info("Starting training session")
        train_bot(
            debug=args.debug,
            verbose=args.verbose,
            timesteps=args.timesteps
        )
    elif args.command == "random":
        logger.info("Running random actions")

        try:
            logger.info("Creating WebSocket client...")
            ws_client = WebSocketClient()

            logger.info("Connecting to RuneLite client...")
            if ws_client.wait_for_connection(timeout=10.0):
                logger.info("Connected to RuneLite client!")

                env = RuneScapeEnv(task=args.task, debug=args.debug)
                random_actions(env, logger, debug=args.debug)
                env.close()
            else:
                logger.error("Failed to connect to RuneLite client")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=args.debug)
            if args.debug:
                import traceback
                logger.error(traceback.format_exc())
    else:
        logger.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
