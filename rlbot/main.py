#!/usr/bin/env python3
"""
RuneScape AI Training Bot - Main Entry Point

This script provides a command-line interface for the RuneScape AI training system.
It allows for testing the connection and training an agent using the custom environment.
"""

import argparse
import logging
from pathlib import Path

from rich.console import Console
from dotenv import load_dotenv

from src.training import test_connection, train_combat_bot


def main():
    """Main entry point for the application."""
    # Load environment variables if any
    load_dotenv()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="RuneScape AI Training")
    parser.add_argument("--test", action="store_true", help="Test connection only without training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of timesteps to train for")
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose or args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Initialize console for prettier output
    console = Console()
    
    console.print("[bold cyan]RuneScape AI Training System[/bold cyan]")
    console.print(f"Debug mode: {'[green]Enabled[/green]' if args.debug else '[red]Disabled[/red]'}")
    console.print(f"Verbose logging: {'[green]Enabled[/green]' if args.verbose else '[red]Disabled[/red]'}")
    
    # Ensure necessary directories exist
    Path("./rlbot/logs").mkdir(parents=True, exist_ok=True)
    Path("./rlbot/models").mkdir(parents=True, exist_ok=True)
    Path("./rlbot/checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Determine which function to run
    if args.test:
        console.print("[bold yellow]Running connection test...[/bold yellow]")
        test_connection(debug=args.debug or args.verbose)
    else:
        console.print(f"[bold yellow]Starting training for {args.timesteps} timesteps...[/bold yellow]")
        train_combat_bot(
            debug=args.debug, 
            verbose=args.verbose, 
            timesteps=args.timesteps
        )


if __name__ == "__main__":
    main() 