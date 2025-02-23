# RuneScape AI Training Bot

This project implements a reinforcement learning bot for RuneScape using PPO (Proximal Policy Optimization) from Stable Baselines3. The bot learns to navigate and engage in combat through a custom Gymnasium environment that interfaces with RuneLite via WebSocket.

## Features

- Custom Gymnasium environment for RuneScape
- PPO-based reinforcement learning
- Real-time screenshot processing and state management
- TensorBoard integration for training visualization
- Memory-optimized for Apple Silicon (M1/M2) Macs
- Automatic combat engagement and navigation

## Requirements

- Python 3.8+
- RuneLite with RLBot plugin
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd runescape_bot_runelite
```

2. Install dependencies:
```bash
pip install -r rlbot/requirements.txt
```

3. Install and configure the RLBot plugin in RuneLite

## Usage

1. Start RuneLite with the RLBot plugin enabled

2. Run the training script:
```bash
python rlbot/train2.py
```

For Apple Silicon Macs, you can optimize memory usage by setting:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
```

## Project Structure

- `rlbot/train2.py`: Main training script
- `rlbot/requirements.txt`: Python dependencies
- `rlbot/command_schema.json`: WebSocket command schema
- `rlbot/state_schema.json`: Game state schema

## Training

The bot uses the following features for training:
- Screenshot data (120x160 RGB)
- Player position and stats
- NPC information
- Combat state
- Environment exploration data

Training progress can be monitored in TensorBoard:
```bash
tensorboard --logdir rlbot/logs/tb_logs
```

## License

MIT License 