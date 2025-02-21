# RuneScape AI Bot

This project implements a reinforcement learning agent that can play RuneScape through the RuneLite client API.

## Features

- Fully automated combat training
- Smart NPC targeting and interaction
- Inventory management (food, potions)
- Health and prayer management
- Extensive state observation (skills, inventory, NPCs)
- Configurable reward system
- Real-time monitoring and logging

## Requirements

- Python 3.8+
- RuneLite client with RLBot plugin
- Required Python packages (install via `pip install -r requirements.txt`):
  - gym==0.21.0
  - numpy>=1.19.0
  - websocket-client>=1.2.1
  - stable-baselines3>=1.5.0
  - torch>=1.9.0

## Setup

1. Build and run the modified RuneLite client:
```bash
cd runelite
mvn clean install -DskipTests
java -jar runelite-client/target/client-1.11.2-SNAPSHOT-shaded.jar
```

2. Enable the RLBot plugin in RuneLite settings

3. Set up the Python environment:
```bash
cd rlbot
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Training the Bot

1. Start training:
```bash
python train.py
```

The training script will:
- Create a new log directory with timestamp
- Save checkpoints every 10,000 steps
- Save the best model based on evaluation
- Log training metrics to TensorBoard
- Handle training interruptions gracefully

2. Monitor training progress:
```bash
tensorboard --logdir logs/combat_bot_latest/tensorboard/
```

## Testing the Bot

1. Test a trained model:
```python
from train import test_combat_bot

test_combat_bot(
    model_path="logs/combat_bot_latest/final_model.zip",
    vec_normalize_path="logs/combat_bot_latest/vec_normalize.pkl"
)
```

## Configuration

The bot's behavior can be configured through:

1. Environment parameters in `runescape_env.py`:
- Action space definition
- Observation space structure
- Reward calculation
- Episode termination conditions

2. Training parameters in `train.py`:
- Learning rate
- Batch size
- Training steps
- Model architecture
- Exploration settings

## Safety Features

The bot includes several safety measures:
- Action rate limiting
- Health monitoring
- Safe area restrictions
- Graceful error handling
- Clean shutdown procedures

## Extending the Bot

The bot can be extended to support more tasks by:

1. Adding new action types in `Action` enum
2. Implementing new task handlers in `_execute_action`
3. Creating appropriate reward functions
4. Defining task-specific termination conditions

## Contributing

Feel free to contribute by:
1. Adding new features
2. Improving the reward system
3. Optimizing the model architecture
4. Adding support for more RuneScape activities

## Disclaimer

This bot is for educational purposes only. Using automated tools in RuneScape may violate the game's terms of service. Use at your own risk. 