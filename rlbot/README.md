

A reinforcement learning agent for playing RuneScape using Proximal Policy Optimization (PPO).



This project implements an AI-powered bot for RuneScape using reinforcement learning. The system connects to the RuneLite client (a third-party RuneScape client) via a custom plugin and learns to perform actions automatically.

The agent is trained using PPO (Proximal Policy Optimization) from Stable Baselines3 to learn optimal behavior based on game state observations and rewards.



- **Custom Gymnasium Environment**: Connects to RuneScape through RuneLite via WebSockets
- **Combined Neural Network**: Processes both screenshots and vector data from the game
- **Reward System**: Rewards for gaining XP, managing health, exploring, and combat
- **Multiple Actions**: Movement, combat, interactions with objects and NPCs, camera controls
- **Structured Training**: Checkpoints, TensorBoard logging, and evaluation metrics



- Python 3.8+
- RuneLite client with the RLBot plugin
- Packages listed in `requirements.txt`



1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/runescape_bot_runelite.git
   cd runescape_bot_runelite
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install and configure the RLBot plugin in your RuneLite client





To test if the system can connect to RuneLite:

```bash
python -m rlbot.main --test
```



To start training a combat bot:

```bash
python -m rlbot.main --timesteps 1000000
```

Optional arguments:
- `--debug`: Enable detailed debug logging
- `--verbose`: Enable verbose output
- `--timesteps`: Specify the number of training timesteps (default: 1,000,000)



- `rlbot/` - Main package
  - `src/` - Source code
    - `environment.py` - RuneScape environment implementation
    - `models.py` - Data models for game state
    - `websocket_client.py` - WebSocket communication with RuneLite
    - `extractors.py` - Feature extractors for neural network
    - `training.py` - Training and evaluation functions
  - `main.py` - Entry point script
  - `logs/` - Log files and TensorBoard logs
  - `models/` - Saved model files
  - `checkpoints/` - Training checkpoints



1. The RuneLite client runs with the RLBot plugin enabled, which exposes game state via WebSocket
2. Our Python code connects to this WebSocket to receive game state updates
3. The agent processes these states through a neural network to decide on actions
4. Actions are sent back to RuneLite via WebSocket to control the game character
5. The agent learns through rewards for successful behaviors



The training follows these steps:
1. Environment initialization and connection to RuneLite
2. State observation including screenshot and game data
3. Action selection by the neural network
4. Execution of action in the game
5. Reward calculation based on outcomes
6. Neural network update using PPO algorithm



- Support for more in-game activities beyond combat
- Improved vision processing for scene understanding
- Inventory management capabilities
- Quest completion functionality



This project is licensed under the MIT License - see the LICENSE file for details.



- RuneLite developers for the excellent client
- Stable Baselines3 team for the reinforcement learning framework 