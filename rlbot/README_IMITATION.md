# RuneScape Bot Imitation Learning

This directory contains the imitation learning setup for training the RuneScape bot using human gameplay demonstrations.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the RLBot Recorder plugin enabled in RuneLite:
   - Open RuneLite
   - Go to the Plugin Hub
   - Search for "RLBot Recorder"
   - Install and enable the plugin

## Recording Gameplay

1. Configure the recorder plugin settings in RuneLite:
   - Press F3 to open the configuration panel
   - Find "RLBot Recorder" in the plugin list
   - Adjust settings as needed:
     - Record Hotkey (default: F7)
     - Screenshot Interval (default: 5 ticks)
     - Record Mouse/Keyboard (enabled by default)
     - Save Screenshots (enabled by default)

2. Record gameplay:
   - Start the game and navigate to the area you want to record
   - Press F7 (or your configured hotkey) to start recording
   - Play the game normally, demonstrating the behavior you want the bot to learn
   - Press F7 again to stop recording
   - Recordings are saved in the `recordings` directory

## Training the Model

1. Make sure you have recorded gameplay data in the `recordings` directory

2. Run the imitation learning script:
```bash
python train_imitation.py
```

3. Monitor training progress:
   - Open TensorBoard:
   ```bash
   tensorboard --logdir=logs/imitation_[timestamp]/tensorboard
   ```
   - View training metrics in your browser at http://localhost:6006

4. The script will save:
   - Checkpoints during training (`logs/imitation_[timestamp]/checkpoints/`)
   - The best model based on evaluation (`logs/imitation_[timestamp]/eval/`)
   - The final model after training (`logs/imitation_[timestamp]/final_model.zip`)

## Tips for Good Demonstrations

1. Record multiple sessions of gameplay to provide diverse examples
2. Demonstrate both common actions and edge cases
3. Keep movements smooth and purposeful
4. Include examples of:
   - Combat encounters
   - Navigation
   - Resource gathering
   - Common game interactions

## Troubleshooting

1. If recordings aren't being saved:
   - Check the RuneLite output log for errors
   - Ensure you have write permissions in the recordings directory
   - Verify the plugin is properly enabled

2. If training fails:
   - Check the error messages in the console
   - Verify your recordings contain valid data
   - Ensure you have sufficient disk space for logs and checkpoints

## Advanced Usage

You can customize the training process by modifying parameters in `train_imitation.py`:

```python
train_with_imitation(
    total_timesteps=1000000,  # Increase for more training
    recordings_dir="recordings"  # Change if recordings are elsewhere
)
```

The model architecture and training parameters can be adjusted in the PPO configuration:

```python
model = PPO(
    policy="MultiInputPolicy",
    learning_rate=0.0001,  # Adjust for faster/slower learning
    n_steps=2048,         # Batch size for training
    n_epochs=10,          # Number of training epochs
    batch_size=64,        # Mini-batch size
    ...
)
``` 