# Enhanced RuneScape Bot

An enhanced version of the RuneScape bot that includes full monitor control and window management capabilities.

## Features

- Automatic RuneLite window detection and management
- Full monitor screenshot capabilities
- Game window-specific screenshot capture
- Cross-platform support (Windows and macOS)
- Enhanced game state tracking
- Automatic game launching if not running

## Requirements

- Python 3.8 or higher
- RuneLite client installed
- Operating system: Windows or macOS

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:
```env
REST_API_URL=http://localhost:8080
```

2. Make sure the RuneLite client is installed in the default location:
- Windows: `%USERPROFILE%\AppData\Local\RuneLite\RuneLite.exe`
- macOS: Applications folder

## Usage

1. Start the bot:
```bash
python rlbot/agent2.py
```

The bot will:
1. Check if RuneLite is running
2. Launch RuneLite if it's not running
3. Capture the game state and screenshots
4. Begin its operation based on the configured tasks

## Development

- The bot uses Pydantic models for type safety
- Screenshots are saved in the `enhanced_screenshots` directory
- Window management is handled by the `WindowManager` class
- Screen capture is handled by the `ScreenCapture` class

## Troubleshooting

### Common Issues

1. **RuneLite Not Found**
   - Verify RuneLite is installed in the default location
   - Check if you have necessary permissions

2. **Screenshot Capture Fails**
   - Ensure you have screen recording permissions (especially on macOS)
   - Check if you have write permissions in the screenshots directory

3. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Verify your Python version is compatible

### Platform-Specific Notes

#### macOS
- You may need to grant screen recording permissions to your terminal/IDE
- Quartz framework is required for window management

#### Windows
- PyWin32 is required for window management
- Run as administrator if you encounter permission issues

## License

[Your License Here] 