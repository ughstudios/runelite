package net.runelite.client.plugins.rlbot;

import net.runelite.client.config.Config;
import net.runelite.client.config.ConfigGroup;
import net.runelite.client.config.ConfigItem;
import net.runelite.client.config.Range;

/**
 * Configuration interface for the RLBot plugin.
 */
@ConfigGroup("rlbot")
public interface RLBotConfig extends Config {

    /**
     * Whether to save screenshots.
     */
    @ConfigItem(
        keyName = "saveScreenshots",
        name = "Save Screenshots",
        description = "Save screenshots to disk"
    )
    default boolean saveScreenshots() {
        return false;
    }
    
    /**
     * The interval at which to send game state updates.
     */
    @ConfigItem(
        keyName = "updateInterval",
        name = "Update Interval (ms)",
        description = "The interval in milliseconds at which to send game state updates"
    )
    @Range(
        min = 50,
        max = 1000
    )
    default int updateInterval() {
        return 100;
    }
    
    /**
     * Whether to enable debug logging.
     */
    @ConfigItem(
        keyName = "debugLogging",
        name = "Debug Logging",
        description = "Enable debug logging"
    )
    default boolean debugLogging() {
        return true;
    }

    @ConfigItem(
        keyName = "quietLogging",
        name = "Quiet Logging (suppress info)",
        description = "When enabled, suppress most info-level logs to reduce I/O"
    )
    default boolean quietLogging() { return true; }

    @ConfigItem(
        keyName = "perfLogging",
        name = "Performance Logging",
        description = "Log per-task and input timing measurements"
    )
    default boolean perfLogging() { return true; }

    @ConfigItem(
        keyName = "mouseMoveInterpolationPx",
        name = "Mouse move interpolation (px/step)",
        description = "Lower = smoother moves; higher = fewer steps"
    )
    @Range(min = 5, max = 60)
    default int mouseMoveInterpolationPx() { return 20; }
    
    /**
     * Whether to show the current action in the overlay.
     */
    @ConfigItem(
        keyName = "showOverlay",
        name = "Show Overlay",
        description = "Show the RLBot overlay"
    )
    default boolean showOverlay() {
        return true;
    }
    
    /**
     * Whether to show the cursor overlay.
     */
    @ConfigItem(
        keyName = "showCursorOverlay",
        name = "Show Cursor Overlay",
        description = "Shows a visual overlay of the cursor position",
        position = 99
    )
    default boolean showCursorOverlay()
    {
        return true;
    }

    @ConfigItem(
        keyName = "enableRLAgent",
        name = "Enable RL Agent",
        description = "Run the reinforcement learning agent"
    )
    default boolean enableRLAgent() {
        return true;
    }

    @ConfigItem(
        keyName = "agentIntervalMs",
        name = "Agent Interval (ms)",
        description = "How often the Java agent decides and acts"
    )
    @Range(min = 50, max = 2000)
    default int agentIntervalMs() {
        return 250;
    }

    @ConfigItem(
        keyName = "inventoryFreeSlotsToBank",
        name = "Inventory Free Slots To Bank",
        description = "When free slots are at or below this, navigate to bank"
    )
    @Range(min = 0, max = 28)
    default int inventoryFreeSlotsToBank() {
        return 0; // bank only when inventory is full
    }

    @ConfigItem(
        keyName = "navMinimapClickMsMin",
        name = "Nav Minimap Click Min (ms)",
        description = "Minimum busy time after a minimap step"
    )
    @Range(min = 200, max = 1500)
    default int navMinimapClickMsMin() { return 450; }

    @ConfigItem(
        keyName = "navMinimapClickMsMax",
        name = "Nav Minimap Click Max (ms)",
        description = "Maximum busy time after a minimap step"
    )
    @Range(min = 300, max = 2000)
    default int navMinimapClickMsMax() { return 800; }

    @ConfigItem(
        keyName = "nearHotspotTiles",
        name = "Near Hotspot (tiles)",
        description = "Stop nav steps when this close to hotspot"
    )
    @Range(min = 5, max = 30)
    default int nearHotspotTiles() { return 15; }

    @ConfigItem(
        keyName = "stuckNoProgressWindowMs",
        name = "Stuck Window (ms)",
        description = "Window for considering nav no-progress"
    )
    @Range(min = 3000, max = 15000)
    default int stuckNoProgressWindowMs() { return 8000; }

    @ConfigItem(
        keyName = "stuckRetries",
        name = "Stuck Retries",
        description = "How many no-progress windows before recovery"
    )
    @Range(min = 1, max = 5)
    default int stuckRetries() { return 3; }

    // RL controls

    @ConfigItem(
        keyName = "rlEpsilon",
        name = "RL Epsilon",
        description = "Exploration rate ε (0.0-1.0)"
    )
    @Range(min = 0, max = 1)
    default int rlEpsilon() { return 0; } // default 0 for deterministic behavior

    @ConfigItem(
        keyName = "rlAlpha",
        name = "RL Alpha",
        description = "Learning rate α (0.0-1.0)"
    )
    @Range(min = 1, max = 100)
    default int rlAlpha() { return 20; } // percent

    @ConfigItem(
        keyName = "rlGamma",
        name = "RL Gamma",
        description = "Discount γ (0.0-1.0)"
    )
    @Range(min = 0, max = 100)
    default int rlGamma() { return 90; } // percent

    @ConfigItem(
        keyName = "rlBatchSize",
        name = "RL Batch Size",
        description = "Mini-batch size for DQN training"
    )
    @Range(min = 8, max = 256)
    default int rlBatchSize() { return 32; }

    @ConfigItem(
        keyName = "rlReplayCapacity",
        name = "RL Replay Capacity",
        description = "Replay buffer capacity"
    )
    @Range(min = 1000, max = 200000)
    default int rlReplayCapacity() { return 20000; }

    @ConfigItem(
        keyName = "rlTargetSyncSteps",
        name = "RL Target Sync (steps)",
        description = "How often to sync target network"
    )
    @Range(min = 100, max = 20000)
    default int rlTargetSyncSteps() { return 1000; }

    @ConfigItem(
        keyName = "rngSeed",
        name = "RNG Seed",
        description = "Seed for deterministic decisions (same seed → same behavior)"
    )
    @Range(min = 0, max = 1_000_000_000)
    default int rngSeed() { return 1337; }

    // Removed test toggles; actions will be triggered via UI buttons on RLBot panel
} 