package net.runelite.client.plugins.rlbot;

import net.runelite.client.config.Config;
import net.runelite.client.config.ConfigGroup;
import net.runelite.client.config.ConfigItem;
import net.runelite.client.config.ConfigSection;
import net.runelite.client.config.Range;

@ConfigGroup("rlbot")
public interface RLBotConfig extends Config {

    // Keep only the essentials for a simple learning setup
    @ConfigSection(
        name = "Core",
        description = "Minimal controls for Gym",
        position = 0,
        closedByDefault = false
    )
    String coreSection = "coreSection";

    @ConfigSection(
        name = "Logging",
        description = "Verbosity of RLBot logs",
        position = 1,
        closedByDefault = true
    )
    String loggingSection = "loggingSection";

    // Core
    @ConfigItem(
        section = coreSection,
        keyName = "enableGymControl",
        name = "Enable Gym Control",
        description = "Drive actions from a Gym controller via file-based IPC",
        position = 0
    )
    default boolean enableGymControl() { return true; }

    @ConfigItem(
        section = coreSection,
        keyName = "gymIpcDir",
        name = "Gym IPC Dir",
        description = "Directory for IPC files with the Gym controller",
        position = 1
    )
    default String gymIpcDir() { return "rlbot-ipc"; }

    @ConfigItem(
        section = coreSection,
        keyName = "gymStepIntervalMs",
        name = "Gym Step Interval (ms)",
        description = "How often to exchange observation/action with Gym",
        position = 2
    )
    @Range(min = 50, max = 2000)
    default int gymStepIntervalMs() { return 250; }

    @ConfigItem(
        section = coreSection,
        keyName = "showOverlay",
        name = "Show Overlay",
        description = "Show the RLBot overlay",
        position = 3
    )
    default boolean showOverlay() { return true; }

    // Logging
    enum LoggingLevel { QUIET, NORMAL, VERBOSE }

    @ConfigItem(
        section = loggingSection,
        keyName = "logLevel",
        name = "Log Level",
        description = "Verbosity of RLBot logging",
        position = 0
    )
    default LoggingLevel logLevel() { return LoggingLevel.NORMAL; }
}
