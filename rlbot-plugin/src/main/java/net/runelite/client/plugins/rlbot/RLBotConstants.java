package net.runelite.client.plugins.rlbot;

import java.time.format.DateTimeFormatter;
import java.io.File;

/**
 * Constants used by the RLBot plugin.
 */
public final class RLBotConstants {
    /**
     * Path to the log file for plugin logging.
     */
    public static final String LOG_FILE = new File("/Users/danielgleason/Desktop/Code/my_code/runescape_bot_runelite/rlbot/logs", "rlbot-plugin.log").getAbsolutePath();
        
    /**
     * Date format for logging.
     */
    public static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    
    /**
     * Date format for file names.
     */
    public static final DateTimeFormatter DATE_FORMAT_FILE = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
    
    /**
     * Directory for storing screenshots.
     */
    public static final String SCREENSHOTS_DIR = new File("/Users/danielgleason/Desktop/Code/my_code/runescape_bot_runelite/rlbot/screenshots").getAbsolutePath();
    
    /**
     * Size of each exploration chunk.
     */
    public static final int EXPLORATION_CHUNK_SIZE = 8;
    
    /**
     * Camera rotation amount in degrees.
     */
    public static final int CAMERA_ROTATE_AMOUNT = 45;
    
    /**
     * Camera zoom amount.
     */
    public static final float CAMERA_ZOOM_AMOUNT = 25f;
    
    /**
     * Mouse movement constants.
     */
    public static final int MOUSE_LERP_STEPS = 3;
    public static final long MOUSE_MOVE_TIME = 50;
    public static final long MIN_MOVE_DELAY = 25;
    public static final long MAX_MOVE_DELAY = 75;
    
    /**
     * Screenshot related constants.
     */
    public static final long SCREENSHOT_COOLDOWN = 100;
    public static final int SCREENSHOT_WIDTH = 160;
    public static final int SCREENSHOT_HEIGHT = 120;
    public static final float SCREENSHOT_COMPRESSION = 0.3f;
    
    // REST API removed
    
    /**
     * Action types.
     */
    public static final String ACTION_MOVE_AND_CLICK = "moveAndClick";
    public static final String ACTION_CAMERA_ROTATE = "cameraRotate";
    public static final String ACTION_CAMERA_ZOOM = "cameraZoom";
    public static final String ACTION_PRESS_KEY = "pressKey";
    public static final String ACTION_INTERFACE_ACTION = "interfaceAction";
    
    /**
     * Target types.
     */
    public static final String TARGET_NPC = "npc";
    public static final String TARGET_COORDINATES = "coordinates";
    public static final String TARGET_OBJECT = "object";
    public static final String TARGET_GROUND_ITEM = "ground_item";
    
    // === Simplified defaults to avoid config bloat ===
    public static final int DEFAULT_RNG_SEED = 1337;
    public static final int NAV_MINIMAP_CLICK_MS_MIN = 450;
    public static final int NAV_MINIMAP_CLICK_MS_MAX = 800;
    public static final int NEAR_HOTSPOT_TILES = 15;
    public static final int STUCK_NO_PROGRESS_WINDOW_MS = 8000;
    public static final int STUCK_RETRIES = 3;
    public static final int MOUSE_MOVE_INTERPOLATION_PX = 20;
    public static final int EPISODE_MAX_MS = 300_000; // 5 minutes
    public static final boolean EPISODE_END_ON_BANK_DEPOSIT = true;
    /**
     * Private constructor to prevent instantiation.
     */
    private RLBotConstants() {
        // This class should not be instantiated
    }
} 
