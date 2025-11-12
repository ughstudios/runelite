package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.plugins.rlbot.RLBotLogger;
import net.runelite.client.plugins.rlbot.RLBotConfig;
import net.runelite.client.plugins.rlbot.input.RLBotInputHandler;
import net.runelite.client.plugins.rlbot.RLBotTelemetry;
import net.runelite.api.Player;
import net.runelite.api.AnimationID;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;
import net.runelite.api.ItemContainer;
import net.runelite.api.InventoryID;
import net.runelite.api.Item;
import javax.inject.Inject;

/**
 * Shared utilities available to tasks.
 */
public final class TaskContext {
    public final Client client;
    public final ClientThread clientThread;
    public final RLBotLogger logger;
    public final RLBotInputHandler input;
    public final RLBotConfig config;
    public final RLBotTelemetry telemetry;
    private final java.util.Random deterministicRng;
    // Slim composition helpers
    private final net.runelite.client.plugins.rlbot.tasks.context.BusyTracker busyTracker = new net.runelite.client.plugins.rlbot.tasks.context.BusyTracker();
    private final net.runelite.client.plugins.rlbot.tasks.context.InventoryTracker inventoryTracker = new net.runelite.client.plugins.rlbot.tasks.context.InventoryTracker();
    private final net.runelite.client.plugins.rlbot.tasks.context.NavTracker navTracker = new net.runelite.client.plugins.rlbot.tasks.context.NavTracker();
    private final net.runelite.client.plugins.rlbot.tasks.context.MovementAnalyzer movementAnalyzer = new net.runelite.client.plugins.rlbot.tasks.context.MovementAnalyzer(busyTracker);
    private final net.runelite.client.plugins.rlbot.tasks.context.WidgetLoggerUtil widgetLogger = new net.runelite.client.plugins.rlbot.tasks.context.WidgetLoggerUtil();

    @Inject
    public TaskContext(Client client, ClientThread clientThread, RLBotLogger logger, RLBotInputHandler input, RLBotConfig config, RLBotTelemetry telemetry) {
        this.client = client;
        this.clientThread = clientThread;
        this.logger = logger;
        this.input = input;
        this.config = config;
        this.telemetry = telemetry;
        this.deterministicRng = new java.util.Random(net.runelite.client.plugins.rlbot.RLBotConstants.DEFAULT_RNG_SEED);
    }

    // moved state into dedicated trackers



    public boolean isBusy() { return busyTracker.isBusy(this); }

    /**
     * Refresh movement timestamp from the player's current world location.
     * Call this once per tick to ensure lastMoveMs stays accurate even when
     * no task invokes isBusy().
     */
    public void refreshMovementFromPlayer() {
        busyTracker.refreshMovementFromPlayer(this);
    }

    public void setBusyForMs(long durationMs) { busyTracker.setBusyForMs(this, durationMs); }

    /**
     * Clear any artificial busy lock so a manual action can run immediately.
     */
    public void clearBusyLock() { busyTracker.clearBusyLock(); }

    public boolean timedOutSince(long sinceMs) { return busyTracker.timedOutSince(sinceMs); }



    /**
     * Comprehensive widget scanning and logging using BFS/DFS
     */
    public void logAllOpenWidgets() { widgetLogger.logAllOpenWidgets(this); }
    
    /**
     * Log detailed information about a specific widget
     */
    private void logWidgetDetails(Widget widget, int groupId, int widgetId) { /* moved to WidgetLoggerUtil */ }
    
    /**
     * Log widget children using Depth-First Search
     */
    private void logWidgetChildrenDFS(Widget[] children, int groupId, int parentWidgetId, int depth) { /* moved to WidgetLoggerUtil */ }
    
    /**
     * Log widgets using Breadth-First Search (alternative approach)
     */
    public void logWidgetsBFS() { widgetLogger.logWidgetsBFS(this); }
    
    /**
     * Helper class for BFS widget traversal
     */
    private static class WidgetInfo { WidgetInfo(Widget w,int g,int id,int d){} }
    
    /**
     * Log specific widget groups (useful for debugging specific interfaces)
     */
    public void logSpecificWidgetGroups(int... groupIds) { widgetLogger.logSpecificWidgetGroups(this, groupIds); }
    private boolean isDialogOpen() { return false; }

    public void markCanvasClick() { }

    public void markMenuWalkClick() { }

    public boolean wasLastClickCanvasRecent(long withinMs) { return false; }

    public boolean wasLastClickCanvas() { return false; }

    public boolean isPlayerMovingRecent(long withinMs) { return busyTracker.isPlayerMovingRecent(withinMs); }

    public boolean isWoodcuttingAnim() {
        Player p = client.getLocalPlayer();
        if (p == null) return false;
        int anim = p.getAnimation();
        
        // Check for specific woodcutting animations
        return anim == AnimationID.WOODCUTTING_BRONZE ||
               anim == AnimationID.WOODCUTTING_IRON ||
               anim == AnimationID.WOODCUTTING_STEEL ||
               anim == AnimationID.WOODCUTTING_BLACK ||
               anim == AnimationID.WOODCUTTING_MITHRIL ||
               anim == AnimationID.WOODCUTTING_ADAMANT ||
               anim == AnimationID.WOODCUTTING_RUNE ||
               anim == AnimationID.WOODCUTTING_GILDED ||
               anim == AnimationID.WOODCUTTING_DRAGON ||
               anim == AnimationID.WOODCUTTING_DRAGON_OR ||
               anim == AnimationID.WOODCUTTING_INFERNAL ||
               anim == AnimationID.WOODCUTTING_3A_AXE ||
               anim == AnimationID.WOODCUTTING_CRYSTAL ||
               anim == AnimationID.WOODCUTTING_TRAILBLAZER ||
               anim == AnimationID.WOODCUTTING_2H_BRONZE ||
               anim == AnimationID.WOODCUTTING_2H_IRON ||
               anim == AnimationID.WOODCUTTING_2H_STEEL ||
               anim == AnimationID.WOODCUTTING_2H_BLACK ||
               anim == AnimationID.WOODCUTTING_2H_MITHRIL ||
               anim == AnimationID.WOODCUTTING_2H_ADAMANT ||
               anim == AnimationID.WOODCUTTING_2H_RUNE ||
               anim == AnimationID.WOODCUTTING_2H_DRAGON ||
               anim == AnimationID.WOODCUTTING_2H_CRYSTAL ||
               anim == AnimationID.WOODCUTTING_2H_3A;
    }

    public float getRunEnergy01() {
        try {
            int e = client.getEnergy(); // 0..10000
            return Math.max(0f, Math.min(1f, e / 10000f));
        } catch (Exception ignored) { return 0f; }
    }

    public long getBusyRemainingMsEstimate() { return busyTracker.getBusyRemainingMsEstimate(); }

    public String getBusyDebugString() { return busyTracker.getBusyDebugString(this); }

    public java.util.Random rng() {
        return deterministicRng;
    }

    public int getInventoryFreeSlots() { return inventoryTracker.getInventoryFreeSlots(this); }

    public boolean isInventoryNearFull() { return inventoryTracker.isInventoryNearFull(this); }
    
    public boolean isInventoryFull() { return inventoryTracker.isInventoryFull(this); }

    public void updateNavProgress(int currentDistanceTiles) { navTracker.updateNavProgress(this, currentDistanceTiles); }

    public int getNavNoProgressCount() { return navTracker.getNavNoProgressCount(); }

    public void resetNavProgress() { navTracker.resetNavProgress(); }

    /**
     * Check if the player is currently walking/moving
     */
    public boolean isPlayerWalking() { return movementAnalyzer.isPlayerWalking(this); }
    
    public long getLastMoveMs() { return movementAnalyzer.getLastMoveMs(); }
    
    /**
     * Get debug info for walking detection
     */
    public String getWalkingDebugInfo() { return movementAnalyzer.getWalkingDebugInfo(this); }
    
    /**
     * Java fallback implementation of debug info
     */
    private String getWalkingDebugInfoJava() { return movementAnalyzer.getWalkingDebugInfo(this); }
}
