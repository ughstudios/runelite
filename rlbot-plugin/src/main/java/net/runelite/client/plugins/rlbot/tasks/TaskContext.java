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

    @Inject
    public TaskContext(Client client, ClientThread clientThread, RLBotLogger logger, RLBotInputHandler input, RLBotConfig config, RLBotTelemetry telemetry) {
        this.client = client;
        this.clientThread = clientThread;
        this.logger = logger;
        this.input = input;
        this.config = config;
        this.telemetry = telemetry;
        this.deterministicRng = new java.util.Random(config.rngSeed());
    }

    private long lastActionMs = 0L;
    private long busyUntilMs = 0L;
    private WorldPoint lastWorldPoint = null;
    private long lastMoveMs = 0L;
    // Navigation progress tracking
    private Integer lastNavDistance = null;
    private long lastNavUpdateMs = 0L;
    private int navNoProgressCount = 0;

    // Click execution context
    private boolean lastClickWasCanvas = false;
    private long lastClickMs = 0L;



    public boolean isBusy() {
        Player p = client.getLocalPlayer();
        if (p == null) return false;
        int anim = p.getAnimation();
        boolean performing = anim != -1 && anim != AnimationID.IDLE;
        long now = System.currentTimeMillis();
        WorldPoint wp = p.getWorldLocation();
        if (wp != null) {
            if (lastWorldPoint == null || !wp.equals(lastWorldPoint)) {
                lastWorldPoint = wp;
                lastMoveMs = now;
            }
        }
        boolean moving = isPlayerWalking(); // Use the improved walking detection
        boolean inDialog = isDialogOpen();
        boolean timeLocked = System.currentTimeMillis() < busyUntilMs;
        return performing || moving || inDialog || timeLocked;
    }

    public void setBusyForMs(long durationMs) {
        long now = System.currentTimeMillis();
        long proposed = now + Math.max(25, durationMs); // Reduced from 50ms to 25ms minimum
        // Clamp excessively long busy locks so the agent can't get stuck forever
        if (busyUntilMs > 0 && busyUntilMs - now > 3000L) { // Reduced from 6000L to 3000L
            busyUntilMs = now + 1000L; // Reduced from 2000L to 1000L
        } else {
            busyUntilMs = Math.max(busyUntilMs, proposed);
        }
        lastActionMs = now;
        if (telemetry != null) {
            telemetry.setBusyRemainingMs(durationMs);
            telemetry.incEpisodeSteps();
        }
    }

    public boolean timedOutSince(long sinceMs) {
        return System.currentTimeMillis() - lastActionMs > sinceMs;
    }



    /**
     * Comprehensive widget scanning and logging using BFS/DFS
     */
    public void logAllOpenWidgets() {
        logger.info("=== WIDGET SCAN START ===");
        
        // Get all widget roots
        Widget[] widgetRoots = client.getWidgetRoots();
        if (widgetRoots == null) {
            logger.warn("No widget roots found");
            return;
        }
        
        int totalWidgets = 0;
        int visibleWidgets = 0;
        
        for (int rootId = 0; rootId < widgetRoots.length; rootId++) {
            Widget rootWidget = widgetRoots[rootId];
            if (rootWidget == null) continue;
            
            totalWidgets++;
            
            if (!rootWidget.isHidden()) {
                visibleWidgets++;
                logger.info("Root widget " + rootId + " is visible:");
                logWidgetDetails(rootWidget, rootId, 0);
            }
        }
        
        logger.info("=== WIDGET SCAN SUMMARY ===");
        logger.info("Total root widgets: " + totalWidgets);
        logger.info("Visible root widgets: " + visibleWidgets);
        logger.info("=== WIDGET SCAN END ===");
    }
    
    /**
     * Log detailed information about a specific widget
     */
    private void logWidgetDetails(Widget widget, int groupId, int widgetId) {
        StringBuilder sb = new StringBuilder();
        sb.append("  Widget[").append(groupId).append(",").append(widgetId).append("]: ");
        
        // Basic properties
        sb.append("hidden=").append(widget.isHidden());
        sb.append(", visible=").append(!widget.isHidden());
        sb.append(", x=").append(widget.getRelativeX());
        sb.append(", y=").append(widget.getRelativeY());
        sb.append(", width=").append(widget.getWidth());
        sb.append(", height=").append(widget.getHeight());
        
        // Text content
        String text = widget.getText();
        if (text != null && !text.isEmpty()) {
            sb.append(", text=\"").append(text).append("\"");
        }
        
        // Name
        String name = widget.getName();
        if (name != null && !name.isEmpty()) {
            sb.append(", name=\"").append(name).append("\"");
        }
        
        // Actions
        String[] actions = widget.getActions();
        if (actions != null && actions.length > 0) {
            sb.append(", actions=[");
            for (int i = 0; i < actions.length; i++) {
                if (actions[i] != null && !actions[i].isEmpty()) {
                    if (i > 0) sb.append(", ");
                    sb.append("\"").append(actions[i]).append("\"");
                }
            }
            sb.append("]");
        }
        
        // Item ID
        int itemId = widget.getItemId();
        if (itemId != -1) {
            sb.append(", itemId=").append(itemId);
        }
        
        // Item quantity
        int itemQuantity = widget.getItemQuantity();
        if (itemQuantity > 0) {
            sb.append(", quantity=").append(itemQuantity);
        }
        
        // Model ID
        int modelId = widget.getModelId();
        if (modelId != -1) {
            sb.append(", modelId=").append(modelId);
        }
        
        // Sprite ID
        int spriteId = widget.getSpriteId();
        if (spriteId != -1) {
            sb.append(", spriteId=").append(spriteId);
        }
        
        // Color
        int textColor = widget.getTextColor();
        if (textColor != -1) {
            sb.append(", textColor=").append(textColor);
        }
        
        // Opacity
        int opacity = widget.getOpacity();
        if (opacity != -1) {
            sb.append(", opacity=").append(opacity);
        }
        
        // Parent widget info
        Widget parent = widget.getParent();
        if (parent != null) {
            sb.append(", parent=[").append(parent.getId()).append("]");
        }
        
        // Children count
        Widget[] children = widget.getChildren();
        if (children != null && children.length > 0) {
            int childCount = 0;
            for (Widget child : children) {
                if (child != null && !child.isHidden()) {
                    childCount++;
                }
            }
            if (childCount > 0) {
                sb.append(", children=").append(childCount);
            }
        }
        
        logger.info(sb.toString());
        
        // Recursively log children using DFS
        if (children != null && children.length > 0) {
            logWidgetChildrenDFS(children, groupId, widgetId, 1);
        }
    }
    
    /**
     * Log widget children using Depth-First Search
     */
    private void logWidgetChildrenDFS(Widget[] children, int groupId, int parentWidgetId, int depth) {
        for (int i = 0; i < children.length; i++) {
            Widget child = children[i];
            if (child == null || child.isHidden()) continue;
            
            StringBuilder indent = new StringBuilder();
            for (int d = 0; d < depth; d++) {
                indent.append("    ");
            }
            
            StringBuilder sb = new StringBuilder();
            sb.append(indent).append("Child[").append(i).append("]: ");
            
            // Basic child properties
            sb.append("hidden=").append(child.isHidden());
            sb.append(", visible=").append(!child.isHidden());
            sb.append(", x=").append(child.getRelativeX());
            sb.append(", y=").append(child.getRelativeY());
            
            // Text content
            String text = child.getText();
            if (text != null && !text.isEmpty()) {
                sb.append(", text=\"").append(text).append("\"");
            }
            
            // Actions
            String[] actions = child.getActions();
            if (actions != null && actions.length > 0) {
                sb.append(", actions=[");
                for (int j = 0; j < actions.length; j++) {
                    if (actions[j] != null && !actions[j].isEmpty()) {
                        if (j > 0) sb.append(", ");
                        sb.append("\"").append(actions[j]).append("\"");
                    }
                }
                sb.append("]");
            }
            
            // Item info
            int itemId = child.getItemId();
            if (itemId != -1) {
                sb.append(", itemId=").append(itemId);
                int quantity = child.getItemQuantity();
                if (quantity > 0) {
                    sb.append(", quantity=").append(quantity);
                }
            }
            
            logger.info(sb.toString());
            
            // Recursively process children
            Widget[] grandChildren = child.getChildren();
            if (grandChildren != null && grandChildren.length > 0) {
                logWidgetChildrenDFS(grandChildren, groupId, parentWidgetId, depth + 1);
            }
        }
    }
    
    /**
     * Log widgets using Breadth-First Search (alternative approach)
     */
    public void logWidgetsBFS() {
        logger.info("=== WIDGET BFS SCAN START ===");
        
        Widget[] widgetRoots = client.getWidgetRoots();
        if (widgetRoots == null) {
            logger.warn("No widget roots found for BFS scan");
            return;
        }
        
        java.util.Queue<WidgetInfo> queue = new java.util.LinkedList<>();
        java.util.Set<String> visited = new java.util.HashSet<>();
        
        // Start with all visible root widgets
        for (int rootId = 0; rootId < widgetRoots.length; rootId++) {
            Widget rootWidget = widgetRoots[rootId];
            if (rootWidget != null && !rootWidget.isHidden()) {
                queue.offer(new WidgetInfo(rootWidget, rootId, 0, 0));
            }
        }
        
        int level = 0;
        int widgetsAtCurrentLevel = queue.size();
        int widgetsAtNextLevel = 0;
        
        while (!queue.isEmpty()) {
            WidgetInfo current = queue.poll();
            widgetsAtCurrentLevel--;
            
            String widgetKey = current.groupId + "," + current.widgetId;
            if (visited.contains(widgetKey)) continue;
            visited.add(widgetKey);
            
            // Log current widget
            StringBuilder indent = new StringBuilder();
            for (int i = 0; i < current.depth; i++) {
                indent.append("  ");
            }
            
            logger.info(indent + "BFS Level " + current.depth + ": Widget[" + current.groupId + "," + current.widgetId + "]");
            
            // Add children to queue
            Widget[] children = current.widget.getChildren();
            if (children != null) {
                for (int i = 0; i < children.length; i++) {
                    Widget child = children[i];
                    if (child != null && !child.isHidden()) {
                        queue.offer(new WidgetInfo(child, current.groupId, i, current.depth + 1));
                        widgetsAtNextLevel++;
                    }
                }
            }
            
            // Level transition
            if (widgetsAtCurrentLevel == 0 && !queue.isEmpty()) {
                level++;
                widgetsAtCurrentLevel = widgetsAtNextLevel;
                widgetsAtNextLevel = 0;
                logger.info("--- BFS Level " + level + " ---");
            }
        }
        
        logger.info("=== WIDGET BFS SCAN END ===");
    }
    
    /**
     * Helper class for BFS widget traversal
     */
    private static class WidgetInfo {
        final Widget widget;
        final int groupId;
        final int widgetId;
        final int depth;
        
        WidgetInfo(Widget widget, int groupId, int widgetId, int depth) {
            this.widget = widget;
            this.groupId = groupId;
            this.widgetId = widgetId;
            this.depth = depth;
        }
    }
    
    /**
     * Log specific widget groups (useful for debugging specific interfaces)
     */
    public void logSpecificWidgetGroups(int... groupIds) {
        logger.info("=== SPECIFIC WIDGET GROUPS SCAN ===");
        
        Widget[] widgetRoots = client.getWidgetRoots();
        if (widgetRoots == null) {
            logger.warn("No widget roots found");
            return;
        }
        
        for (int groupId : groupIds) {
            if (groupId < 0 || groupId >= widgetRoots.length) {
                logger.warn("Invalid group ID: " + groupId);
                continue;
            }
            
            Widget rootWidget = widgetRoots[groupId];
            if (rootWidget == null) {
                logger.info("Group " + groupId + ": null");
                continue;
            }
            
            logger.info("=== Group " + groupId + " ===");
            int visibleCount = 0;
            
            if (!rootWidget.isHidden()) {
                visibleCount++;
                logWidgetDetails(rootWidget, groupId, 0);
            }
            
            logger.info("Group " + groupId + " visible widgets: " + visibleCount);
        }
        
        logger.info("=== SPECIFIC WIDGET GROUPS SCAN END ===");
    }
    private boolean isDialogOpen() {
        Widget w = client.getWidget(net.runelite.api.widgets.WidgetInfo.DIALOG_NPC_TEXT);
        if (w != null && !w.isHidden()) return true;
        Widget c = client.getWidget(net.runelite.api.widgets.WidgetInfo.DIALOG_PLAYER_TEXT);
        return c != null && !c.isHidden();
    }

    public void markCanvasClick() {
        lastClickWasCanvas = true;
        lastClickMs = System.currentTimeMillis();
    }

    public void markMenuWalkClick() {
        lastClickWasCanvas = false;
        lastClickMs = System.currentTimeMillis();
    }

    public boolean wasLastClickCanvasRecent(long withinMs) {
        return System.currentTimeMillis() - lastClickMs <= withinMs && lastClickWasCanvas;
    }

    public boolean wasLastClickCanvas() { return lastClickWasCanvas; }

    public boolean isPlayerMovingRecent(long withinMs) {
        return System.currentTimeMillis() - lastMoveMs < withinMs;
    }

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

    public long getBusyRemainingMsEstimate() {
        long now = System.currentTimeMillis();
        return Math.max(0L, busyUntilMs - now);
    }

    public String getBusyDebugString() {
        Player p = client.getLocalPlayer();
        int anim = p != null ? p.getAnimation() : -1;
        long now = System.currentTimeMillis();
        boolean timeLocked = now < busyUntilMs;
        boolean recentlyMoved = now - lastMoveMs < 600;
        return "anim=" + anim +
            ", moving=" + recentlyMoved +
            ", dialog=" + isDialogOpen() +
            ", busyMsLeft=" + getBusyRemainingMsEstimate();
    }

    public java.util.Random rng() {
        return deterministicRng;
    }

    public int getInventoryFreeSlots() {
        try {
            ItemContainer inv = client.getItemContainer(InventoryID.INVENTORY);
            if (inv == null) {
                logger.info("[Context] Inventory container is null, assuming empty (28 free slots)");
                return 28;
            }
            Item[] items = inv.getItems();
            if (items == null) {
                logger.info("[Context] Inventory items array is null, assuming empty (28 free slots)");
                return 28;
            }
            
            int occupied = 0;
            for (Item it : items) {
                // Count non-empty slots
                if (it != null) {
                    int id = it.getId();
                    if (id > 0 && id != -1) {
                        occupied++;
                    }
                }
            }
            
            int free = 28 - occupied;
            logger.info("[Context] Inventory check: " + occupied + " occupied, " + free + " free slots");
            return free;
        } catch (Exception e) {
            logger.error("[Context] Error reading inventory: " + e.getMessage());
            return 28; // Default to empty on error
        }
    }

    public boolean isInventoryNearFull() {
        // Consider inventory near full when 5 or fewer free slots remain
        // This allows the bot to bank before completely filling up
        int freeSlots = getInventoryFreeSlots();
        boolean nearFull = freeSlots <= 5; // Bank when 5 or fewer slots remain
        logger.info("[Context] isInventoryNearFull: " + freeSlots + " free slots, nearFull=" + nearFull);
        return nearFull;
    }

    public void updateNavProgress(int currentDistanceTiles) {
        long now = System.currentTimeMillis();
        if (lastNavDistance == null) {
            lastNavDistance = currentDistanceTiles;
            lastNavUpdateMs = now;
            navNoProgressCount = 0;
            return;
        }
        if (currentDistanceTiles <= lastNavDistance - 3) {
            // Made progress
            navNoProgressCount = 0;
            lastNavDistance = currentDistanceTiles;
            lastNavUpdateMs = now;
        } else {
            // No significant progress
            if (now - lastNavUpdateMs > Math.max(3000, config.stuckNoProgressWindowMs())) {
                navNoProgressCount++;
                lastNavUpdateMs = now;
            }
        }
    }

    public int getNavNoProgressCount() {
        return navNoProgressCount;
    }

    public void resetNavProgress() {
        lastNavDistance = null;
        lastNavUpdateMs = 0L;
        navNoProgressCount = 0;
    }

    /**
     * Check if the player is currently walking/moving
     */
    public boolean isPlayerWalking() {
        Player p = client.getLocalPlayer();
        if (p == null) return false;
        
        try {
            // Get all the values we need to pass to Python
            long now = System.currentTimeMillis();
            int anim = p.getAnimation();
            int poseAnim = p.getPoseAnimation();
            int graphic = p.getGraphic();
            boolean isInteracting = p.getInteracting() != null;
            boolean isRunning = client.getVarpValue(173) == 1;
            
            // Call Python script using ProcessBuilder
            ProcessBuilder pb = new ProcessBuilder(
                "python3",
                "runelite/runelite-client/src/main/java/net/runelite/client/plugins/rlbot/tasks/PlayerWalkingDetector.py",
                "is_player_walking",
                String.valueOf(lastMoveMs),
                String.valueOf(anim),
                String.valueOf(poseAnim),
                String.valueOf(graphic),
                String.valueOf(isInteracting),
                String.valueOf(isRunning),
                String.valueOf(now)
            );
            
            Process process = pb.start();
            java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(process.getInputStream())
            );
            
            String result = reader.readLine();
            process.waitFor();
            
            if (result != null && result.trim().equals("True")) {
                return true;
            } else {
                return false;
            }
            
        } catch (Exception e) {
            // Fallback to Java implementation if Python call fails
            logger.warn("Python walking detection failed, falling back to Java implementation: " + e.getMessage());
            return isPlayerWalkingJava();
        }
    }
    
    /**
     * Java fallback implementation of walking detection
     */
    private boolean isPlayerWalkingJava() {
        Player p = client.getLocalPlayer();
        if (p == null) return false;
        
        // Method 1: Position change detection (most reliable)
        long now = System.currentTimeMillis();
        boolean positionChanged = now - lastMoveMs < 2000; // 2 second threshold
        
        // Method 2: Check for specific walking/running animation IDs
        int anim = p.getAnimation();
        boolean isWalkingAnim = anim == 819 || anim == 820 || anim == 821 || anim == 822 || anim == 824;
        
        // Method 3: Check pose animation (might be different from main animation)
        int poseAnim = p.getPoseAnimation();
        boolean hasPoseAnimation = poseAnim != -1;
        
        // Method 4: Check graphic effects (some movement might show graphics)
        int graphic = p.getGraphic();
        boolean hasGraphic = graphic != -1;
        
        // Method 5: Check if player is interacting with something (might indicate movement)
        boolean isInteracting = p.getInteracting() != null;
        
        // Method 6: Check if player is running (from RunHelper)
        boolean isRunning = client.getVarpValue(173) == 1;
        
        // Combine all methods - if any indicate movement, consider player as moving
        return positionChanged || isWalkingAnim || hasPoseAnimation || hasGraphic || isInteracting || isRunning;
    }
    
    public long getLastMoveMs() {
        return lastMoveMs;
    }
    
    /**
     * Get debug info for walking detection
     */
    public String getWalkingDebugInfo() {
        Player p = client.getLocalPlayer();
        if (p == null) return "No player";
        
        try {
            // Get all the values we need to pass to Python
            long now = System.currentTimeMillis();
            int anim = p.getAnimation();
            int poseAnim = p.getPoseAnimation();
            int graphic = p.getGraphic();
            boolean isInteracting = p.getInteracting() != null;
            boolean isRunning = client.getVarpValue(173) == 1;
            String worldLocation = p.getWorldLocation() != null ? p.getWorldLocation().toString() : "null";
            
            // Call Python script using ProcessBuilder
            ProcessBuilder pb = new ProcessBuilder(
                "python3",
                "runelite/runelite-client/src/main/java/net/runelite/client/plugins/rlbot/tasks/PlayerWalkingDetector.py",
                "get_walking_debug_info",
                String.valueOf(lastMoveMs),
                String.valueOf(anim),
                String.valueOf(poseAnim),
                String.valueOf(graphic),
                String.valueOf(isInteracting),
                String.valueOf(isRunning),
                worldLocation,
                String.valueOf(now)
            );
            
            Process process = pb.start();
            java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(process.getInputStream())
            );
            
            String result = reader.readLine();
            process.waitFor();
            
            if (result != null) {
                return result;
            } else {
                return getWalkingDebugInfoJava();
            }
            
        } catch (Exception e) {
            // Fallback to Java implementation if Python call fails
            logger.warn("Python debug info failed, falling back to Java implementation: " + e.getMessage());
            return getWalkingDebugInfoJava();
        }
    }
    
    /**
     * Java fallback implementation of debug info
     */
    private String getWalkingDebugInfoJava() {
        Player p = client.getLocalPlayer();
        if (p == null) return "No player";
        
        long now = System.currentTimeMillis();
        long timeSinceMove = now - lastMoveMs;
        boolean positionChanged = timeSinceMove < 2000;
        
        int anim = p.getAnimation();
        int poseAnim = p.getPoseAnimation();
        int graphic = p.getGraphic();
        boolean isInteracting = p.getInteracting() != null;
        boolean isRunning = client.getVarpValue(173) == 1;
        
        return String.format("Pos:%s Time:%dms A:%d P:%d G:%d Int:%s Run:%s", 
            p.getWorldLocation(), 
            timeSinceMove,
            anim,
            poseAnim,
            graphic,
            isInteracting ? "Y" : "N",
            isRunning ? "Y" : "N");
    }
}


