package net.runelite.client.plugins.rlbot.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.BasicStroke;
import java.awt.Shape;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import javax.inject.Inject;
// import lombok.Setter; // Removed for Java 17 compatibility
import net.runelite.client.plugins.rlbot.RLBotTelemetry;
import net.runelite.client.plugins.rlbot.RLBotConfig;
import net.runelite.client.plugins.rlbot.RLBotAgent;
import net.runelite.client.ui.overlay.OverlayPanel;
import net.runelite.client.ui.overlay.OverlayPosition;
import net.runelite.client.ui.overlay.OverlayPriority;
import net.runelite.client.ui.overlay.components.LineComponent;
import net.runelite.client.ui.overlay.components.TitleComponent;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.Player;
import net.runelite.api.AnimationID;
import net.runelite.api.Skill;

/**
 * Overlay for the RLBot plugin.
 * Displays information about the bot's status.
 */
public class RLBotOverlay extends OverlayPanel {
    
    /**
     * The maximum number of actions to display in the history.
     */
    private static final int MAX_ACTION_HISTORY = 5;
    
    /**
     * The duration to display an action in the history.
     */
    private static final Duration ACTION_DISPLAY_DURATION = Duration.ofSeconds(10);
    
    /**
     * The plugin configuration.
     */
    private final RLBotConfig config;
    private final RLBotTelemetry telemetry;
    private final RLBotAgent agent;
    
    /**
     * The current action being performed.
     */
    private String currentAction;
    
    /**
     * Sets the current action being performed.
     */
    public void setCurrentAction(String currentAction) {
        this.currentAction = currentAction;
    }
    
    // REST/WebSocket removed; always show in-process status
    
    /**
     * The time the bot started.
     */
    private final Instant startTime = Instant.now();
    
    /**
     * The list of recent actions.
     */
    private final List<ActionHistoryEntry> actionHistory = new ArrayList<>();
    
    /**
     * Creates a new RLBotOverlay.
     *
     * @param config The plugin configuration
     */
    @Inject
    private Client client;

    @Inject
    private TaskContext taskContext;

    @Inject
    public RLBotOverlay(RLBotConfig config, RLBotTelemetry telemetry, RLBotAgent agent) {
        super();
        this.config = config;
        this.telemetry = telemetry;
        this.agent = agent;
        setPosition(OverlayPosition.TOP_LEFT);
        setPriority(OverlayPriority.LOW);
    }
    
    @Override
    public Dimension render(Graphics2D graphics) {
        if (!config.showOverlay()) {
            return null;
        }
        
        panelComponent.getChildren().clear();
        
        // Add title
        panelComponent.getChildren().add(TitleComponent.builder()
            .text("RLBot")
            .color(Color.GREEN)
            .build());
        
        // Add connection status
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Status:")
            .right("In-Process")
            .rightColor(Color.GREEN)
            .build());
        
        // Add uptime
        Duration uptime = Duration.between(startTime, Instant.now());
        long hours = uptime.toHours();
        long minutes = uptime.toMinutesPart();
        long seconds = uptime.toSecondsPart();
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Uptime:")
            .right(String.format("%02d:%02d:%02d", hours, minutes, seconds))
            .build());
        
        // Add current action
        if (currentAction != null && !currentAction.isEmpty()) {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Current action:")
                .right(currentAction)
                .build());
        }

        // Telemetry
        if (telemetry != null) {
            String taskName = telemetry.getMode();
            if (taskName != null && !taskName.isEmpty()) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Current task:")
                    .right(taskName)
                    .build());
            }
            if (telemetry.getTargetName() != null && !telemetry.getTargetName().isEmpty()) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Target:")
                    .right(telemetry.getTargetName())
                    .build());
            }
            if (telemetry.getDistanceTiles() >= 0) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Dist:")
                    .right(Integer.toString(telemetry.getDistanceTiles()))
                    .build());
            }
            if (telemetry.getBusyRemainingMs() > 0) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Busy ms:")
                    .right(Long.toString(telemetry.getBusyRemainingMs()))
                    .build());
            }
        }
        
        // Agent State Information
        if (agent != null && config.enableRLAgent()) {
            panelComponent.getChildren().add(TitleComponent.builder()
                .text("Agent State")
                .color(Color.CYAN)
                .build());
            
            // RL Training Metrics
            long steps = agent.getSteps();
            double episodeReturn = agent.getEpisodeReturn();
            int epsilonPct = agent.getLastEpsilonPct();
            Float trainLoss = agent.getLastTrainLoss();
            int totalActions = agent.getTotalActions();
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Steps:")
                .right(Long.toString(steps))
                .build());
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Episode Return:")
                .right(String.format("%.2f", episodeReturn))
                .rightColor(episodeReturn > 0 ? Color.GREEN : Color.RED)
                .build());
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Epsilon:")
                .right(epsilonPct + "%")
                .rightColor(epsilonPct > 50 ? Color.YELLOW : Color.GREEN)
                .build());
            
            if (trainLoss != null) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Train Loss:")
                    .right(String.format("%.4f", trainLoss))
                    .build());
            }
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Total Actions:")
                .right(Integer.toString(totalActions))
                .build());
        }
        
        // Inventory Status
        if (taskContext != null) {
            panelComponent.getChildren().add(TitleComponent.builder()
                .text("Inventory")
                .color(Color.ORANGE)
                .build());
            
            int freeSlots = taskContext.getInventoryFreeSlots();
            boolean nearFull = taskContext.isInventoryNearFull();
            boolean full = taskContext.isInventoryFull();
            
            Color inventoryColor = full ? Color.RED : (nearFull ? Color.YELLOW : Color.GREEN);
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Free Slots:")
                .right(Integer.toString(freeSlots))
                .rightColor(inventoryColor)
                .build());
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Status:")
                .right(full ? "FULL" : (nearFull ? "NEAR FULL" : "OK"))
                .rightColor(inventoryColor)
                .build());
        }
        
        // Navigation & Movement Status
        if (taskContext != null) {
            panelComponent.getChildren().add(TitleComponent.builder()
                .text("Navigation")
                .color(Color.MAGENTA)
                .build());
            
            boolean isWalking = taskContext.isPlayerWalking();
            boolean isStuck = !taskContext.isPlayerMovingRecent(3000);
            int navNoProgress = taskContext.getNavNoProgressCount();
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Walking:")
                .right(isWalking ? "YES" : "NO")
                .rightColor(isWalking ? Color.GREEN : Color.RED)
                .build());
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Stuck:")
                .right(isStuck ? "YES" : "NO")
                .rightColor(isStuck ? Color.RED : Color.GREEN)
                .build());
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Nav Failures:")
                .right(Integer.toString(navNoProgress))
                .rightColor(navNoProgress > 3 ? Color.RED : Color.GREEN)
                .build());
            
            // Run energy
            float runEnergy = taskContext.getRunEnergy01();
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Run Energy:")
                .right(String.format("%.0f%%", runEnergy * 100))
                .rightColor(runEnergy > 0.5f ? Color.GREEN : Color.YELLOW)
                .build());
        }
        
        // Goal State Information
        if (client != null && taskContext != null) {
            panelComponent.getChildren().add(TitleComponent.builder()
                .text("Goal State")
                .color(Color.PINK)
                .build());
            
            try {
                Player player = client.getLocalPlayer();
                if (player != null) {
                    boolean inWilderness = player.getWorldLocation().getY() > 3523;
                    panelComponent.getChildren().add(LineComponent.builder()
                        .left("Wilderness:")
                        .right(inWilderness ? "YES" : "NO")
                        .rightColor(inWilderness ? Color.RED : Color.GREEN)
                        .build());
                }
                
                // Current goal based on inventory and location
                String currentGoal = "Find Trees";
                if (taskContext.isInventoryNearFull()) {
                    currentGoal = "Bank Items";
                } else if (player != null && player.getWorldLocation().getY() <= 3523) {
                    currentGoal = "Cross Wilderness";
                }
                
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Current Goal:")
                    .right(currentGoal)
                    .build());
                
                // Woodcutting XP
                try {
                    int woodcuttingXp = client.getSkillExperience(Skill.WOODCUTTING);
                    panelComponent.getChildren().add(LineComponent.builder()
                        .left("WC XP:")
                        .right(Integer.toString(woodcuttingXp))
                        .build());
                } catch (Exception ignored) {}
                
            } catch (Exception ignored) {}
        }
        
        // Performance Metrics
        if (agent != null && config.enableRLAgent()) {
            panelComponent.getChildren().add(TitleComponent.builder()
                .text("Performance")
                .color(Color.BLUE)
                .build());
            
            // Performance metrics using getter methods
            double stepsPerSecond = agent.getStepsPerSecond();
            long episodeStartMs = agent.getEpisodeStartMs();
            long episodeDurationMs = System.currentTimeMillis() - episodeStartMs;
            float actionDiversity = agent.getActionDiversity();
            double efficiency = agent.getEfficiency();
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Steps/sec:")
                .right(String.format("%.2f", stepsPerSecond))
                .rightColor(stepsPerSecond > 1.0 ? Color.GREEN : Color.YELLOW)
                .build());
            
            // Episode duration
            long episodeMinutes = episodeDurationMs / 60000;
            long episodeSeconds = (episodeDurationMs % 60000) / 1000;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Episode Time:")
                .right(String.format("%02d:%02d", episodeMinutes, episodeSeconds))
                .build());
            
            // Action diversity
            if (actionDiversity > 0) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Action Diversity:")
                    .right(String.format("%.2f", actionDiversity))
                    .rightColor(actionDiversity > 0.5f ? Color.GREEN : Color.YELLOW)
                    .build());
            }
            
            // Efficiency metrics
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Efficiency:")
                .right(String.format("%.4f", efficiency))
                .rightColor(efficiency > 0 ? Color.GREEN : Color.RED)
                .build());
        }
        
        // Add animation ID display
        if (client != null && client.getLocalPlayer() != null) {
            int animId = client.getLocalPlayer().getAnimation();
            String animText = Integer.toString(animId);
            Color animColor = animId == -1 ? Color.GREEN : Color.YELLOW;
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Animation:")
                .right(animText)
                .rightColor(animColor)
                .build());
        }
        
        // Add detailed movement detection info
        if (client != null && client.getLocalPlayer() != null && taskContext != null) {
            Player p = client.getLocalPlayer();
            long now = System.currentTimeMillis();
            long timeSinceMove = now - taskContext.getLastMoveMs();
            
            // Position change
            boolean positionChanged = timeSinceMove < 2000;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Position Changed:")
                .right(positionChanged ? "YES" : "NO")
                .rightColor(positionChanged ? Color.GREEN : Color.RED)
                .build());
            
            // Animation
            int anim = p.getAnimation();
            boolean isWalkingAnim = anim == 819 || anim == 820 || anim == 821 || anim == 822 || anim == 824;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Walking Animation:")
                .right(isWalkingAnim ? "YES (" + anim + ")" : "NO")
                .rightColor(isWalkingAnim ? Color.GREEN : Color.RED)
                .build());
            
            // Pose Animation
            int poseAnim = p.getPoseAnimation();
            boolean hasPoseAnimation = poseAnim != -1;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Pose Animation:")
                .right(hasPoseAnimation ? "YES (" + poseAnim + ")" : "NO")
                .rightColor(hasPoseAnimation ? Color.GREEN : Color.RED)
                .build());
            
            // Graphic
            int graphic = p.getGraphic();
            boolean hasGraphic = graphic != -1;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Graphic Effect:")
                .right(hasGraphic ? "YES (" + graphic + ")" : "NO")
                .rightColor(hasGraphic ? Color.GREEN : Color.RED)
                .build());
            
            // Interacting
            boolean isInteracting = p.getInteracting() != null;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Interacting:")
                .right(isInteracting ? "YES" : "NO")
                .rightColor(isInteracting ? Color.GREEN : Color.RED)
                .build());
            
            // Running
            boolean isRunning = client.getVarpValue(173) == 1;
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Running:")
                .right(isRunning ? "YES" : "NO")
                .rightColor(isRunning ? Color.GREEN : Color.RED)
                .build());
            
            // Time since last move
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Time Since Move:")
                .right(timeSinceMove + "ms")
                .rightColor(timeSinceMove < 2000 ? Color.GREEN : Color.RED)
                .build());
            
            // Overall walking detection
            boolean isWalking = taskContext.isPlayerWalking();
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Overall Walking:")
                .right(isWalking ? "YES" : "NO")
                .rightColor(isWalking ? Color.GREEN : Color.RED)
                .build());
        }
        
        // Draw goal object outline (green if unobstructed, red if occluded)
        try {
            if (client != null && taskContext != null) {
                GameObject goal = null;
                String mode = telemetry != null ? telemetry.getMode() : null;
                if (mode != null && mode.toLowerCase().contains("bank")) {
                    goal = net.runelite.client.plugins.rlbot.tasks.ObjectFinder.findNearestByNames(taskContext,
                        new String[]{"bank booth", "bank chest", "bank"}, null);
                } else if (mode != null && (mode.toLowerCase().contains("chop") || mode.toLowerCase().contains("tree"))) {
                    goal = net.runelite.client.plugins.rlbot.tasks.ObjectFinder.findNearestByNames(taskContext,
                        new String[]{"tree", "oak", "willow", "yew", "maple"}, "Chop down");
                }
                if (goal != null) {
                    Shape hull = goal.getConvexHull();
                    if (hull != null) {
                        graphics.setColor(Color.GREEN);
                        graphics.setStroke(new BasicStroke(2f));
                        graphics.draw(hull);
                    }
                }
            }
        } catch (Exception ignored) {}

        // Add action history
        if (!actionHistory.isEmpty()) {
            panelComponent.getChildren().add(TitleComponent.builder()
                .text("Recent Actions")
                .color(Color.YELLOW)
                .build());
            
            // Clean up old actions
            Instant now = Instant.now();
            actionHistory.removeIf(entry -> Duration.between(entry.timestamp, now).compareTo(ACTION_DISPLAY_DURATION) > 0);
            
            // Add recent actions to the panel
            for (int i = 0; i < actionHistory.size() && i < MAX_ACTION_HISTORY; i++) {
                ActionHistoryEntry entry = actionHistory.get(i);
                panelComponent.getChildren().add(LineComponent.builder()
                    .left(entry.action)
                    .right(formatTimestamp(entry.timestamp))
                    .rightColor(Color.LIGHT_GRAY)
                    .build());
            }
        }
        
        return super.render(graphics);
    }
    
    /**
     * Adds an action to the history.
     *
     * @param action The action to add
     */
    public void addAction(String action) {
        if (action == null || action.isEmpty()) {
            return;
        }
        
        actionHistory.add(0, new ActionHistoryEntry(action, Instant.now()));
        
        // Limit the history size
        while (actionHistory.size() > MAX_ACTION_HISTORY) {
            actionHistory.remove(actionHistory.size() - 1);
        }
    }
    
    /**
     * Formats a timestamp as "mm:ss" ago.
     *
     * @param timestamp The timestamp to format
     * @return The formatted timestamp
     */
    private String formatTimestamp(Instant timestamp) {
        Duration elapsed = Duration.between(timestamp, Instant.now());
        long minutes = elapsed.toMinutes();
        long seconds = elapsed.toSecondsPart();
        
        if (minutes > 0) {
            return String.format("%dm %ds ago", minutes, seconds);
        } else {
            return String.format("%ds ago", seconds);
        }
    }
    
    /**
     * Entry for the action history.
     */
    private static class ActionHistoryEntry {
        private final String action;
        private final Instant timestamp;
        
        private ActionHistoryEntry(String action, Instant timestamp) {
            this.action = action;
            this.timestamp = timestamp;
        }
    }
} 