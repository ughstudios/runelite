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
        
        // Main title with status indicator
        String statusText = "RLBot - Learning";
        Color statusColor = Color.GREEN;
        if (agent != null && config.enableRLAgent()) {
            int epsilonPct = agent.getLastEpsilonPct();
            if (epsilonPct > 80) {
                statusText = "RLBot - Exploring";
                statusColor = Color.YELLOW;
            } else if (epsilonPct > 50) {
                statusText = "RLBot - Learning";
                statusColor = Color.ORANGE;
            } else {
                statusText = "RLBot - Exploiting";
                statusColor = Color.GREEN;
            }
        }
        
        panelComponent.getChildren().add(TitleComponent.builder()
            .text(statusText)
            .color(statusColor)
            .build());
        
        // Learning Progress Section (Most Important)
        if (agent != null && config.enableRLAgent()) {
            renderLearningProgress();
        }
        
        // Current Activity Section
        renderCurrentActivity();
        
        // Quick Stats Section
        renderQuickStats();

        // Q-values Visualization
        renderQValues();
        
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
     * Render a wide, compact visualization of the agent's latest Q-values in 2 columns.
     */
    private void renderQValues() {
        if (agent == null || !config.enableRLAgent()) return;
        // Use real-time Q prediction to avoid overlay staleness
        float[] q = agent.predictQNowSafe();
        if (q == null || q.length == 0) return;

        panelComponent.getChildren().add(TitleComponent.builder()
            .text("Policy")
            .color(new Color(180, 120, 255))
            .build());

        // Compute normalization range
        float min = q[0], max = q[0];
        for (float v : q) { if (v < min) min = v; if (v > max) max = v; }
        float range = Math.max(1e-6f, max - min);

        // Sort indices by descending Q and show ALL actions
        java.util.List<Integer> idx = new java.util.ArrayList<>();
        for (int i = 0; i < q.length; i++) idx.add(i);
        idx.sort((a,b) -> Float.compare(q[b], q[a]));
        int chosen = agent.getLastChosenAction();
        boolean wasExplore = agent.wasLastActionExploratory();

        // Render as a single, readable column to avoid overlap
        for (int k = 0; k < idx.size(); k++) {
            int i = idx.get(k);
            float norm = (q[i] - min) / range;
            String name = shortActionName(agent.getActionName(i));
            // Fixed widths: name(10) bar(12) q(>=5)
            String line = String.format("%-10s %.2f", name, q[i]);
            panelComponent.getChildren().add(LineComponent.builder()
                .left(line)
                .leftColor(getActionColor(chosen, i))
                .build());
        }

        // Show chosen and last actions
        if (chosen >= 0) {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Current:")
                .right(shortActionName(agent.getActionName(chosen)) + (wasExplore ? " (explore)" : " (greedy)"))
                .rightColor(wasExplore ? Color.ORANGE : Color.CYAN)
                .build());
        }
        String lastAction = getLastExecutedAction();
        if (lastAction != null && !lastAction.isEmpty()) {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Last:")
                .right(lastAction)
                .rightColor(Color.YELLOW)
                .build());
        }
    }
    
    /**
     * Get color for action based on whether it's chosen.
     */
    private Color getActionColor(int chosenAction, int actionIndex) {
        if (actionIndex == chosenAction) {
            return Color.GREEN;
        }
        return Color.LIGHT_GRAY;
    }
    
    /**
     * Get the name of the last executed action with timing and result info.
     */
    private String getLastExecutedAction() {
        if (agent == null) return null;
        
        try {
            // Get the most recent action from the recent actions list
            java.util.LinkedList<Integer> recentActions = agent.getRecentActions();
            if (recentActions != null && !recentActions.isEmpty()) {
                Integer lastActionIndex = recentActions.peekLast();
                if (lastActionIndex != null && lastActionIndex >= 0) {
                    String actionName = shortActionName(agent.getActionName(lastActionIndex));
                    
                    // Add current task status for context
                    if (telemetry != null) {
                        String currentTask = telemetry.getMode();
                        if (currentTask != null && !currentTask.isEmpty()) {
                            // Show both the RL action and what task is currently running
                            return actionName + " → " + currentTask;
                        }
                    }
                    
                    // Add action count for this action type
                    int[] actionCounts = agent.getActionCounts();
                    if (actionCounts != null && lastActionIndex < actionCounts.length) {
                        int count = actionCounts[lastActionIndex];
                        return actionName + " (×" + count + ")";
                    }
                    
                    return actionName;
                }
            }
        } catch (Exception ignored) {}
        
        return "None";
    }

    private String qBar(float v01, int width) {
        int n = Math.max(0, Math.min(width, Math.round(v01 * width)));
        StringBuilder sb = new StringBuilder();
        // ASCII-friendly bar to avoid missing glyph boxes in RuneLite font
        for (int i = 0; i < n; i++) sb.append('|');
        for (int i = n; i < width; i++) sb.append('-');
        return sb.toString();
    }

    private String shortActionName(String simpleClass) {
        if (simpleClass == null) return "?";
        String s = simpleClass;
        if (s.contains("BankDeposit")) return "Bank";
        if (s.contains("NavigateToBank")) return "NavBank";
        if (s.contains("CrossWildernessDitchOut")) return "DitchOut";
        if (s.contains("CrossWildernessDitch")) return "Ditch";
        if (s.contains("ChopNearestTree")) return "Chop";
        if (s.contains("NavigateToTree")) return "NavTrees";
        if (s.contains("CameraAdjustment")) return "CamAdj";
        if (s.contains("CameraRotate")) return "CamRot";
        if (s.contains("RandomCameraMovement")) return "CamRnd";
        if (s.contains("Explore")) return "Explore";
        if (s.contains("Idle")) return "Idle";
        return s.length() > 12 ? s.substring(0, 12) : s;
    }

    /**
     * Renders the learning progress section with clear, user-friendly metrics.
     */
    private void renderLearningProgress() {
        panelComponent.getChildren().add(TitleComponent.builder()
            .text("Learning Progress")
            .color(Color.CYAN)
            .build());
        
        // Episode Return (Reward) - Most important metric
        double episodeReturn = agent.getEpisodeReturn();
        String rewardText = String.format("%.1f", episodeReturn);
        Color rewardColor = episodeReturn > 0 ? Color.GREEN : Color.RED;
        if (Math.abs(episodeReturn) < 0.1) rewardColor = Color.YELLOW;
        
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Reward:")
            .right(rewardText)
            .rightColor(rewardColor)
            .build());
        
        // Learning Phase (Exploration vs Exploitation)
        int epsilonPct = agent.getLastEpsilonPct();
        String phaseText;
        Color phaseColor;
        if (epsilonPct > 80) {
            phaseText = "Exploring (" + epsilonPct + "%)";
            phaseColor = Color.YELLOW;
        } else if (epsilonPct > 50) {
            phaseText = "Learning (" + epsilonPct + "%)";
            phaseColor = Color.ORANGE;
        } else {
            phaseText = "Using Knowledge (" + epsilonPct + "%)";
            phaseColor = Color.GREEN;
        }
        
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Phase:")
            .right(phaseText)
            .rightColor(phaseColor)
            .build());
        
        // Training Progress
        Float trainLoss = agent.getLastTrainLoss();
        if (trainLoss != null) {
            String lossText = String.format("%.3f", trainLoss);
            Color lossColor = trainLoss < 0.1 ? Color.GREEN : (trainLoss < 0.5 ? Color.YELLOW : Color.RED);
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Learning:")
                .right(lossText)
                .rightColor(lossColor)
                .build());
        }
        
        // Episode Duration
        long episodeStartMs = agent.getEpisodeStartMs();
        long episodeDurationMs = System.currentTimeMillis() - episodeStartMs;
        long episodeMinutes = episodeDurationMs / 60000;
        long episodeSeconds = (episodeDurationMs % 60000) / 1000;
        
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Episode:")
            .right(String.format("%02d:%02d", episodeMinutes, episodeSeconds))
            .build());
        
        // Total Experience
        long steps = agent.getSteps();
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Experience:")
            .right(formatNumber(steps) + " steps")
            .build());
    }
    
    /**
     * Renders the current activity section with what the bot is doing right now.
     */
    private void renderCurrentActivity() {
        panelComponent.getChildren().add(TitleComponent.builder()
            .text("Current Activity")
            .color(Color.GREEN)
            .build());
        
        // Current action
        if (currentAction != null && !currentAction.isEmpty()) {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Action:")
                .right(currentAction)
                .build());
        }
        
        // Current task from telemetry (condensed)
        if (telemetry != null) {
            String taskName = telemetry.getMode();
            if (taskName != null && !taskName.isEmpty()) {
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("Task:")
                    .right(taskName)
                    .build());
            }
        }
        
        // Inventory status (simplified)
        if (taskContext != null) {
            int freeSlots = taskContext.getInventoryFreeSlots();
            boolean nearFull = taskContext.isInventoryNearFull();
            boolean full = taskContext.isInventoryFull();
            
            String inventoryStatus;
            Color inventoryColor;
            if (full) {
                inventoryStatus = "FULL";
                inventoryColor = Color.RED;
            } else if (nearFull) {
                inventoryStatus = "NEAR FULL";
                inventoryColor = Color.YELLOW;
            } else {
                inventoryStatus = "OK (" + freeSlots + " free)";
                inventoryColor = Color.GREEN;
            }
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Inventory:")
                .right(inventoryStatus)
                .rightColor(inventoryColor)
                .build());
        }
    }
    
    /**
     * Renders quick stats section with essential information.
     */
    private void renderQuickStats() {
        panelComponent.getChildren().add(TitleComponent.builder()
            .text("Quick Stats")
            .color(Color.ORANGE)
            .build());
        
        // Uptime
        Duration uptime = Duration.between(startTime, Instant.now());
        long hours = uptime.toHours();
        long minutes = uptime.toMinutesPart();
        panelComponent.getChildren().add(LineComponent.builder()
            .left("Runtime:")
            .right(String.format("%02d:%02d", hours, minutes))
            .build());
        
        // Performance indicator
        if (agent != null && config.enableRLAgent()) {
            double stepsPerSecond = agent.getStepsPerSecond();
            String perfText = String.format("%.1f", stepsPerSecond) + "/sec";
            Color perfColor = stepsPerSecond > 1.0 ? Color.GREEN : Color.YELLOW;
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Speed:")
                .right(perfText)
                .rightColor(perfColor)
                .build());
        }
        
        // Woodcutting XP
        if (client != null) {
            try {
                int woodcuttingXp = client.getSkillExperience(Skill.WOODCUTTING);
                panelComponent.getChildren().add(LineComponent.builder()
                    .left("WC XP:")
                    .right(formatNumber(woodcuttingXp))
                    .build());
            } catch (Exception ignored) {}
        }
        
        // Movement status
        if (taskContext != null) {
            boolean isWalking = taskContext.isPlayerWalking();
            boolean isChopping = taskContext.isWoodcuttingAnim();
            boolean isStuck = !taskContext.isPlayerMovingRecent(3000) && !isWalking && !isChopping;
            
            String movementStatus;
            Color movementColor;
            if (isStuck) {
                movementStatus = "STUCK";
                movementColor = Color.RED;
            } else if (isWalking) {
                movementStatus = "MOVING";
                movementColor = Color.GREEN;
            } else if (isChopping) {
                movementStatus = "CHOPPING";
                movementColor = Color.GREEN;
            } else {
                movementStatus = "IDLE";
                movementColor = Color.YELLOW;
            }
            
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Movement:")
                .right(movementStatus)
                .rightColor(movementColor)
                .build());
        }
    }
    
    /**
     * Formats large numbers with K/M suffixes for better readability.
     */
    private String formatNumber(long number) {
        if (number >= 1_000_000) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.1fK", number / 1_000.0);
        } else {
            return Long.toString(number);
        }
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