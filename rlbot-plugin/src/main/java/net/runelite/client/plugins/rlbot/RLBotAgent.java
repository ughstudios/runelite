package net.runelite.client.plugins.rlbot;

import com.google.inject.Singleton;
import javax.inject.Inject;
import net.runelite.api.Client;
import net.runelite.api.GameState;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.ui.DrawManager;
import net.runelite.client.config.ConfigManager;
// RLBotGameStateGenerator import removed
import net.runelite.client.plugins.rlbot.input.RLBotInputHandler;
import net.runelite.client.plugins.rlbot.tasks.*;
import java.util.Random;
import net.runelite.client.plugins.rlbot.tasks.ObjectFinder;
import net.runelite.api.GameObject;
import net.runelite.client.plugins.rlbot.rewards.LogQualityRewards;
import java.util.concurrent.ConcurrentLinkedQueue;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.Player;

@Singleton
public class RLBotAgent {

    private final Client client;
    // private final ClientThread clientThread;
    private final RLBotLogger logger;
    private final RLBotConfig config;
    private final ConfigManager configManager;
    // private final RLBotInputHandler inputHandler;
    // private final RLBotGameStateGenerator gameStateGenerator;
    private final RLBotTelemetry telemetry;

    private long lastRunMillis;
    private TaskContext taskContext;
    private java.util.List<Task> tasks;
    private final Random rng;
    private net.runelite.client.plugins.rlbot.ipc.ExternalControlBridge external;
    private Float externalLastReward = 0.0f;
    private long steps = 0;
    private Integer prevFreeSlots = null;
    private Boolean prevBankOpen = null;
    private long episodeStartMs = 0L;
    private double episodeReturn = 0.0;
    private Integer lastActionIndex = null;
    private Integer prevTreeDist = null;
    private Integer prevBankDist = null;
    private Integer prevWoodcutXp = null;
    private int[] actionCounts = null;
    private long lastStatusLogStep = 0L;
    private java.util.LinkedList<Integer> recentActions = new java.util.LinkedList<>();
    private Boolean prevTreeVisible = null;
    private Boolean prevBankVisible = null;
    private Boolean prevDitchVisible = null;

    private int lastChosenAction = -1;

    private final ConcurrentLinkedQueue<Task> manualTasks = new ConcurrentLinkedQueue<>();

    @Inject
    public RLBotAgent(
        Client client,
        ClientThread clientThread,
        RLBotLogger logger,
        RLBotConfig config,
        ConfigManager configManager,
        RLBotInputHandler inputHandler,
        DrawManager drawManager,
        RLBotTelemetry telemetry
    ) {
        this.client = client;
        // this.clientThread = clientThread;
        this.logger = logger;
        this.config = config;
        this.configManager = configManager;
        // this.inputHandler = inputHandler;
        this.telemetry = telemetry;
        // this.gameStateGenerator = new RLBotGameStateGenerator(client, logger, new RLBotScreenshotUtil(drawManager, logger, config));
        this.lastRunMillis = 0L;
        this.taskContext = new TaskContext(client, clientThread, logger, inputHandler, config, telemetry);
        this.rng = new Random(config.rngSeed());
        this.tasks = java.util.Arrays.asList(
            new BankDepositTask(),                 // 0
            new NavigateToBankHotspotTask(),       // 1
            new ChopNearestTreeTask(),             // 2 (Chop)
            new NavigateToTreeHotspotTask(),       // 3
            new ExploreTask(),                     // 4
            // Camera controls
            new CameraLeftTask(),                  // 5
            new CameraRightTask(),                 // 6
            new CameraUpTask(),                    // 7
            new CameraDownTask(),                  // 8
            new CameraZoomInTask(),                // 9
            new CameraZoomOutTask(),               // 10
            new IdleTask()                         // 11
        );
        this.episodeStartMs = System.currentTimeMillis();
    }

    public void onTick() {
        if (client.getGameState() != GameState.LOGGED_IN) {
            return;
        }

        // Run any manually triggered task first (one per tick), regardless of RL toggle
        Task manual = manualTasks.poll();
        if (manual != null) {
            runTaskSafe(manual);
            return;
        }

        // External control only
        if (!config.enableGymControl()) {
            return;
        }

        // External control path (OpenAI Gym / IPC)
        {
            long now = System.currentTimeMillis();
            int interval = Math.max(100, config.gymStepIntervalMs());
            if (now - lastRunMillis < interval) {
                return;
            }
            lastRunMillis = now;
            // Initialize bridge and action space lazily
            if (external == null) {
                external = new net.runelite.client.plugins.rlbot.ipc.ExternalControlBridge(logger, config.gymIpcDir());
            }

            float[] state = buildDjlStateVector();
            try {
                java.util.List<String> actionNames = new java.util.ArrayList<>();
                for (Task t : tasks) actionNames.add(t.getClass().getSimpleName());
                external.writeActionSpaceIfNeeded(actionNames, state.length);
            } catch (Exception ignored) {}

            // Compute reward for the last action and publish observation
            float rewardNow = computeImmediateReward();
            externalLastReward = rewardNow;
            Integer lastIdx = lastActionIndex;
            String lastName = (lastIdx != null && lastIdx >= 0 && lastIdx < tasks.size()) ? tasks.get(lastIdx).getClass().getSimpleName() : null;
            external.publishObservation(state, rewardNow, lastIdx, lastName);

            // Read and execute next action if provided
            Integer act = external.tryReadAction();
            if (act != null) {
                int actionIndex = Math.max(0, Math.min(tasks.size() - 1, act));
                Task t = tasks.get(actionIndex);
                lastActionIndex = actionIndex;
                if (steps == 0) episodeStartMs = System.currentTimeMillis();
                actionCounts = (actionCounts == null ? new int[tasks.size()] : actionCounts);
                actionCounts[actionIndex]++;
                telemetry.setMode("External");
                runTaskSafe(t);
                steps++;
            }
            return; // End of Gym-controlled tick
        }
    }

    // Public triggers for UI buttons
    public void triggerNavigateTrees() { logger.info("[UI] Enqueue NavigateToTreeHotspotTask"); manualTasks.offer(new NavigateToTreeHotspotTask()); }
    public void triggerChopTree() { logger.info("[UI] Enqueue ChopNearestTreeTask"); manualTasks.offer(new ChopNearestTreeTask()); }
    public void triggerNavigateBank() { logger.info("[UI] Enqueue NavigateToBankHotspotTask"); manualTasks.offer(new NavigateToBankHotspotTask()); }
    public void triggerBankDeposit() { logger.info("[UI] Enqueue BankDepositTask"); manualTasks.offer(new BankDepositTask()); }
    public void triggerCrossWilderness() { logger.info("[UI] Enqueue CrossWildernessDitchTask"); manualTasks.offer(new CrossWildernessDitchTask()); }
    public void triggerCrossWildernessOut() { logger.info("[UI] Enqueue CrossWildernessDitchOutTask"); manualTasks.offer(new CrossWildernessDitchOutTask()); }
    public void triggerExplore() { logger.info("[UI] Enqueue ExploreTask"); manualTasks.offer(new ExploreTask()); }
    public void triggerRotateCamera() { logger.info("[UI] Enqueue CameraRotateTask"); manualTasks.offer(new CameraRotateTask()); }
    public void triggerRandomCameraMovement() { logger.info("[UI] Enqueue RandomCameraMovementTask"); manualTasks.offer(new RandomCameraMovementTask()); }

    public void triggerRotateDirection(String dir, int steps) {
        logger.info("[UI] Enqueue rotate direction dir=" + dir + " steps=" + steps);
        CameraRotateStepTask.Direction d;
        try { d = CameraRotateStepTask.Direction.valueOf(dir); } catch (Exception e) { d = CameraRotateStepTask.Direction.LEFT; }
        int n = Math.max(1, Math.min(steps, 20));
        for (int i = 0; i < n; i++) {
            manualTasks.offer(new CameraRotateStepTask(d));
        }
    }

    public void triggerExploreCardinal(String cardinal, int tiles) {
        logger.info("[UI] Enqueue explore cardinal dir=" + cardinal + " tiles=" + tiles);
        ExploreStepTask.Cardinal c;
        try { c = ExploreStepTask.Cardinal.valueOf(cardinal); } catch (Exception e) { c = ExploreStepTask.Cardinal.NORTH; }
        manualTasks.offer(new ExploreStepTask(c, tiles));
    }

    // Manual test: Click at canvas X/Y using same validated flow as tasks
    public void triggerClickAtXY(int x, int y, String expectedActionLabel) {
        logger.info("[UI] Manual ClickAtXY requested at (" + x + "," + y + ") with action='" + expectedActionLabel + "'");
        try {
            java.awt.Point p = new java.awt.Point(Math.max(0, x), Math.max(0, y));
            // Ensure we execute on the RuneLite client thread due to scene/widget access in validation
            taskContext.clientThread.invoke(() -> {
                try {
                    boolean ok = taskContext.input.moveAndClickWithValidation(p, expectedActionLabel != null ? expectedActionLabel : "Chop down");
                    logger.info("[UI] Manual ClickAtXY result=" + ok);
                    if (ok) taskContext.markCanvasClick(); else taskContext.markMenuWalkClick();
                } catch (Exception ex) {
                    logger.error("[UI] Manual ClickAtXY failed on client thread: " + ex.getMessage());
                }
            });
        } catch (Exception e) {
            logger.error("[UI] Manual ClickAtXY scheduling failed: " + e.getMessage());
        }
    }

    private void runTaskSafe(Task t) {
        try {
            // Soft gating: if not eligible, penalize but still allow execution so RL can learn
            boolean eligible = true;
            try { eligible = t.shouldRun(taskContext); } catch (Exception ignored) {}
            if (!eligible) {
                logger.info("[Agent] Task not eligible now (soft): " + t.getClass().getSimpleName());
                addExternalPenalty(0.2f); // small penalty to teach policy
            }
            if (t instanceof NavigateToBankHotspotTask) telemetry.setMode("Navigate: Bank");
            else if (t instanceof NavigateToTreeHotspotTask) telemetry.setMode("Navigate: Trees");
            else if (t instanceof BankDepositTask) telemetry.setMode("Banking");
            else if (t instanceof ChopNearestTreeTask) telemetry.setMode("Chop");
            else telemetry.setMode("Act");
            // Allow manual tasks to run immediately even if moving
            boolean isManual = (manualTasks != null && !manualTasks.isEmpty()) ||
                               (t instanceof ChopNearestTreeTask && lastActionIndex == null);
            if (isManual) {
                taskContext.clearBusyLock();
            }
            // Guard: block automated clicks while moving to reduce misclicks
            if (!isManual && taskContext.isPlayerWalking() && (t instanceof ChopNearestTreeTask || t instanceof BankDepositTask)) {
                logger.info("[Agent] Skipping automated click task while moving");
            } else {
                long t0 = System.nanoTime();
                t.run(taskContext);
                long t1 = System.nanoTime();
                logger.perf(t.getClass().getSimpleName() + " took " + ((t1 - t0) / 1_000_000) + " ms");
            }
            logger.info("[Agent] Completed tick for task=" + t.getClass().getSimpleName());
        } catch (Exception e) {
            logger.error("Task error in " + t.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    // Internal RL policy removed

    // phaseBank hysteresis removed per requirement to avoid hard-coded strategies

    // computeStateKey removed (unused)

    private float[] buildDjlStateVector() {
        int free = Math.max(0, Math.min(28, taskContext.getInventoryFreeSlots()));
        float freeBucket = free / 4.0f;
        boolean bankOpen = taskContext.client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER) != null;
        float movingRecent = taskContext.isPlayerMovingRecent(600) ? 1f : 0f;
        float runEnergy = taskContext.getRunEnergy01();
        
        // === CAMERA STATE ===
        int yaw = client.getCameraYawTarget() & 2047;
        float yawSin = (float)Math.sin(yaw * (2*Math.PI/2048.0));
        float yawCos = (float)Math.cos(yaw * (2*Math.PI/2048.0));
        int pitch = client.getCameraPitchTarget();
        float pitch01 = Math.max(0f, Math.min(1f, (pitch - 128) / 384f));
        
        // === DISTANCE STATE ===
        int tDist = distanceToNearestTree();
        int bDist = distanceToNearestBank();
        int dDist = distanceToNearestDitch();
        float tBucket = tDist < 0 ? 10f : Math.min(10f, tDist / 5.0f);
        float bBucket = bDist < 0 ? 10f : Math.min(10f, bDist / 5.0f);
        float dBucket = dDist < 0 ? 10f : Math.min(10f, dDist / 5.0f);
        
        // === BEARING STATE ===
        float[] treeBearing = bearingToNearestTree();
        float[] bankBearing = bearingToNearestBank();
        float[] ditchBearing = bearingToNearestDitch();
        
        // === VISIBILITY STATE ===
        boolean treeVisible = isAnyTreeVisible();
        boolean bankVisible = isAnyBankVisible();
        boolean ditchVisible = isAnyDitchVisible();
        float treeVis = treeVisible ? 1f : 0f;
        float bankVis = bankVisible ? 1f : 0f;
        float ditchVis = ditchVisible ? 1f : 0f;
        
        // === PROGRESS STATE ===
        float woodcuttingAnim = taskContext.isWoodcuttingAnim() ? 1f : 0f;
        float inventoryNearFull = taskContext.isInventoryNearFull() ? 1f : 0f;
        float isStuck = (!taskContext.isPlayerMovingRecent(3000)) ? 1f : 0f; // Stuck if not moving for 3 seconds
        float navNoProgress = Math.min(5f, taskContext.getNavNoProgressCount()) / 5f; // Normalize navigation failures
        
        // === ACTION HISTORY STATE ===
        float recentActionDiversity = 0f;
        if (recentActions.size() >= 3) {
            // Calculate diversity of recent actions (lower = more repetitive)
            java.util.Set<Integer> uniqueActions = new java.util.HashSet<>(recentActions);
            recentActionDiversity = (float) uniqueActions.size() / recentActions.size();
        }
        
        // === GOAL STATE ===
        float currentGoal = 0f; // 0 = find trees, 1 = go to bank, 2 = cross wilderness
        if (inventoryNearFull > 0.5f) {
            currentGoal = 1f; // Goal is to bank
        } else if (dDist >= 0 && dDist <= 5) {
            currentGoal = 2f; // Goal is to cross wilderness
        } else {
            currentGoal = 0f; // Goal is to find trees
        }
        
        // === CLICK SUCCESS STATE ===
        float lastClickSuccess = taskContext.wasLastClickCanvasRecent(800) ? 1f : 0f;
        
        // === WILDERNESS STATE ===
        float inWilderness = 0f;
        try {
            Player player = client.getLocalPlayer();
            if (player != null) {
                inWilderness = player.getWorldLocation().getY() > 3523 ? 1f : 0f;
            }
        } catch (Exception ignored) {}
        
        // === OBSTRUCTION STATE ===
        float obstructionAttempts = 0f;
        if (lastActionIndex != null && lastActionIndex == 2) { // ChopNearestTreeTask
            // If we just tried to chop but aren't woodcutting and not moving, likely obstruction
            if (!taskContext.isWoodcuttingAnim() && !taskContext.isPlayerMovingRecent(1000)) {
                obstructionAttempts = 1f; // Signal obstruction state
            }
        }
        
        // === CLICK ALIGNMENT STATE ===
        float clickAlignmentIssue = 0f;
        if (lastActionIndex != null && lastActionIndex == 2) { // ChopNearestTreeTask
            // If we clicked recently but aren't woodcutting, likely click alignment issue
            if (!taskContext.isWoodcuttingAnim() && taskContext.wasLastClickCanvasRecent(500)) {
                clickAlignmentIssue = 1f; // Signal click alignment problem
            }
        }
        
        // === TREE TYPE STATE === (NEW: Enhanced for tree-specific learning)
        float[] treeInfo = getNearestTreeInfo();
        float nearestTreeTier = treeInfo[0];        // 0-1: Tree quality tier (1-9 normalized)
        float nearestTreeLevelReq = treeInfo[1];   // 0-1: Required WC level (1-90 normalized) 
        float playerWcLevel = treeInfo[2];         // 0-1: Player's WC level (1-90 normalized)
        
        return new float[] {
            freeBucket, bankOpen ? 1f : 0f, movingRecent, runEnergy,
            yawSin, yawCos, pitch01,
            tBucket, bBucket, dBucket,
            treeBearing[0], treeBearing[1], bankBearing[0], bankBearing[1], ditchBearing[0], ditchBearing[1],
            treeVis, bankVis, ditchVis,
            woodcuttingAnim, inventoryNearFull, isStuck, navNoProgress,
            recentActionDiversity, currentGoal, lastClickSuccess, inWilderness, obstructionAttempts, clickAlignmentIssue,
            nearestTreeTier, nearestTreeLevelReq, playerWcLevel  // NEW: Tree-specific features
        };
    }

    private boolean isAnyTreeVisible() {
        try {
            int wc = 1; try { wc = client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
            String[] allowed = net.runelite.client.plugins.rlbot.tasks.TreeDiscovery.allowedTreeNamesForLevel(wc);
            GameObject go = ObjectFinder.findNearestByNames(taskContext, allowed, "Chop down");
            if (go == null) return false;
            
            // Double-check that this tree actually has a chop action
            try {
                net.runelite.api.ObjectComposition comp = client.getObjectDefinition(go.getId());
                if (comp != null && comp.getActions() != null) {
                    boolean hasChopAction = false;
                    for (String action : comp.getActions()) {
                        if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                            hasChopAction = true;
                            break;
                        }
                    }
                    if (!hasChopAction) {
                        net.runelite.client.plugins.rlbot.tasks.TreeDiscovery.markDepleted(go.getWorldLocation());
                        return false;
                    }
                }
            } catch (Exception ignored) {}
            
            return ObjectFinder.projectToCanvas(taskContext, go) != null;
        } catch (Exception ignored) { return false; }
    }

    private boolean isAnyBankVisible() {
        try {
            GameObject go = ObjectFinder.findNearestByNames(taskContext, new String[]{"bank booth", "bank chest", "bank"}, null);
            if (go == null) return false;
            return ObjectFinder.projectToCanvas(taskContext, go) != null;
        } catch (Exception ignored) { return false; }
    }

    private boolean isAnyDitchVisible() {
        try {
            GameObject go = ObjectFinder.findNearestByNames(taskContext, new String[]{"wilderness ditch"}, "Cross");
            if (go == null) return false;
            return ObjectFinder.projectToCanvas(taskContext, go) != null;
        } catch (Exception ignored) { return false; }
    }

    private int distanceToNearestTree() {
        try {
            int wc = 1; try { wc = client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
            String[] allowed = net.runelite.client.plugins.rlbot.tasks.TreeDiscovery.allowedTreeNamesForLevel(wc);
            GameObject go = ObjectFinder.findNearestByNames(taskContext, allowed, "Chop down");
            if (go == null || client.getLocalPlayer() == null) return -1;
            
            // Double-check that this tree actually has a chop action
            try {
                net.runelite.api.ObjectComposition comp = client.getObjectDefinition(go.getId());
                if (comp != null && comp.getActions() != null) {
                    boolean hasChopAction = false;
                    for (String action : comp.getActions()) {
                        if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                            hasChopAction = true;
                            break;
                        }
                    }
                    if (!hasChopAction) {
                        net.runelite.client.plugins.rlbot.tasks.TreeDiscovery.markDepleted(go.getWorldLocation());
                        return -1;
                    }
                }
            } catch (Exception ignored) {}
            
            return client.getLocalPlayer().getWorldLocation().distanceTo(go.getWorldLocation());
        } catch (Exception ignored) { return -1; }
    }

    private int distanceToNearestBank() {
        try {
            GameObject go = ObjectFinder.findNearestByNames(taskContext, new String[]{"bank booth", "bank chest", "bank"}, null);
            if (go == null || client.getLocalPlayer() == null) return -1;
            return client.getLocalPlayer().getWorldLocation().distanceTo(go.getWorldLocation());
        } catch (Exception ignored) { return -1; }
    }

    private int distanceToNearestDitch() {
        try {
            GameObject go = ObjectFinder.findNearestByNames(taskContext, new String[]{"wilderness ditch"}, "Cross");
            if (go == null || client.getLocalPlayer() == null) return -1;
            return client.getLocalPlayer().getWorldLocation().distanceTo(go.getWorldLocation());
        } catch (Exception ignored) { return -1; }
    }

    private float[] bearingTo(WorldPoint to) {
        try {
            if (to == null || client.getLocalPlayer() == null) return new float[] {0f, 0f};
            WorldPoint me = client.getLocalPlayer().getWorldLocation();
            double ang = Math.atan2(to.getY() - me.getY(), to.getX() - me.getX());
            return new float[] {(float)Math.sin(ang), (float)Math.cos(ang)};
        } catch (Exception ignored) { return new float[] {0f, 0f}; }
    }

    private float[] bearingToNearestTree() {
        try {
            int wc = 1; try { wc = client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
            String[] allowed = net.runelite.client.plugins.rlbot.tasks.TreeDiscovery.allowedTreeNamesForLevel(wc);
            GameObject go = ObjectFinder.findNearestByNames(taskContext, allowed, "Chop down");
            if (go == null) return new float[] {0f, 0f};
            
            // Double-check that this tree actually has a chop action
            try {
                net.runelite.api.ObjectComposition comp = client.getObjectDefinition(go.getId());
                if (comp != null && comp.getActions() != null) {
                    boolean hasChopAction = false;
                    for (String action : comp.getActions()) {
                        if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                            hasChopAction = true;
                            break;
                        }
                    }
                    if (!hasChopAction) {
                        net.runelite.client.plugins.rlbot.tasks.TreeDiscovery.markDepleted(go.getWorldLocation());
                        return new float[] {0f, 0f};
                    }
                }
            } catch (Exception ignored) {}
            
            return bearingTo(go.getWorldLocation());
        } catch (Exception ignored) { return new float[] {0f, 0f}; }
    }

    private float[] bearingToNearestBank() {
        try {
            GameObject go = ObjectFinder.findNearestByNames(taskContext, new String[]{"bank booth", "bank chest", "bank"}, null);
            return go == null ? new float[] {0f, 0f} : bearingTo(go.getWorldLocation());
        } catch (Exception ignored) { return new float[] {0f, 0f}; }
    }

    private float[] bearingToNearestDitch() {
        try {
            GameObject go = ObjectFinder.findNearestByNames(taskContext, new String[]{"wilderness ditch"}, "Cross");
            return go == null ? new float[] {0f, 0f} : bearingTo(go.getWorldLocation());
        } catch (Exception ignored) { return new float[] {0f, 0f}; }
    }

    private float computeImmediateReward() {
        int curFree = taskContext.getInventoryFreeSlots();
        boolean curBankOpen = curBankOpen();
        float r = 0.0f; // Start with neutral reward
        
        // === PROGRESS REWARDS ===
        
        // Major reward for woodcutting (the main goal)
        if (taskContext.isWoodcuttingAnim()) {
            r += 2.0f; // Significant reward for productive woodcutting
            // Only log woodcutting reward occasionally to reduce spam
            if (steps % 20 == 0) {
                logger.info("[RL] WOODCUTTING REWARD: Agent is actively woodcutting (+2.0 reward)");
            }
        } else {
            // Small penalty for not woodcutting when inventory has space
            if (curFree > 0) {
                r -= 0.1f; // Encourage finding trees to chop
            }
        }
        
        // Major reward for XP gains (measurable progress) - ENHANCED WITH QUALITY REWARDS
        try {
            int curXp = client.getSkillExperience(net.runelite.api.Skill.WOODCUTTING);
            if (prevWoodcutXp != null && curXp > prevWoodcutXp) {
                int xpGained = curXp - prevWoodcutXp;
                // Use quality-based XP reward system that favors higher-value logs
                float qualityXpReward = LogQualityRewards.calculateExperienceReward(xpGained);
                r += qualityXpReward;
                logger.info("[RL] XP reward: " + xpGained + " XP -> " + String.format("%.2f", qualityXpReward) + " reward");
            }
        } catch (Exception ignored) {}
        
        // Major reward for gaining logs (inventory progress) - ENHANCED WITH QUALITY REWARDS
        if (prevFreeSlots != null) {
            int delta = prevFreeSlots - curFree; // +1 when we gained a log
            if (delta > 0) {
                // Use quality-based log reward system that favors higher-tier logs
                float qualityLogReward = LogQualityRewards.calculateLogQualityReward(client, delta);
                r += qualityLogReward;
                logger.info("[RL] Log reward: " + delta + " logs gained -> " + String.format("%.2f", qualityLogReward) + " reward");
            }
            
            // Major reward for successful banking (inventory emptying)
            int emptyDelta = curFree - prevFreeSlots; // Positive when inventory empties
            if (emptyDelta > 10) { // Significant inventory emptying indicates successful banking
                float bankingSuccessReward = 3.0f; // Large reward for successful banking
                r += bankingSuccessReward;
                logger.info("[RL] BANKING SUCCESS: Inventory emptied by " + emptyDelta + " slots (+3.0 reward)");
            }
        }
        
        // Major reward for successful banking (completing the cycle)
        if (prevBankOpen != null && prevBankOpen && curBankOpen && prevFreeSlots != null && curFree > prevFreeSlots) {
            int deposited = curFree - prevFreeSlots;
            if (deposited > 0) r += Math.min(10.0f, deposited * 1.5f); // Very large reward for successful banking
        }
        
        // Reward for having bank UI open when inventory is full (encourages opening banks)
        if (curBankOpen && taskContext.isInventoryFull()) {
            r += 1.0f; // Good reward for having bank open when needed
            if (steps % 20 == 0) { // Log occasionally to avoid spam
                logger.info("[RL] BANK UI REWARD: Bank open with full inventory (+1.0 reward)");
            }
        }
        
        // === DISTANCE-BASED REWARDS ===
        
        // Strong rewards for moving toward the current goal
        boolean needBank = taskContext.isInventoryNearFull();
        if (needBank) {
            int d = distanceToNearestBank();
            if (prevBankDist != null && d >= 0 && prevBankDist >= 0) {
                if (d < prevBankDist) {
                    r += 0.5f; // Good reward for moving closer to bank
                } else if (d > prevBankDist) {
                    r -= 0.3f; // Penalty for moving away from bank
                }
            }
        } else {
            int d = distanceToNearestTree();
            if (prevTreeDist != null && d >= 0 && prevTreeDist >= 0) {
                if (d < prevTreeDist) {
                    r += 0.5f; // Good reward for moving closer to trees
                } else if (d > prevTreeDist) {
                    r -= 0.3f; // Penalty for moving away from trees
                }
            }
        }
        
        // === ENHANCED BANK NAVIGATION REWARDS ===
        
        // Reward navigating towards bank when inventory is full
        if (lastActionIndex != null && lastActionIndex == 1) { // NavigateToBankHotspotTask
            if (taskContext.isInventoryFull()) {
                // Strong reward for navigating to bank when inventory is completely full
                r += 1.5f; // Increased from 1.0f to make it more attractive
                if (steps % 20 == 0) {
                    logger.info("[RL] BANK NAVIGATION REWARD: Navigating to bank with full inventory (+1.5 reward)");
                }
                
                // Additional reward for making progress towards bank
                if (taskContext.isPlayerMovingRecent(1000)) {
                    r += 0.5f; // Bonus for actually moving towards bank
                    if (steps % 20 == 0) {
                        logger.info("[RL] BANK MOVEMENT BONUS: Making progress towards bank (+0.5 reward)");
                    }
                }
            } else if (taskContext.isInventoryNearFull()) {
                // Moderate reward for navigating to bank when inventory is nearly full
                r += 0.8f; // Increased from 0.5f
                if (steps % 20 == 0) {
                    logger.info("[RL] BANK NAVIGATION REWARD: Navigating to bank with nearly full inventory (+0.8 reward)");
                }
            } else {
                // Penalty for navigating to bank when inventory has plenty of space
                r -= 1.2f; // Increased penalty from 0.8f
                if (steps % 20 == 0) {
                    logger.info("[RL] BANK NAVIGATION PENALTY: Navigating to bank with inventory space (-1.2 reward)");
                }
            }
        }
        
        // === BANK DEPOSIT SUCCESS REWARDS ===
        
        // Reward successful bank deposits
        if (lastActionIndex != null && lastActionIndex == 0) { // BankDepositTask
            boolean bankOpen = curBankOpen();
            boolean inventoryFull = taskContext.isInventoryFull();
            boolean inventoryEmpty = taskContext.getInventoryFreeSlots() >= 28;
            
            if (bankOpen && inventoryEmpty) {
                // Major reward for successfully depositing all items
                r += 2.0f;
                if (steps % 20 == 0) {
                    logger.info("[RL] BANK DEPOSIT SUCCESS: Successfully deposited all items (+2.0 reward)");
                }
            } else if (bankOpen && !inventoryFull) {
                // Moderate reward for opening bank and depositing some items
                r += 1.0f;
                if (steps % 20 == 0) {
                    logger.info("[RL] BANK DEPOSIT PARTIAL: Deposited some items (+1.0 reward)");
                }
            } else if (!bankOpen && inventoryFull) {
                // Small penalty for failing to open bank when inventory is full
                r -= 0.3f;
                if (steps % 20 == 0) {
                    logger.info("[RL] BANK OPEN FAILURE: Failed to open bank with full inventory (-0.3 reward)");
                }
            }
        }
        
        // === WILDERNESS DITCH REWARDS (further reduced to avoid obsession) ===
        // Penalize ditch actions unless they are clearly necessary: standing adjacent, and crossing OUT while
        // in wilderness with a banking goal. Never hard-gate.
        try {
            boolean inWild = false;
            Player player = client.getLocalPlayer();
            if (player != null) {
                inWild = player.getWorldLocation().getY() > 3523;
            }
            // reuse previously computed 'needBank' from distance-based rewards above
            int dd = distanceToNearestDitch();
            if (lastActionIndex != null && (lastActionIndex == 98 || lastActionIndex == 99)) { // Ditch tasks removed
                boolean targetVisible = isAnyBankVisible() || isAnyTreeVisible();
                if (dd < 0 || dd > 2) {
                    r -= 1.0f; // strong penalty when not adjacent to ditch
                }
                if (targetVisible) {
                    r -= 0.6f; // don't abandon visible targets for ditch
                }
                // Only small reward for the specific helpful case: banking goal while inside wilderness and using OUT task
                if (inWild && needBank && lastActionIndex == 3 && dd >= 0 && dd <= 2) {
                    r += 0.4f;
                } else {
                    r -= 0.2f; // default discourage
                }
            } else {
                // Mild penalty for hovering near ditch without reason to reduce fixation
                if (dd >= 0 && dd <= 2 && !needBank) {
                    r -= 0.05f;
                }
            }
        } catch (Exception ignored) {}
        
        // === PENALTIES FOR GETTING STUCK ===
        
        // Major penalty for being stuck (not moving for extended periods)
        if (!taskContext.isPlayerMovingRecent(5000)) { // 5 seconds
            r -= 1.0f; // Significant penalty for being stuck
        }
        
        // === CAMERA ROTATION LEARNING ===
        // Reward only when camera reveals new targets for the current goal; otherwise penalize (no gating)
        if (lastActionIndex != null) {
            boolean isCameraAction = lastActionIndex >= 7 && lastActionIndex <= 12;
            if (isCameraAction) {
                boolean curTreeVis = isAnyTreeVisible();
                boolean curBankVis = isAnyBankVisible();
                boolean curDitchVis = isAnyDitchVisible();
                boolean wasStuck = !taskContext.isPlayerMovingRecent(3000);

                boolean revealedNew = (prevTreeVisible != null && !prevTreeVisible && curTreeVis)
                                    || (prevBankVisible != null && !prevBankVisible && curBankVis)
                                    || (prevDitchVisible != null && !prevDitchVisible && curDitchVis);

                boolean targetsAlreadyVisible = (prevTreeVisible != null && prevTreeVisible)
                                             || (prevBankVisible != null && prevBankVisible)
                                             || (prevDitchVisible != null && prevDitchVisible);

                // Align camera rewards with current goal (needBank from above)
                if (needBank) {
                    if (prevBankVisible != null && !prevBankVisible && curBankVis) {
                        r += 0.6f; // strong reward: revealed bank when we need bank
                    } else if (targetsAlreadyVisible) {
                        r -= 0.5f; // camera while targets already visible
                    } else if (wasStuck) {
                        r += 0.05f; // tiny nudge if stuck
                    } else {
                        r -= 0.6f; // discourage blind camera moves toward bank goal
                    }
                } else {
                    if (prevTreeVisible != null && !prevTreeVisible && curTreeVis) {
                        r += 0.6f; // strong reward: revealed tree when we need trees
                    } else if (targetsAlreadyVisible) {
                        r -= 0.5f;
                    } else if (wasStuck) {
                        r += 0.05f;
                    } else {
                        r -= 0.5f; // discourage blind camera moves while seeking trees
                    }
                }

                // Additional penalty for rapid camera oscillation
                if (recentActions.size() >= 3) {
                    int n = recentActions.size();
                    Integer a1 = recentActions.get(n - 1);
                    Integer a2 = recentActions.get(n - 2);
                    Integer a3 = recentActions.get(n - 3);
                    if (a1 != null && a2 != null && a3 != null) {
                        boolean altLR = (isCamLeft(a1) && isCamRight(a2)) || (isCamRight(a1) && isCamLeft(a2));
                        boolean altUD = (isCamUp(a1) && isCamDown(a2)) || (isCamDown(a1) && isCamUp(a2));
                        if (altLR || altUD || (isCameraIndex(a1) && isCameraIndex(a2) && isCameraIndex(a3))) {
                            r -= 0.4f; // discourage oscillation/spam more strongly
                        }
                    }
                }
            }
        }
        
        // === ENCOURAGE CAMERA ROTATION WHEN STUCK ===
        
        // If we're stuck trying to interact with objects, encourage camera rotation
        if (lastActionIndex != null) {
            boolean inventoryFull = taskContext.isInventoryNearFull();
            boolean bankVisible = isAnyBankVisible();
            boolean bankOpen = curBankOpen();
            
            // If trying to bank but can't (inventory full, bank visible, but bank not open)
            if (inventoryFull && bankVisible && !bankOpen && lastActionIndex == 0) {
                // Encourage camera rotation as a recovery action
                r -= 0.3f; // Additional penalty for failed banking to encourage trying other actions
            }
            
            // If trying to chop trees but not woodcutting
            if (!inventoryFull && lastActionIndex == 2 && !taskContext.isWoodcuttingAnim()) {
                // Encourage camera rotation to find better tree angles
                r -= 0.2f; // Additional penalty for failed tree chopping
            }
        }
        
        // === CLICK ALIGNMENT LEARNING ===
        
        // Track if clicks are misaligned with targets
        if (lastActionIndex != null) {
            if (lastActionIndex == 2) { // ChopNearestTreeTask
                // If we clicked but didn't start woodcutting, the click might be misaligned
                if (!taskContext.isWoodcuttingAnim() && taskContext.wasLastClickCanvasRecent(500)) {
                    // Recent click but no woodcutting - likely misaligned click
                    r -= 0.3f; // Penalty for misaligned clicks
                }
            }
        }
        
        // === OBSTRUCTION LEARNING ===
        
        // Track obstruction state for learning
        if (lastActionIndex != null) {
            if (lastActionIndex == 2) { // ChopNearestTreeTask
                // Check if the tree interaction failed due to obstruction
                if (!taskContext.isWoodcuttingAnim() && !taskContext.isPlayerMovingRecent(1000)) {
                    // Likely obstruction - penalize this behavior
                    r -= 0.5f; // Penalty for getting stuck on obstructed objects
                }
            } else if (lastActionIndex == 0) { // BankDepositTask
                // Check if banking failed due to obstruction
                if (taskContext.isInventoryNearFull() && !curBankOpen() && !taskContext.isPlayerMovingRecent(1000)) {
                    // Likely banking obstruction - penalize this behavior
                    r -= 0.5f; // Penalty for getting stuck on obstructed banks
                }
            }
        }
        
        // === STRATEGY DIVERSITY REWARDS ===
        
        // Reward trying different strategies when the current one fails
        if (lastActionIndex != null) {
            if (lastActionIndex == 2) { // ChopNearestTreeTask
                // If tree chopping failed, reward trying other actions
                if (!taskContext.isWoodcuttingAnim()) {
                    // Encourage trying camera rotation or movement as alternative strategies
                    if (recentActions.size() >= 2) {
                        Integer prevAction = recentActions.get(recentActions.size() - 2);
                        if (prevAction != null && prevAction == 2) {
                            // We tried chopping twice in a row - encourage different action
                            r -= 0.2f; // Additional penalty for repeating failed action
                        }
                    }
                }
            }
        }
        
        // === NAVIGATION SUCCESS REWARDS ===
        
        // Reward successful navigation around obstructions
        if (lastActionIndex != null) {
            if (lastActionIndex == 3) { // NavigateToTreeHotspotTask
                if (taskContext.isPlayerMovingRecent(1000)) {
                    r += 0.3f; // Reward for successful navigation
                }
            }
        }
        
        // === EXPLORATION LEARNING ===
        
        // Reward exploration when it leads to progress
        if (lastActionIndex != null) {
            int exploreIndex = 6; // ExploreTask index
            if (lastActionIndex == exploreIndex) {
                if (taskContext.isPlayerMovingRecent(1000)) {
                    r += 0.2f; // Reward for successful exploration movement
                } else {
                    r -= 0.2f; // Penalty for failed exploration
                }
            }
        }
        
        // === AGGRESSIVE EXPLORATION REWARDS ===
        
        // Major reward for discovering new higher-tier trees during exploration
        if (lastActionIndex != null) {
            int exploreIndex = 6; // ExploreTask index
            if (lastActionIndex == exploreIndex) {
                // Check if we discovered any new trees since last exploration
                int wc = 1; try { wc = client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
                java.util.List<WorldPoint> currentTrees = TreeDiscovery.getAvailableTrees();
                
                // Calculate best tier of currently available trees
                int bestCurrentTier = 0;
                for (WorldPoint tree : currentTrees) {
                    java.util.List<net.runelite.client.plugins.rlbot.config.RLBotConfigManager.TreeLocation> trees = net.runelite.client.plugins.rlbot.config.RLBotConfigManager.getTrees();
                    for (net.runelite.client.plugins.rlbot.config.RLBotConfigManager.TreeLocation t : trees) {
                        if (t.toWorldPoint().equals(tree)) {
                            int tier = LogQualityRewards.getLogQualityTier(t.name);
                            if (tier > bestCurrentTier) bestCurrentTier = tier;
                            break;
                        }
                    }
                }
                
                // Reward based on tier improvement
                if (bestCurrentTier >= 3) { // Willow or better
                    r += 2.0f; // Major reward for finding willow+ trees
                    logger.info("[RL] Exploration reward: Found tier " + bestCurrentTier + " trees (+2.0 reward)");
                } else if (bestCurrentTier >= 2) { // Oak
                    r += 0.5f; // Moderate reward for oak trees
                }
                
                // Additional reward if we're exploring when we could be chopping low-tier trees
                boolean hasLowTierTrees = bestCurrentTier > 0 && bestCurrentTier < 3; // Oak or regular trees
                int maxPossibleTier = wc >= 75 ? 8 : wc >= 60 ? 7 : wc >= 45 ? 5 : wc >= 30 ? 3 : wc >= 15 ? 2 : 1;
                if (hasLowTierTrees && maxPossibleTier > bestCurrentTier) {
                    r += 1.0f; // Bonus reward for exploring when higher-tier trees are possible
                    logger.info("[RL] Exploration bonus: Seeking tier " + maxPossibleTier + " trees when only tier " + bestCurrentTier + " available (+1.0 reward)");
                }
            }
        }
        
        // === ACTION-SPECIFIC REWARDS ===
        // Explicit shaping to learn priorities:
        // 1) NavigateToBank when inventory full and away from bank
        // 2) BankDeposit when already close to bank
        // 3) NavigateToTree when inventory not full
        // 4) ChopNearestTree when near a tree (higher than navigate), navigate as second when not full
        if (lastActionIndex != null) {
            boolean invFull = taskContext.isInventoryFull();
            boolean invNotFull = !invFull;
            int bankDist = distanceToNearestBank();
            int treeDist = distanceToNearestTree();
            boolean bankClose = bankDist >= 0 && bankDist <= 4;
            boolean treeClose = treeDist >= 0 && treeDist <= 4;

            // Resolve indices for clarity
            int BANK_DEPOSIT = 0;
            int NAV_BANK = 1;
            int CHOP = 2; // ChopNearestTreeTask
            int NAV_TREE = 3; // NavigateToTreeHotspotTask

            // 1) If inventory full and far from bank, prefer navigation to bank
            if (invFull && !bankClose && lastActionIndex == NAV_BANK) {
                r += 1.2f;
            }
            if (invFull && !bankClose && lastActionIndex == CHOP) {
                r -= 1.2f; // discourage chopping when full and away from bank
            }

            // 2) If close to bank and inventory near/full, prefer deposit over navigation
            if (bankClose && taskContext.isInventoryNearFull()) {
                if (lastActionIndex == BANK_DEPOSIT) r += 1.5f;
                if (lastActionIndex == NAV_BANK) r -= 0.8f; // demote continued nav when already close
            }

            // 3) If inventory not full, prefer navigating to trees over bank navigation
            if (invNotFull) {
                if (lastActionIndex == NAV_TREE) r += 0.8f;
                if (lastActionIndex == NAV_BANK) r -= 0.8f; // demote bank nav with space
            }

            // 4) If near trees and not full, prefer chopping over nav to trees
            if (invNotFull && treeClose) {
                if (lastActionIndex == CHOP) r += 1.2f;
                if (lastActionIndex == NAV_TREE) r -= 0.5f;
            }

            // Additional shaping: discourage NavigateToTree when already within 15 tiles of a tree
            if (invNotFull && treeDist >= 0 && treeDist <= 15 && lastActionIndex == NAV_TREE) {
                // Scale penalty slightly by proximity (closer -> larger penalty), capped
                float proximityPenalty = Math.min(1.0f, 0.3f + (15 - Math.max(0, treeDist)) * 0.05f);
                r -= proximityPenalty;
                if (steps % 20 == 0) {
                    logger.info("[RL] NAV TREE PROXIMITY PENALTY: distance=" + treeDist + " (-" + String.format("%.2f", proximityPenalty) + ")");
                }
            }
        }
        
        // Rewards for productive actions
        if (lastActionIndex != null) {
            if (lastActionIndex == 2) { // ChopNearestTreeTask
                if (taskContext.isWoodcuttingAnim()) {
                    r += 0.5f; // Additional reward for chopping action that led to woodcutting
                } else {
                    r -= 0.2f; // Small penalty for failed tree chopping action
                }
            } else if (lastActionIndex == 0) { // BankDepositTask
                if (taskContext.isInventoryFull()) {
                    r += 1.0f; // Strong reward for banking when inventory is completely full
                    if (steps % 20 == 0) {
                        logger.info("[RL] BANK OPENING REWARD: Opening bank with full inventory (+1.0 reward)");
                    }
                } else if (taskContext.isInventoryNearFull()) {
                    r += 0.5f; // Moderate reward for banking when inventory is nearly full
                    if (steps % 20 == 0) {
                        logger.info("[RL] BANK OPENING REWARD: Opening bank with nearly full inventory (+0.5 reward)");
                    }
                } else {
                    r -= 1.2f; // Strong penalty for trying to open bank when inventory has space
                    if (steps % 20 == 0) {
                        logger.info("[RL] BANK OPENING PENALTY: Trying to open bank with inventory space (-1.2 reward)");
                    }
                }
            } else if (lastActionIndex == 2) { // CrossWildernessDitchTask (unused)
                r -= 0.2f; // No general reward; discourage unless clearly needed
            }
        }
        
        // === FAILURE PENALTIES ===
        
        // Major penalty for repeated failed actions
        if (recentActions.size() >= 5) {
            // Check if we're stuck in a loop of the same action
            Integer lastAction = recentActions.getLast();
            int sameActionCount = 0;
            for (Integer action : recentActions) {
                if (action != null && action.equals(lastAction)) sameActionCount++;
            }
            if (sameActionCount >= 4) {
                r -= 2.0f; // Much larger penalty for being stuck in action loops
            }
        }
        
        // === ACTION-SPECIFIC FAILURE PENALTIES ===
        
        // Penalty for failed banking attempts
        if (lastActionIndex != null) {
            if (lastActionIndex == 0) { // BankDepositTask
                if (taskContext.isInventoryNearFull()) {
                    // Check if banking actually succeeded (inventory should be empty after successful banking)
                    if (prevFreeSlots != null && curFree == prevFreeSlots) {
                        // Banking failed - inventory still full
                        r -= 1.0f; // Large penalty for failed banking
                    } else {
                        r += 0.5f; // Reward for successful banking
                    }
                } else {
                    r -= 0.3f; // Penalty for unnecessary banking
                }
            // REMOVED DUPLICATE: ChopNearestTreeTask reward is already handled above
            // This section was causing conflicting rewards for the same action
            } else if (lastActionIndex == 2) { // CrossWildernessDitchTask
                r += 0.3f; // Reward for wilderness crossing attempts
            }
        }
        
        // === EXTERNAL PENALTIES ===
        
        // Apply external penalties (e.g., from chat messages about unreachable objects)
        r -= drainExternalPenalty();
        
        // === SMART TREE AVOIDANCE === (Help agent learn faster)
        // If we're near a tree we can't chop, add a small proximity penalty to discourage staying in that area
        try {
            int wcLevel = client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING);
            net.runelite.api.GameObject nearestTree = ObjectFinder.findNearestByNames(taskContext,
                new String[]{"tree", "oak", "willow", "teak", "maple", "mahogany", "yew", "magic", "redwood"}, "Chop down");
            if (nearestTree != null && client.getLocalPlayer() != null) {
                try {
                    net.runelite.api.ObjectComposition comp = client.getObjectDefinition(nearestTree.getId());
                    if (comp != null) {
                        String treeName = comp.getName().toLowerCase();
                        int requiredLevel = getRequiredWoodcuttingLevel(treeName);
                        if (requiredLevel > wcLevel) {
                            int distance = nearestTree.getWorldLocation().distanceTo(client.getLocalPlayer().getWorldLocation());
                            if (distance <= 5) {
                                float avoidancePenalty = Math.min(0.5f, (requiredLevel - wcLevel) * 0.01f);
                                r -= avoidancePenalty;
                                logger.info("[RL] AVOIDANCE PENALTY: Near impossible " + treeName + " at distance " + distance +
                                            " (-" + String.format("%.2f", avoidancePenalty) + ")");
                            }
                        }
                    }
                } catch (Exception ignored) {}
            }
        } catch (Exception ignored) {}
        
        // === INVENTORY MANAGEMENT PENALTIES ===
        
        // Strong penalty for choosing non-banking tasks when inventory is full
        if (lastActionIndex != null && taskContext.isInventoryFull()) {
            boolean isBankingTask = (lastActionIndex == 0) || (lastActionIndex == 1); // BankDepositTask or NavigateToBankHotspotTask
            if (!isBankingTask) {
                float inventoryPenalty = 1.5f; // Strong penalty to encourage banking
                r -= inventoryPenalty;
                String taskName = lastActionIndex < tasks.size() ? tasks.get(lastActionIndex).getClass().getSimpleName() : "Unknown";
                logger.info("[RL] INVENTORY FULL PENALTY: Chose " + taskName + " instead of banking when inventory full (-" + 
                           String.format("%.1f", inventoryPenalty) + " reward)");
            }
        }
        
        // Enhanced penalty for choosing banking tasks when inventory has plenty of space
        if (lastActionIndex != null && taskContext.getInventoryFreeSlots() >= 20) { // Changed from 28 to 20 for more sensitive detection
            boolean isBankingTask = (lastActionIndex == 0) || (lastActionIndex == 1); // BankDepositTask or NavigateToBankHotspotTask
            if (isBankingTask) {
                int freeSlots = taskContext.getInventoryFreeSlots();
                float emptyInventoryPenalty = 1.0f + (freeSlots - 20) * 0.1f; // Escalating penalty based on free space
                r -= emptyInventoryPenalty;
                String taskName = lastActionIndex < tasks.size() ? tasks.get(lastActionIndex).getClass().getSimpleName() : "Unknown";
                logger.info("[RL] INVENTORY SPACE PENALTY: Chose " + taskName + " with " + freeSlots + " free slots (-" + 
                           String.format("%.1f", emptyInventoryPenalty) + " reward)");
            }
        }
        
        // Additional penalty for trying to open bank when inventory is completely empty
        if (lastActionIndex != null && lastActionIndex == 0 && taskContext.getInventoryFreeSlots() >= 28) {
            float extremeEmptyPenalty = 2.0f; // Very strong penalty for completely unnecessary banking
            r -= extremeEmptyPenalty;
            logger.info("[RL] EXTREME EMPTY INVENTORY PENALTY: Trying to open bank with completely empty inventory (-" + 
                       String.format("%.1f", extremeEmptyPenalty) + " reward)");
        }
        
        // === TIME PENALTY ===
        
        // Small time penalty to encourage efficiency
        r -= 0.01f;
        
        return r;
    }

    // External penalty accumulator (thread-safe simple float via volatile + synchronized drain)
    private final java.util.concurrent.atomic.AtomicInteger externalPenaltyMilli = new java.util.concurrent.atomic.AtomicInteger(0);
    public void addExternalPenalty(float penalty) {
        int milli = Math.max(0, (int)Math.round(penalty * 1000f));
        externalPenaltyMilli.addAndGet(milli);
    }
    private float drainExternalPenalty() {
        int milli = externalPenaltyMilli.getAndSet(0);
        return milli / 1000f;
    }
    
    /**
     * Get the required woodcutting level for a specific tree type.
     * This helps the agent learn not to waste time on impossible actions.
     */
    private int getRequiredWoodcuttingLevel(String treeName) {
        if (treeName == null) return 1;
        String lower = treeName.toLowerCase();
        
        if (lower.contains("oak")) return 15;
        if (lower.contains("willow")) return 30;
        if (lower.contains("teak")) return 35;
        if (lower.contains("maple")) return 45;
        if (lower.contains("mahogany")) return 50;
        if (lower.contains("yew")) return 60;
        if (lower.contains("magic")) return 75;
        if (lower.contains("redwood")) return 90;
        
        return 1; // Regular trees require level 1
    }
    
    /**
     * Get information about the nearest CHOPPABLE tree for state representation.
     * Only considers trees the player can actually chop to avoid obsession with high-level trees.
     * Returns [tier, levelRequirement, playerLevel] to help agent learn tree-specific behavior.
     */
    private float[] getNearestTreeInfo() {
        try {
            // Get player's woodcutting level
            int playerWcLevel = 1;
            try {
                playerWcLevel = client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING);
            } catch (Exception ignored) {}
            
            // IMPORTANT: Only look for trees the player can actually chop!
            String[] allowedTreeTypes = TreeDiscovery.allowedTreeNamesForLevel(playerWcLevel);
            
            // Also look for any nearby higher-level trees to include in state (but mark as impossible)
            net.runelite.api.GameObject nearestChoppable = ObjectFinder.findNearestByNames(taskContext, allowedTreeTypes, "Chop down");
            net.runelite.api.GameObject nearestAny = ObjectFinder.findNearestByNames(taskContext, 
                new String[]{"tree", "oak", "willow", "teak", "maple", "mahogany", "yew", "magic", "redwood"}, "Chop down");
            
            // Prefer choppable tree, but if there's a closer impossible tree, include that info
            net.runelite.api.GameObject targetTree = nearestChoppable;
            if (nearestAny != null && nearestChoppable != null) {
                int distChoppable = nearestChoppable.getWorldLocation().distanceTo(client.getLocalPlayer().getWorldLocation());
                int distAny = nearestAny.getWorldLocation().distanceTo(client.getLocalPlayer().getWorldLocation());
                // If impossible tree is much closer, use it (so agent learns to avoid it)
                if (distAny < distChoppable - 3) {
                    targetTree = nearestAny;
                }
            } else if (nearestAny != null && nearestChoppable == null) {
                targetTree = nearestAny; // Only impossible trees nearby
            }
            
            if (targetTree != null) {
                try {
                    net.runelite.api.ObjectComposition comp = client.getObjectDefinition(targetTree.getId());
                    if (comp != null) {
                        String treeName = comp.getName().toLowerCase();
                        int tier = LogQualityRewards.getLogQualityTier(treeName);
                        int levelReq = getRequiredWoodcuttingLevel(treeName);
                        
                        // Normalize values for neural network (0-1 range)
                        float tierNorm = Math.max(0f, Math.min(1f, tier / 9.0f)); // Tiers 1-9
                        float levelReqNorm = Math.max(0f, Math.min(1f, levelReq / 90.0f)); // Levels 1-90
                        float playerLevelNorm = Math.max(0f, Math.min(1f, playerWcLevel / 90.0f)); // Levels 1-90
                        
                        // Log when we're near impossible trees
                        if (levelReq > playerWcLevel) {
                            logger.info("[RL] STATE: Near impossible " + treeName + " (req " + levelReq + ", have " + playerWcLevel + ") - agent should learn to avoid!");
                        }
                        
                        return new float[]{tierNorm, levelReqNorm, playerLevelNorm};
                    }
                } catch (Exception ignored) {}
            }
            
            // No tree found - return neutral values
            float playerLevelNorm = Math.max(0f, Math.min(1f, playerWcLevel / 90.0f));
            return new float[]{0f, 0f, playerLevelNorm}; // No tree, but include player level
            
        } catch (Exception ignored) {
            return new float[]{0f, 0f, 0f}; // Fallback to zeros
        }
    }

    private boolean curBankOpen() {
        return taskContext.client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER) != null;
    }

    // Removed internal training/logging hooks

    private String buildActionDistributionString() {
        if (actionCounts == null || actionCounts.length != tasks.size()) return "n/a";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tasks.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(tasks.get(i).getClass().getSimpleName()).append('=');
            sb.append(actionCounts[i]);
        }
        return sb.toString();
    }

    // Training loop (experience replay) will be added to update DJL network weights.
    
    // Getter methods for overlay access
    public long getSteps() { return steps; }
    public double getEpisodeReturn() { return episodeReturn; }
    // Internal RL metrics removed
    public long getEpisodeStartMs() { return episodeStartMs; }
    public int[] getActionCounts() { return actionCounts; }
    public java.util.LinkedList<Integer> getRecentActions() { return recentActions; }
    public int getTotalActions() {
        if (actionCounts == null) return 0;
        int total = 0;
        for (int count : actionCounts) {
            total += count;
        }
        return total;
    }
    public double getStepsPerSecond() {
        long episodeDurationMs = System.currentTimeMillis() - episodeStartMs;
        return episodeDurationMs > 0 ? (steps * 1000.0) / episodeDurationMs : 0.0;
    }
    public double getEfficiency() {
        return steps > 0 ? episodeReturn / steps : 0.0;
    }
    public float getActionDiversity() {
        if (recentActions == null || recentActions.size() < 3) return 0f;
        java.util.Set<Integer> uniqueActions = new java.util.HashSet<>(recentActions);
        return (float) uniqueActions.size() / recentActions.size();
    }

    // === Policy visualization getters === (internal RL removed)
    public int getLastChosenAction() { return lastChosenAction; }
    public int getNumActions() { return tasks != null ? tasks.size() : 0; }
    public String getActionName(int idx) {
        try { return tasks.get(idx).getClass().getSimpleName(); } catch (Exception e) { return "Action" + idx; }
    }

    // Real-time Q for overlay: compute from current state each render
    // public float[] predictQNowSafe() { return null; }

    // Helpers for camera action oscillation detection
    private boolean isCameraIndex(int idx) { return idx >= 7 && idx <= 12; }
    private boolean isCamLeft(int idx) { return getSafeName(idx).contains("CameraLeft"); }
    private boolean isCamRight(int idx) { return getSafeName(idx).contains("CameraRight"); }
    private boolean isCamUp(int idx) { return getSafeName(idx).contains("CameraUp"); }
    private boolean isCamDown(int idx) { return getSafeName(idx).contains("CameraDown"); }
    private String getSafeName(int idx) {
        try { return tasks.get(idx).getClass().getSimpleName(); } catch (Exception e) { return ""; }
    }
}
