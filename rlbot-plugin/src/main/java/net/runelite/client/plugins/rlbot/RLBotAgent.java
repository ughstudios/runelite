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
import net.runelite.client.plugins.rlbot.rewards.LogQualityRewards;
import java.util.Random;
import net.runelite.client.plugins.rlbot.rl.DJLDqnPolicy;
import net.runelite.client.plugins.rlbot.tasks.ObjectFinder;
import net.runelite.api.GameObject;
import net.runelite.client.plugins.rlbot.rl.ReplayBuffer;
import net.runelite.client.plugins.rlbot.rl.Transition;
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
    private DJLDqnPolicy dqn;
    private ReplayBuffer replay;
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
    private int lastEpsilonPct = 0;
    private Float lastTrainLoss = null;
    private long lastStatusLogStep = 0L;
    private java.util.LinkedList<Integer> recentActions = new java.util.LinkedList<>();

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
            new BankDepositTask(),
            new NavigateToBankHotspotTask(),
            new CrossWildernessDitchTask(), // Higher priority - cross wilderness before chopping
            new CrossWildernessDitchOutTask(), // Cross back out of wilderness
            new ChopNearestTreeTask(),
            new NavigateToTreeHotspotTask(),
            new ExploreTask(),
            new CameraRotateTask(),
            new CameraAdjustmentTask(), // Enhanced camera controls for stuck/idle situations
            new IdleTask()
        );
        if (config.enableRLAgent()) {
            // Defer DJL model init to onTick (client thread) to avoid constructor-time client calls here
            this.dqn = null;
            this.replay = new ReplayBuffer(config.rlReplayCapacity());
            this.episodeStartMs = System.currentTimeMillis();
        }
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

        if (!config.enableRLAgent()) {
            return;
        }

        // Lazy-initialize DQN on client thread after login to avoid constructor-time client calls here
        if (dqn == null) {
            try {
                float[] s = buildDjlStateVector();
                dqn = new DJLDqnPolicy(s.length, tasks.size());
                logger.info("[RL] Initialized DQN with stateDim=" + s.length + " actions=" + tasks.size());
                actionCounts = new int[tasks.size()];
                // Try load model from disk on startup
                try {
                    java.nio.file.Path modelDir = java.nio.file.Paths.get(System.getProperty("user.home"), ".rlbot", "models");
                    boolean loaded = dqn.loadFrom(modelDir);
                    logger.info("[RL] Model load " + (loaded ? "succeeded" : "not found/failed") + " from " + modelDir);
                } catch (Exception ignored) {}
            } catch (Exception e) {
                logger.error("[RL] DQN init failed: " + e.getMessage());
                return;
            }
        }

        long now = System.currentTimeMillis();
        int interval = Math.max(25, config.agentIntervalMs()); // Reduced from 50ms to 25ms for more responsiveness
        if (now - lastRunMillis < interval) {
            return;
        }
        lastRunMillis = now;

        telemetry.setBusyRemainingMs(0);

        // Choose and run task via RL policy exclusively
        float[] state = buildDjlStateVector();
        int actionIndex = selectActionIndex();
        if (actionIndex >= 0) {
            Task t = tasks.get(actionIndex);
            logger.info("[Agent] RL picked task=" + t.getClass().getSimpleName());
            if (actionCounts != null && actionIndex >= 0 && actionIndex < actionCounts.length) {
                actionCounts[actionIndex]++;
            }
            runTaskSafe(t);
            float reward = computeImmediateReward();
            logger.info("[RL] stepReward=" + String.format("%.3f", reward));
            float[] nextState = buildDjlStateVector();
            boolean done = false;
            // Synthetic episode boundary: end when inventory empties after banking, or after a fixed horizon
            if ((prevBankOpen != null && prevBankOpen && !curBankOpen()) || steps % 300 == 0) {
                done = true;
            }
            replay.add(new Transition(state, actionIndex, reward, nextState, done));
            steps++;
            episodeReturn += reward;
            maybeTrain();
            prevFreeSlots = taskContext.getInventoryFreeSlots();
            prevBankOpen = client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER) != null;
            prevTreeDist = distanceToNearestTree();
            prevBankDist = distanceToNearestBank();
            lastActionIndex = actionIndex;
            
            // Track recent actions for penalty calculation
            recentActions.offer(actionIndex);
            if (recentActions.size() > 10) {
                recentActions.poll(); // Keep only last 10 actions
            }
            
            try { prevWoodcutXp = client.getSkillExperience(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
            if (done || reward >= 9.9f || System.currentTimeMillis() - episodeStartMs > 5 * 60_000L) {
                logger.info("[RL] Episode end: return=" + String.format("%.2f", episodeReturn) +
                    " durationMs=" + (System.currentTimeMillis() - episodeStartMs));
                episodeReturn = 0.0;
                episodeStartMs = System.currentTimeMillis();
            }
            dqn.saveIfNeeded(steps);
            maybeLogStatus();
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

    private void runTaskSafe(Task t) {
        try {
            if (!t.shouldRun(taskContext)) {
                logger.info("[Agent] RL selected task but not eligible: " + t.getClass().getSimpleName());
                return;
            }
            if (t instanceof NavigateToBankHotspotTask) telemetry.setMode("Navigate: Bank");
            else if (t instanceof NavigateToTreeHotspotTask) telemetry.setMode("Navigate: Trees");
            else if (t instanceof BankDepositTask) telemetry.setMode("Banking");
            else if (t instanceof ChopNearestTreeTask) telemetry.setMode("Chop");
            else telemetry.setMode("Act");
            // Guard: do not initiate new clicks while player is moving
            if (taskContext.isPlayerWalking() && (t instanceof ChopNearestTreeTask || t instanceof BankDepositTask)) {
                logger.info("[Agent] Skipping click task while moving");
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

    private int selectActionIndex() {
        boolean[] mask = new boolean[tasks.size()];
        int eligibleCount = 0;
        int freeSlots = taskContext.getInventoryFreeSlots();
        // Inventory is considered full only when there are 0 free slots
        boolean inventoryFull = taskContext.isInventoryFull();
        boolean treeVisible = isAnyTreeVisible();
        boolean bankVisible = isAnyBankVisible();
        boolean ditchVisible = isAnyDitchVisible();
        int stuckCount = taskContext.getNavNoProgressCount();
        
        // Debug logging to understand the inventory state
        logger.info("[RL] Inventory state: freeSlots=" + freeSlots + ", inventoryFull=" + inventoryFull);
        
        for (int i = 0; i < tasks.size(); i++) {
            try {
                Task ti = tasks.get(i);
                boolean allowed;
                // Gate banking: only if inventory full OR bank is already open
                if (ti instanceof BankDepositTask || ti instanceof NavigateToBankHotspotTask) {
                    boolean bankOpen = client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER) != null;
                    allowed = inventoryFull || bankOpen;
                    logger.info("[RL] Banking task " + ti.getClass().getSimpleName() + " allowed=" + allowed + " (freeSlots=" + freeSlots + ", inventoryFull=" + inventoryFull + ", bankOpen=" + bankOpen + ")");
                } else if (ti instanceof ChopNearestTreeTask || ti instanceof NavigateToTreeHotspotTask) {
                    // Gate trees: only if inventory not full
                    allowed = !inventoryFull;
                } else {
                    allowed = true;
                }
                // Mask chop if no tree visible
                if (ti instanceof ChopNearestTreeTask && !treeVisible) allowed = false;
                // Prefer BankDepositTask over NavigateToBank when a bank is visible
                if (ti instanceof NavigateToBankHotspotTask && bankVisible) {
                    // leave allowed true, but we'll select BankDeposit in the deterministic preference block
                }
                // Ditch crossing allowed when ditch visible and inventory not full
                if (ti instanceof CrossWildernessDitchTask) {
                    allowed = !inventoryFull && ditchVisible;
                }
                // Idle gating: allow idling when currently busy or when woodcutting anim plays
                if (ti instanceof IdleTask) {
                    boolean cutting = taskContext.isWoodcuttingAnim();
                    boolean busy = taskContext.isBusy() && !taskContext.timedOutSince(200);
                    allowed = cutting || busy;
                }
                // Prevent other tasks from running when woodcutting (except IdleTask)
                if (!(ti instanceof IdleTask) && taskContext.isWoodcuttingAnim()) {
                    allowed = false;
                }
                // Never rotate or explore when inventory is full; must bank
                if (inventoryFull && (ti instanceof CameraRotateTask || ti instanceof ExploreTask)) {
                    allowed = false;
                }
                // Also, if inventoryFull and bank is visible or bank is open, prefer banking tasks over rotation
                if (inventoryFull && (ti instanceof CameraRotateTask) && (bankVisible || curBankOpen())) {
                    allowed = false;
                }
                mask[i] = allowed && ti.shouldRun(taskContext);
                if (mask[i]) eligibleCount++;
            } catch (Exception e) {
                mask[i] = false;
            }
        }
        // Deterministic preference: when a target is visible and close, prefer interacting task over navigation
        try {
            int chopIdx = -1, navTreeIdx = -1, bankIdx = -1, navBankIdx = -1;
            for (int i = 0; i < tasks.size(); i++) {
                if (tasks.get(i) instanceof ChopNearestTreeTask) chopIdx = i;
                if (tasks.get(i) instanceof NavigateToTreeHotspotTask) navTreeIdx = i;
                if (tasks.get(i) instanceof BankDepositTask) bankIdx = i;
                if (tasks.get(i) instanceof NavigateToBankHotspotTask) navBankIdx = i;
            }
            int tDist = distanceToNearestTree();
            if (!inventoryFull && treeVisible && tDist >= 0 && tDist <= 6 && chopIdx >= 0 && mask[chopIdx]) {
                return chopIdx; // Prefer chopping when close and visible
            }
            int bDist = distanceToNearestBank();
            if (inventoryFull && ((bankVisible && bDist >= 0 && bDist <= 6) || (bDist >= 0 && bDist <= 6)) && bankIdx >= 0 && mask[bankIdx]) {
                return bankIdx; // Prefer banking when close and visible
            }
        } catch (Exception ignored) {}
        if (eligibleCount == 0) {
            // Recovery: if nothing eligible, try exploration or camera rotate
            for (int i = 0; i < tasks.size(); i++) {
                Task ti = tasks.get(i);
                // If inventory is full, don't recover with rotate/explore; force NavigateToBank
                if (inventoryFull) {
                    if (ti instanceof NavigateToBankHotspotTask || ti instanceof BankDepositTask) {
                        return i;
                    }
                    continue;
                }
                if (ti instanceof ExploreTask || ti instanceof CameraRotateTask) {
                    return i;
                }
            }
            return -1;
        }
        float[] q = new float[tasks.size()];
        try {
            float[] state = buildDjlStateVector();
            if (dqn != null) {
                q = dqn.predictQ(state);
            }
        } catch (Exception ignored) {}
        int baseEps = Math.max(0, Math.min(100, config.rlEpsilon()));
        lastEpsilonPct = baseEps;
        // Deterministic: no random exploration when rlEpsilon default is 0
        int best = -1;
        float bestQ = -1e9f;
        for (int i = 0; i < q.length; i++) {
            if (!mask[i]) continue;
            if (q[i] > bestQ) { bestQ = q[i]; best = i; }
        }
        return best >= 0 ? best : 0;
    }

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
        if (lastActionIndex != null && lastActionIndex == 3) { // ChopNearestTreeTask
            // If we just tried to chop but aren't woodcutting and not moving, likely obstruction
            if (!taskContext.isWoodcuttingAnim() && !taskContext.isPlayerMovingRecent(1000)) {
                obstructionAttempts = 1f; // Signal obstruction state
            }
        }
        
        // === CLICK ALIGNMENT STATE ===
        float clickAlignmentIssue = 0f;
        if (lastActionIndex != null && lastActionIndex == 3) { // ChopNearestTreeTask
            // If we clicked recently but aren't woodcutting, likely click alignment issue
            if (!taskContext.isWoodcuttingAnim() && taskContext.wasLastClickCanvasRecent(500)) {
                clickAlignmentIssue = 1f; // Signal click alignment problem
            }
        }
        
        return new float[] {
            freeBucket, bankOpen ? 1f : 0f, movingRecent, runEnergy,
            yawSin, yawCos, pitch01,
            tBucket, bBucket, dBucket,
            treeBearing[0], treeBearing[1], bankBearing[0], bankBearing[1], ditchBearing[0], ditchBearing[1],
            treeVis, bankVis, ditchVis,
            woodcuttingAnim, inventoryNearFull, isStuck, navNoProgress,
            recentActionDiversity, currentGoal, lastClickSuccess, inWilderness, obstructionAttempts, clickAlignmentIssue
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
        }
        
        // Major reward for successful banking (completing the cycle)
        if (prevBankOpen != null && prevBankOpen && curBankOpen && prevFreeSlots != null && curFree > prevFreeSlots) {
            int deposited = curFree - prevFreeSlots;
            if (deposited > 0) r += Math.min(10.0f, deposited * 1.5f); // Very large reward for successful banking
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
        
        // === WILDERNESS DITCH REWARDS ===
        
        // Strong rewards for wilderness ditch interaction
        int dd = distanceToNearestDitch();
        if (dd >= 0 && dd <= 3) {
            r += 0.3f; // Good reward for being near ditch
            if (dd <= 1) {
                r += 0.5f; // Extra reward for being very close to ditch
            }
        }
        
        // === PENALTIES FOR GETTING STUCK ===
        
        // Major penalty for being stuck (not moving for extended periods)
        if (!taskContext.isPlayerMovingRecent(5000)) { // 5 seconds
            r -= 1.0f; // Significant penalty for being stuck
        }
        
        // === CAMERA ROTATION LEARNING ===
        
        // Reward camera rotation when it helps find targets
        if (lastActionIndex != null) {
            int cameraRotateIndex = 6; // Camera rotate task index
            if (lastActionIndex == cameraRotateIndex) {
                // Check if camera rotation revealed something useful
                boolean treeVisible = isAnyTreeVisible();
                boolean bankVisible = isAnyBankVisible();
                boolean ditchVisible = isAnyDitchVisible();
                
                if (treeVisible || bankVisible || ditchVisible) {
                    r += 0.3f; // Reward for camera rotation that reveals targets
                } else {
                    // Small penalty for camera rotation that doesn't help
                    r -= 0.1f;
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
            if (!inventoryFull && lastActionIndex == 3 && !taskContext.isWoodcuttingAnim()) {
                // Encourage camera rotation to find better tree angles
                r -= 0.2f; // Additional penalty for failed tree chopping
            }
        }
        
        // === CLICK ALIGNMENT LEARNING ===
        
        // Track if clicks are misaligned with targets
        if (lastActionIndex != null) {
            if (lastActionIndex == 3) { // ChopNearestTreeTask
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
            if (lastActionIndex == 3) { // ChopNearestTreeTask
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
            if (lastActionIndex == 3) { // ChopNearestTreeTask
                // If tree chopping failed, reward trying other actions
                if (!taskContext.isWoodcuttingAnim()) {
                    // Encourage trying camera rotation or movement as alternative strategies
                    if (recentActions.size() >= 2) {
                        Integer prevAction = recentActions.get(recentActions.size() - 2);
                        if (prevAction != null && prevAction == 3) {
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
            if (lastActionIndex == 4) { // NavigateToTreeHotspotTask
                if (taskContext.isPlayerMovingRecent(1000)) {
                    r += 0.3f; // Reward for successful navigation
                }
            }
        }
        
        // === EXPLORATION LEARNING ===
        
        // Reward exploration when it leads to progress
        if (lastActionIndex != null) {
            int exploreIndex = 7; // ExploreTask index
            if (lastActionIndex == exploreIndex) {
                if (taskContext.isPlayerMovingRecent(1000)) {
                    r += 0.2f; // Reward for successful exploration movement
                } else {
                    r -= 0.2f; // Penalty for failed exploration
                }
            }
        }
        
        // === ACTION-SPECIFIC REWARDS ===
        
        // Rewards for productive actions
        if (lastActionIndex != null) {
            if (lastActionIndex == 3) { // ChopNearestTreeTask
                if (taskContext.isWoodcuttingAnim()) {
                    r += 1.0f; // Reward for successful tree chopping
                } else {
                    r -= 0.2f; // Small penalty for failed tree chopping
                }
            } else if (lastActionIndex == 0) { // BankDepositTask
                if (taskContext.isInventoryNearFull()) {
                    r += 0.5f; // Reward for banking when needed
                } else {
                    r -= 0.3f; // Penalty for unnecessary banking
                }
            } else if (lastActionIndex == 2) { // CrossWildernessDitchTask
                r += 0.3f; // Reward for wilderness crossing attempts
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
            } else if (lastActionIndex == 3) { // ChopNearestTreeTask
                if (taskContext.isWoodcuttingAnim()) {
                    r += 1.0f; // Reward for successful tree chopping
                } else {
                    r -= 0.5f; // Larger penalty for failed tree chopping
                }
            } else if (lastActionIndex == 2) { // CrossWildernessDitchTask
                r += 0.3f; // Reward for wilderness crossing attempts
            }
        }
        
        // === EXTERNAL PENALTIES ===
        
        // Apply external penalties (e.g., from chat messages about unreachable objects)
        r -= drainExternalPenalty();
        
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

    private boolean curBankOpen() {
        return taskContext.client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER) != null;
    }

    private void maybeTrain() {
        if (replay == null || replay.size() < config.rlBatchSize()) return;
        int batch = config.rlBatchSize();
        java.util.List<Transition> ts = replay.sample(batch);
        float[][] s = new float[ts.size()][];
        float[][] ns = new float[ts.size()][];
        int[] a = new int[ts.size()];
        float[] r = new float[ts.size()];
        boolean[] d = new boolean[ts.size()];
        for (int i = 0; i < ts.size(); i++) {
            Transition t = ts.get(i);
            s[i] = t.state;
            ns[i] = t.nextState;
            a[i] = t.action;
            r[i] = t.reward;
            d[i] = t.done;
        }
        try {
            float loss = dqn.updateBatch(s, a, r, ns, d, Math.max(0f, Math.min(1f, config.rlGamma() / 100f)));
            lastTrainLoss = loss;
        } catch (Exception ignored) {}
        if (steps % Math.max(1, config.rlTargetSyncSteps()) == 0) {
            try { dqn.syncTarget(); } catch (Exception ignored) {}
        }
    }

    private void maybeLogStatus() {
        if (steps - lastStatusLogStep < 10) return;
        lastStatusLogStep = steps;
        String dist = buildActionDistributionString();
        String lossStr = lastTrainLoss == null ? "n/a" : String.format("%.5f", lastTrainLoss);
        int rep = replay == null ? 0 : replay.size();
        logger.info("[RL] steps=" + steps + " replay=" + rep + " eps=" + lastEpsilonPct + "% loss=" + lossStr + " actions=" + dist);
        if (replay != null && replay.size() < config.rlBatchSize()) {
            logger.info("[RL] training skipped (replay " + replay.size() + " < batch " + config.rlBatchSize() + ")");
        }
    }

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
    public int getLastEpsilonPct() { return lastEpsilonPct; }
    public Float getLastTrainLoss() { return lastTrainLoss; }
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
}


