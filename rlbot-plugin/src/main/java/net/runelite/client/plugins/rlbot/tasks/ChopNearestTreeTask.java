package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.GameObject;

/**
 * Orchestrates tree chopping by delegating to specialized classes.
 */
public class ChopNearestTreeTask implements Task {
    
    private static long lastInvokeMs = 0L;
    private static int recentInvokeFailures = 0;
    
    @Override
    public boolean shouldRun(TaskContext context) {
        context.logger.info("[ChopTask] shouldRun() - ENTRY");
        
        if (context.isInventoryFull()) {
            context.logger.info("[ChopTask] shouldRun() = false (inventory full)");
            return false;
        }
        
        if (context.isWoodcuttingAnim()) {
            context.logger.info("[ChopTask] shouldRun() = false (already woodcutting)");
            return false;
        }
        
        TreeDiscovery.scanAndDiscoverTrees(context);
        
        int wcLevel = getWoodcuttingLevel(context);
        String[] allowedTrees = TreeDiscovery.allowedTreeNamesForLevel(wcLevel);
        context.logger.info("[ChopTask] WC level: " + wcLevel + ", allowed trees: " + java.util.Arrays.toString(allowedTrees));
        
        GameObject candidate = findValidTree(context, allowedTrees);
        
        if (candidate == null) {
            boolean hasDiscovered = !TreeDiscovery.getDiscoveredTrees().isEmpty();
            if (!hasDiscovered) {
                TreeDiscovery.scanAndDiscoverTrees(context);
                hasDiscovered = !TreeDiscovery.getDiscoveredTrees().isEmpty();
            }
            context.logger.info("[ChopTask] shouldRun() = " + hasDiscovered + " (discovered trees: " + TreeDiscovery.getDiscoveredTrees().size() + ")");
            return hasDiscovered;
        }
        
        if (!TreeValidator.isValidTree(context, candidate, wcLevel)) {
            context.logger.info("[ChopTask] shouldRun() = false (invalid tree)");
            TreeDiscovery.markDepleted(candidate.getWorldLocation());
            return false;
        }
        
        if (TreeValidator.isWildernessConflict(context, candidate)) {
            context.logger.info("[ChopTask] shouldRun() = false (wilderness conflict)");
            return false;
        }
        
        boolean canSeeTree = ObjectFinder.projectToCanvas(context, candidate) != null;
        context.logger.info("[ChopTask] shouldRun() = " + canSeeTree + " (can see tree)");
        
        return canSeeTree;
    }
    
    @Override
    public void run(TaskContext context) {
        context.logger.info("[ChopTask] run() - ENTRY");
        
        lastInvokeMs = System.currentTimeMillis();
        UiHelper.closeObstructions(context);
        
        if (context.isBusy() && !context.timedOutSince(4000)) {
            context.logger.info("[ChopTask] run() - EXIT (busy)");
            return;
        }
        
        if (context.client.getLocalPlayer() == null) {
            context.logger.info("[ChopTask] run() - EXIT (no player)");
            return;
        }
        
        int wcLevel = getWoodcuttingLevel(context);
        String[] allowedTrees = TreeDiscovery.allowedTreeNamesForLevel(wcLevel);
        
        GameObject tree = findValidTree(context, allowedTrees);
        
        if (tree == null || TreeDiscovery.isDepleted(tree.getWorldLocation())) {
            context.logger.info("[ChopTask] run() - PATH: Navigate to trees");
            TreeNavigator.navigateToTrees(context);
            return;
        }
        
        if (!TreeValidator.hasAxe(context)) {
            context.logger.warn("[ChopTask] run() - PATH: No axe found");
            context.setBusyForMs(100);
            return;
        }
        
        context.logger.info("[ChopTask] run() - PATH: Attempt to click tree");
        boolean clicked = TreeClicker.clickTree(context, tree);
        
        if (clicked) {
            context.logger.info("[ChopTask] run() - SUCCESS: Tree clicked");
            return;
        }
        
        if (TreeClicker.adjustCameraForTree(context, tree)) {
            context.logger.info("[ChopTask] run() - PATH: Camera adjustment");
            return;
        }
        
        context.logger.warn("[ChopTask] run() - PATH: Mark tree as depleted");
        TreeDiscovery.markDepleted(tree.getWorldLocation());
        
        if (!context.isWoodcuttingAnim()) {
            handleRecovery(context);
        }
    }
    
    private int getWoodcuttingLevel(TaskContext context) {
        try {
            return context.client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING);
        } catch (Exception e) {
            return 1;
        }
    }
    
    private GameObject findValidTree(TaskContext context, String[] allowedTrees) {
        String[] actionNames = {"Chop down", "Chop", "Cut", "Cut down"};
        
        for (String actionName : actionNames) {
            GameObject candidate = ObjectFinder.findNearestByNames(context, allowedTrees, actionName);
            if (candidate != null && !TreeDiscovery.isDepleted(candidate.getWorldLocation())) {
                context.logger.info("[ChopTask] Found candidate tree with action '" + actionName + "' at " + candidate.getWorldLocation());
                return candidate;
            }
        }
        
        return null;
    }
    
    private void handleRecovery(TaskContext context) {
        long now = System.currentTimeMillis();
        if (now - lastInvokeMs < 4500L) {
            recentInvokeFailures++;
        } else {
            recentInvokeFailures = 0;
        }
        
        if (recentInvokeFailures >= 3) {
            context.logger.info("[ChopTask] run() - PATH: Recovery movement");
            TreeNavigator.performRecoveryMovement(context);
            recentInvokeFailures = 0;
        }
    }
}