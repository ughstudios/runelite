package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.GameObject;

/**
 * Orchestrates tree chopping by delegating to specialized classes.
 */
public class ChopNearestTreeTask implements Task {

    @Override
    public void run(TaskContext context) {
        context.logger.info("[ChopTask] run() - ENTRY");
        
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
        TreeClicker.Result clickResult = TreeClicker.clickTree(context, tree);
        
        if (clickResult == TreeClicker.Result.CLICKED) {
            context.logger.info("[ChopTask] run() - SUCCESS: Tree clicked");
            return;
        }

        if (TreeClicker.adjustCameraForTree(context, tree)) {
            context.logger.info("[ChopTask] run() - PATH: Camera adjustment");
            return;
        }

        context.logger.warn("[ChopTask] run() - PATH: Click validation failed");
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
}
