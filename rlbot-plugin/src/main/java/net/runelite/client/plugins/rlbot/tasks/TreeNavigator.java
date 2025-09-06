package net.runelite.client.plugins.rlbot.tasks;

import java.util.List;
import net.runelite.api.GameObject;
import net.runelite.api.coords.WorldPoint;

/**
 * Handles navigation and movement for tree-related tasks.
 */
public class TreeNavigator {
    
    /**
     * Navigates to an available tree or explores to find new ones.
     */
    public static void navigateToTrees(TaskContext context) {
        context.logger.info("[TreeNavigator] Starting navigation to find trees");
        
        WorldPoint playerPos = context.client.getLocalPlayer() != null ? 
            context.client.getLocalPlayer().getWorldLocation() : null;
            
        if (playerPos == null) {
            context.logger.warn("[TreeNavigator] Player position is null");
            return;
        }
        
        List<WorldPoint> availableTrees = TreeDiscovery.getAvailableTrees();
        if (!availableTrees.isEmpty()) {
            WorldPoint targetTree = availableTrees.get((int)(Math.random() * availableTrees.size()));
            context.logger.info("[TreeNavigator] Navigating to available tree at " + targetTree);
            WorldPathing.clickStepToward(context, targetTree, 6);
            context.setBusyForMs(500);
        } else {
            exploreForTrees(context, playerPos);
        }
    }
    
    /**
     * Explores in a random direction to discover new trees.
     */
    private static void exploreForTrees(TaskContext context, WorldPoint playerPos) {
        context.logger.info("[TreeNavigator] No available trees found, exploring to discover new ones");
        
        int dx = (int)(Math.random() * 20) - 10;
        int dy = (int)(Math.random() * 20) - 10;
        WorldPoint explorePoint = new WorldPoint(
            playerPos.getX() + dx, playerPos.getY() + dy, playerPos.getPlane());
            
        context.logger.info("[TreeNavigator] Exploring to " + explorePoint + " to find new trees");
        WorldPathing.clickStepToward(context, explorePoint, 6);
        context.setBusyForMs(500);
    }
    
    /**
     * Navigates toward the nearest discovered tree.
     */
    public static void navigateToNearestTree(TaskContext context) {
        WorldPoint playerPos = context.client.getLocalPlayer() != null ? 
            context.client.getLocalPlayer().getWorldLocation() : null;
            
        if (playerPos == null) {
            context.logger.warn("[TreeNavigator] Player position is null");
            return;
        }
        
        WorldPoint nearest = TreeDiscovery.getNearestDiscoveredTree(playerPos);
        if (nearest != null) {
            context.logger.info("[TreeNavigator] Navigating to nearest discovered tree at " + nearest);
            WorldPathing.clickStepToward(context, nearest, 6);
            context.setBusyForMs(500);
        } else {
            context.logger.info("[TreeNavigator] No discovered trees to navigate to");
        }
    }
    
    /**
     * Performs recovery movement when chopping attempts fail repeatedly.
     */
    public static void performRecoveryMovement(TaskContext context) {
        context.logger.info("[TreeNavigator] Performing recovery movement");
        
        CameraHelper.sweepYawSmall(context, 12);
        
        WorldPoint playerPos = context.client.getLocalPlayer() != null ? 
            context.client.getLocalPlayer().getWorldLocation() : null;
            
        if (playerPos != null) {
            WorldPoint recoveryPoint = new WorldPoint(playerPos.getX() + 1, playerPos.getY(), playerPos.getPlane());
            context.logger.info("[TreeNavigator] Moving to recovery point: " + recoveryPoint);
            WorldPathing.clickStepToward(context, recoveryPoint, 4);
            context.setBusyForMs(200);
        }
    }
}
