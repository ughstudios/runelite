package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.GameObject;

/**
 * Handles clicking on trees for woodcutting.
 */
public class TreeClicker {
    
    private static final String[] CHOP_ACTIONS = {"Chop down", "Chop", "Cut", "Cut down"};
    
    /**
     * Clicks a tree using the same simple approach as manual clicks (client thread + moveAndClickWithValidation).
     */
    public static boolean clickTree(TaskContext context, GameObject tree) {
        if (tree == null) {
            context.logger.info("[TreeClicker] Tree is null, cannot click");
            return false;
        }
        
        context.logger.info("[TreeClicker] Attempting to click tree at " + tree.getWorldLocation());
        
        try {
            // Get tree composition and find chop action (same as manual fallback)
            net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(tree.getId());
            if (comp == null) {
                context.logger.warn("[TreeClicker] No object composition for tree");
                return false;
            }
            
            String[] actions = comp.getActions();
            if (actions == null) {
                context.logger.warn("[TreeClicker] Tree has no actions");
                return false;
            }
            
            int chopIdx = -1;
            String chopLabel = null;
            for (int i = 0; i < actions.length; i++) {
                String action = actions[i];
                if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                    chopIdx = i;
                    chopLabel = action;
                    break;
                }
            }
            
            if (chopIdx < 0) {
                context.logger.warn("[TreeClicker] No chop action found");
                return false;
            }
            
            context.logger.info("[TreeClicker] Found chop action: " + chopLabel + " at index " + chopIdx);
            
            // Project to canvas point
            java.awt.Point projPoint = ObjectFinder.projectToClickablePoint(context, tree);
            if (projPoint == null || projPoint.x < 0 || projPoint.y < 0 || projPoint.x >= 765 || projPoint.y >= 503) {
                projPoint = ObjectFinder.projectToCanvas(context, tree);
            }
            
            if (projPoint == null) {
                context.logger.warn("[TreeClicker] Cannot project tree to canvas");
                return false;
            }
            
            context.logger.info("[TreeClicker] Clicking at (" + projPoint.x + "," + projPoint.y + ") with action: " + chopLabel);
            
            // Use the same client-thread approach as manual clicks
            final java.awt.Point finalClickPoint = projPoint;
            final String finalChopLabel = chopLabel;
            final java.util.concurrent.atomic.AtomicBoolean clickResult = new java.util.concurrent.atomic.AtomicBoolean(false);
            final java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
            
            context.clientThread.invoke(() -> {
                try {
                    boolean ok = context.input.moveAndClickWithValidation(finalClickPoint, finalChopLabel);
                    clickResult.set(ok);
                } catch (Exception e) {
                    context.logger.warn("[TreeClicker] moveAndClickWithValidation failed on client thread: " + e.getMessage());
                    clickResult.set(false);
                } finally {
                    latch.countDown();
                }
            });
            
            // Wait briefly for the click to execute
            try { 
                latch.await(600, java.util.concurrent.TimeUnit.MILLISECONDS); 
            } catch (InterruptedException ie) { 
                Thread.currentThread().interrupt(); 
            }
            
            if (clickResult.get()) {
                context.logger.info("[TreeClicker] Successfully clicked tree");
                context.setBusyForMs(300);
                return true;
            } else {
                context.logger.warn("[TreeClicker] Click validation failed");
                return false;
            }
            
        } catch (Exception e) {
            context.logger.warn("[TreeClicker] Error during tree click: " + e.getMessage());
            return false;
        }
    }
    
    
    /**
     * Attempts camera adjustments to make tree clickable.
     */
    public static boolean adjustCameraForTree(TaskContext context, GameObject tree) {
        if (tree == null) return false;
        
        int adjustmentAttempts = TreeDiscovery.getCameraAdjustmentAttempts(tree.getWorldLocation());
        if (adjustmentAttempts >= 5) {
            context.logger.warn("[TreeClicker] Too many camera adjustment attempts");
            return false;
        }
        
        TreeDiscovery.incrementCameraAdjustmentAttempts(tree.getWorldLocation());
        context.logger.info("[TreeClicker] Camera adjustment attempt " + (adjustmentAttempts + 1) + "/5");
        
        switch (adjustmentAttempts) {
            case 0:
                context.logger.info("[TreeClicker] Rotating camera right");
                context.input.rotateCameraRightSmall();
                break;
            case 1:
                context.logger.info("[TreeClicker] Rotating camera left");
                context.input.rotateCameraLeftSmall();
                break;
            case 2:
                context.logger.info("[TreeClicker] Tilting camera up");
                context.input.tiltCameraUpSmall();
                break;
            case 3:
                context.logger.info("[TreeClicker] Tilting camera down");
                context.input.tiltCameraDownSmall();
                break;
            default:
                context.logger.info("[TreeClicker] Zooming out");
                context.input.zoomOutSmall();
                break;
        }
        
        context.setBusyForMs(500);
        return true;
    }
}
