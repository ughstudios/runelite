package net.runelite.client.plugins.rlbot.tasks;

/**
 * Specialized camera adjustment task that handles various camera movements
 * when the agent is idle, stuck, or not actively chopping trees.
 * This provides more intelligent camera behavior than the general CameraRotateTask.
 */
public class CameraAdjustmentTask implements Task {
    private static long lastAdjustmentMs = 0L;
    private static int adjustmentCount = 0;
    
    // Different types of camera adjustments for different situations
    private enum AdjustmentType {
        SCAN_FOR_TREES,      // Look around for trees when none visible
        UNSTUCK_ROTATION,    // Rotate when stuck navigating
        IDLE_EXPLORATION,    // Explore view when idle too long
        ZOOM_PERSPECTIVE,    // Adjust zoom for better perspective
        RESET_VIEW          // Reset to neutral view
    }
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Don't run if busy with other actions
        if (ctx.isBusy() && !ctx.timedOutSince(400)) return false;
        
        // Don't run too frequently
        long now = System.currentTimeMillis();
        if (now - lastAdjustmentMs < 2500) return false; // 2.5 second cooldown
        
        // Check specific conditions based on user requirements
        boolean shouldAdjust = false;
        String reason = "";
        
        // Primary conditions: Not chopping, not banking, not moving
        boolean isChopping = ctx.isWoodcuttingAnim();
        boolean isBanking = isBankOpen(ctx) || isDepositBoxOpen(ctx);
        boolean isMoving = ctx.isPlayerMovingRecent(3000); // 3 seconds
        boolean isWalking = ctx.isPlayerWalking();
        
        // Case 1: Agent is not chopping logs AND not moving for 3+ seconds
        if (!isChopping && !isMoving) {
            shouldAdjust = true;
            reason = "not chopping logs and hasn't moved for 3+ seconds";
        }
        
        // Case 2: Agent is not banking AND not moving for 3+ seconds  
        else if (!isBanking && !isMoving && !isChopping) {
            shouldAdjust = true;
            reason = "not banking and hasn't moved for 3+ seconds";
        }
        
        // Case 3: Agent is completely idle (not chopping, not banking, not walking) for 4+ seconds
        else if (!isChopping && !isBanking && !isWalking && !ctx.isPlayerMovingRecent(4000)) {
            shouldAdjust = true;
            reason = "completely idle for 4+ seconds (not chopping, banking, or walking)";
        }
        
        // Case 4: Navigation stuck (pathfinding issues)
        else if (ctx.getNavNoProgressCount() >= 3) {
            shouldAdjust = true;
            reason = "navigation stuck (no progress count: " + ctx.getNavNoProgressCount() + ")";
        }
        
        // Case 5: Agent hasn't moved for a long time (6+ seconds) regardless of other activities
        else if (!ctx.isPlayerMovingRecent(6000)) {
            shouldAdjust = true;
            reason = "hasn't moved for 6+ seconds (long idle period)";
        }
        
        if (shouldAdjust) {
            ctx.logger.info("[CameraAdjust] Should adjust camera - " + reason + 
                " (isChopping=" + isChopping + ", isBanking=" + isBanking + 
                ", isMoving=" + isMoving + ", isWalking=" + isWalking + ")");
        }
        
        return shouldAdjust;
    }
    
    /**
     * Check if bank interface is open
     */
    private boolean isBankOpen(TaskContext ctx) {
        try {
            net.runelite.api.widgets.Widget bank = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER);
            return bank != null && !bank.isHidden();
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Check if deposit box interface is open
     */
    private boolean isDepositBoxOpen(TaskContext ctx) {
        try {
            net.runelite.api.widgets.Widget depositBox = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.DEPOSIT_BOX_INVENTORY_ITEMS_CONTAINER);
            return depositBox != null && !depositBox.isHidden();
        } catch (Exception e) {
            return false;
        }
    }
    
    @Override
    public void run(TaskContext ctx) {
        UiHelper.closeObstructions(ctx);
        if (ctx.isBusy() && !ctx.timedOutSince(300)) return;
        
        // Select appropriate adjustment type based on situation
        AdjustmentType adjustmentType = selectAdjustmentType(ctx);
        ctx.logger.info("[CameraAdjust] Performing adjustment: " + adjustmentType);
        
        performAdjustment(ctx, adjustmentType);
        
        // Update tracking
        lastAdjustmentMs = System.currentTimeMillis();
        adjustmentCount = (adjustmentCount + 1) % 5; // Cycle through different priorities
        
        ctx.setBusyForMs(500); // Give time for camera to settle
    }
    
    /**
     * Select the most appropriate camera adjustment for the current situation
     */
    private AdjustmentType selectAdjustmentType(TaskContext ctx) {
        int navStuckCount = ctx.getNavNoProgressCount();
        boolean isVeryStuck = navStuckCount >= 6;
        boolean isIdle = !ctx.isWoodcuttingAnim() && !ctx.isPlayerWalking();
        boolean hasntMovedLong = !ctx.isPlayerMovingRecent(7000);
        
        // Priority 1: If very stuck with navigation, try unstuck rotation
        if (isVeryStuck || (navStuckCount >= 3 && hasntMovedLong)) {
            return AdjustmentType.UNSTUCK_ROTATION;
        }
        
        // Priority 2: If idle and not chopping, scan for trees
        if (isIdle && !ctx.isPlayerMovingRecent(4000)) {
            return AdjustmentType.SCAN_FOR_TREES;
        }
        
        // Priority 3: If idle for very long time, do exploration
        if (hasntMovedLong && isIdle) {
            return AdjustmentType.IDLE_EXPLORATION;
        }
        
        // Priority 4: Adjust zoom for better perspective
        if (adjustmentCount % 3 == 0) {
            return AdjustmentType.ZOOM_PERSPECTIVE;
        }
        
        // Default: Reset view to neutral position
        return AdjustmentType.RESET_VIEW;
    }
    
    /**
     * Perform the selected camera adjustment
     */
    private void performAdjustment(TaskContext ctx, AdjustmentType type) {
        try {
            switch (type) {
                case SCAN_FOR_TREES:
                    scanForTrees(ctx);
                    break;
                case UNSTUCK_ROTATION:
                    performUnstuckRotation(ctx);
                    break;
                case IDLE_EXPLORATION:
                    performIdleExploration(ctx);
                    break;
                case ZOOM_PERSPECTIVE:
                    adjustZoomPerspective(ctx);
                    break;
                case RESET_VIEW:
                    resetToNeutralView(ctx);
                    break;
            }
        } catch (Exception e) {
            ctx.logger.warn("[CameraAdjust] Error during adjustment: " + e.getMessage());
            // Fallback: simple rotation
            ctx.input.rotateCameraSafe(90, 0);
        }
    }
    
    /**
     * Scan around looking for trees by rotating camera
     */
    private void scanForTrees(TaskContext ctx) {
        ctx.logger.info("[CameraAdjust] Scanning for trees");
        
        // Rotate 60 degrees and zoom out for better tree visibility
        ctx.input.rotateCameraSafe(120, 0);
        ctx.input.zoomOutSmall();
        
        ctx.logger.info("[CameraAdjust] Rotated camera to scan for trees");
    }
    
    /**
     * Perform camera rotation to help get unstuck from navigation issues
     */
    private void performUnstuckRotation(TaskContext ctx) {
        int stuckCount = ctx.getNavNoProgressCount();
        ctx.logger.info("[CameraAdjust] Performing unstuck rotation (stuck count: " + stuckCount + ")");
        
        // More dramatic rotations for higher stuck counts
        if (stuckCount >= 8) {
            // Very stuck: 180 degree turn and zoom out
            ctx.input.rotateCameraSafe(240, 0);
            ctx.input.zoomOutSmall();
            ctx.input.zoomOutSmall();
            ctx.logger.info("[CameraAdjust] Performed dramatic 180° rotation for very stuck situation");
        } else if (stuckCount >= 5) {
            // Moderately stuck: 90 degree turn and tilt
            ctx.input.rotateCameraSafe(180, -30);
            ctx.input.zoomOutSmall();
            ctx.logger.info("[CameraAdjust] Performed 90° rotation with tilt for stuck situation");
        } else {
            // Slightly stuck: 45 degree turn
            ctx.input.rotateCameraSafe(90, 0);
            ctx.logger.info("[CameraAdjust] Performed 45° rotation for mild stuck situation");
        }
    }
    
    /**
     * Explore the surrounding area when idle for too long
     */
    private void performIdleExploration(TaskContext ctx) {
        ctx.logger.info("[CameraAdjust] Performing idle exploration");
        
        // Cycle through different viewing angles
        int cycle = adjustmentCount % 4;
        switch (cycle) {
            case 0:
                // Look left and up
                ctx.input.rotateCameraSafe(-120, -40);
                ctx.logger.info("[CameraAdjust] Looking left and up");
                break;
            case 1:
                // Look right and down
                ctx.input.rotateCameraSafe(120, 40);
                ctx.logger.info("[CameraAdjust] Looking right and down");
                break;
            case 2:
                // Zoom out for wider view
                ctx.input.zoomOutSmall();
                ctx.input.zoomOutSmall();
                ctx.logger.info("[CameraAdjust] Zooming out for wider view");
                break;
            case 3:
                // Reset and look around
                ctx.input.rotateCameraSafe(0, 0);
                ctx.input.rotateCameraSafe(180, 0);
                ctx.logger.info("[CameraAdjust] Reset view and look around");
                break;
        }
    }
    
    /**
     * Adjust zoom level for better perspective
     */
    private void adjustZoomPerspective(TaskContext ctx) {
        ctx.logger.info("[CameraAdjust] Adjusting zoom perspective");
        
        // Alternate between zooming in and out based on adjustment count
        if (adjustmentCount % 2 == 0) {
            ctx.input.zoomOutSmall();
            ctx.input.zoomOutSmall(); // Double zoom out for better visibility
            ctx.logger.info("[CameraAdjust] Zoomed out for better visibility");
        } else {
            ctx.input.zoomInSmall();
            ctx.logger.info("[CameraAdjust] Zoomed in for detail view");
        }
    }
    
    /**
     * Reset camera to a neutral, good overview position
     */
    private void resetToNeutralView(TaskContext ctx) {
        ctx.logger.info("[CameraAdjust] Resetting to neutral view");
        
        try {
            ctx.clientThread.invoke(() -> {
                try {
                    // Set to a good middle pitch and current yaw
                    ctx.client.setCameraPitchRelaxerEnabled(true);
                    ctx.client.setCameraPitchTarget(300); // Good overview pitch
                    ctx.logger.info("[CameraAdjust] Reset pitch to neutral position");
                } catch (Exception e) {
                    // Fallback to input handler
                    ctx.input.rotateCameraSafe(0, 0);
                    ctx.logger.info("[CameraAdjust] Used fallback neutral view reset");
                }
            });
        } catch (Exception e) {
            // Final fallback
            ctx.input.rotateCameraSafe(0, 0);
            ctx.logger.info("[CameraAdjust] Used final fallback neutral view reset");
        }
    }
}
