package net.runelite.client.plugins.rlbot.tasks;

import java.util.Random;

/**
 * Task that performs random camera movements while the player is walking.
 * This adds human-like behavior by rotating the camera randomly during movement,
 * but avoids interfering with clicking or other actions.
 */
public class RandomCameraMovementTask implements Task {
    
    private static long lastCameraMovementMs = 0L;
    private static final Random random = new Random();
    
    // Camera movement types
    private enum CameraMovement {
        ROTATE_LEFT_SMALL,
        ROTATE_RIGHT_SMALL,
        TILT_UP_SMALL,
        TILT_DOWN_SMALL,
        ZOOM_IN_SMALL,
        ZOOM_OUT_SMALL,
        COMBINED_ROTATION
    }
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Only run when player is walking/moving
        if (!ctx.isPlayerWalking()) {
            return false;
        }
        
        // Don't run if busy with other actions (like clicking) - this includes dialog checks
        if (ctx.isBusy() && !ctx.timedOutSince(200)) {
            return false;
        }
        
        // Don't run too frequently - add some randomness to timing
        long now = System.currentTimeMillis();
        long minInterval = 2000 + random.nextInt(3000); // 2-5 seconds between movements
        if (now - lastCameraMovementMs < minInterval) {
            return false;
        }
        
        // Don't run during woodcutting animation
        if (ctx.isWoodcuttingAnim()) {
            return false;
        }
        
        // Don't run when bank interface is open
        if (isBankOpen(ctx)) {
            return false;
        }
        
        // Only run occasionally (30% chance when conditions are met)
        return random.nextFloat() < 0.3f;
    }
    
    @Override
    public void run(TaskContext ctx) {
        ctx.logger.info("[RandomCamera] Performing random camera movement while walking");
        
        // Select a random camera movement
        CameraMovement movement = selectRandomMovement();
        
        // Perform the movement
        performCameraMovement(ctx, movement);
        
        // Record the movement time
        lastCameraMovementMs = System.currentTimeMillis();
        
        // Set a brief busy period to prevent rapid successive movements
        ctx.setBusyForMs(300 + random.nextInt(500)); // 300-800ms busy time
    }
    
    /**
     * Select a random camera movement type
     */
    private CameraMovement selectRandomMovement() {
        CameraMovement[] movements = CameraMovement.values();
        return movements[random.nextInt(movements.length)];
    }
    
    /**
     * Perform the specified camera movement
     */
    private void performCameraMovement(TaskContext ctx, CameraMovement movement) {
        try {
            ctx.clientThread.invoke(() -> {
                try {
                    switch (movement) {
                        case ROTATE_LEFT_SMALL:
                            // Small left rotation
                            for (int i = 0; i < 2 + random.nextInt(3); i++) {
                                ctx.input.rotateCameraLeftSmall();
                                Thread.sleep(50 + random.nextInt(100)); // Human-like delays
                            }
                            ctx.logger.info("[RandomCamera] Rotated camera left");
                            break;
                            
                        case ROTATE_RIGHT_SMALL:
                            // Small right rotation
                            for (int i = 0; i < 2 + random.nextInt(3); i++) {
                                ctx.input.rotateCameraRightSmall();
                                Thread.sleep(50 + random.nextInt(100));
                            }
                            ctx.logger.info("[RandomCamera] Rotated camera right");
                            break;
                            
                        case TILT_UP_SMALL:
                            // Small upward tilt
                            for (int i = 0; i < 1 + random.nextInt(2); i++) {
                                ctx.input.tiltCameraUpSmall();
                                Thread.sleep(50 + random.nextInt(100));
                            }
                            ctx.logger.info("[RandomCamera] Tilted camera up");
                            break;
                            
                        case TILT_DOWN_SMALL:
                            // Small downward tilt
                            for (int i = 0; i < 1 + random.nextInt(2); i++) {
                                ctx.input.tiltCameraDownSmall();
                                Thread.sleep(50 + random.nextInt(100));
                            }
                            ctx.logger.info("[RandomCamera] Tilted camera down");
                            break;
                            
                        case ZOOM_IN_SMALL:
                            // Small zoom in
                            ctx.input.zoomInSmall();
                            Thread.sleep(100 + random.nextInt(200));
                            ctx.input.zoomInSmall();
                            ctx.logger.info("[RandomCamera] Zoomed in");
                            break;
                            
                        case ZOOM_OUT_SMALL:
                            // Small zoom out
                            ctx.input.zoomOutSmall();
                            Thread.sleep(100 + random.nextInt(200));
                            ctx.input.zoomOutSmall();
                            ctx.logger.info("[RandomCamera] Zoomed out");
                            break;
                            
                        case COMBINED_ROTATION:
                            // Combined horizontal and vertical movement
                            boolean horizontalFirst = random.nextBoolean();
                            if (horizontalFirst) {
                                // Horizontal then vertical
                                for (int i = 0; i < 1 + random.nextInt(2); i++) {
                                    if (random.nextBoolean()) {
                                        ctx.input.rotateCameraLeftSmall();
                                    } else {
                                        ctx.input.rotateCameraRightSmall();
                                    }
                                    Thread.sleep(50 + random.nextInt(100));
                                }
                                Thread.sleep(100 + random.nextInt(200));
                                for (int i = 0; i < 1 + random.nextInt(2); i++) {
                                    if (random.nextBoolean()) {
                                        ctx.input.tiltCameraUpSmall();
                                    } else {
                                        ctx.input.tiltCameraDownSmall();
                                    }
                                    Thread.sleep(50 + random.nextInt(100));
                                }
                            } else {
                                // Vertical then horizontal
                                for (int i = 0; i < 1 + random.nextInt(2); i++) {
                                    if (random.nextBoolean()) {
                                        ctx.input.tiltCameraUpSmall();
                                    } else {
                                        ctx.input.tiltCameraDownSmall();
                                    }
                                    Thread.sleep(50 + random.nextInt(100));
                                }
                                Thread.sleep(100 + random.nextInt(200));
                                for (int i = 0; i < 1 + random.nextInt(2); i++) {
                                    if (random.nextBoolean()) {
                                        ctx.input.rotateCameraLeftSmall();
                                    } else {
                                        ctx.input.rotateCameraRightSmall();
                                    }
                                    Thread.sleep(50 + random.nextInt(100));
                                }
                            }
                            ctx.logger.info("[RandomCamera] Performed combined rotation");
                            break;
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    ctx.logger.warn("[RandomCamera] Camera movement interrupted");
                } catch (Exception e) {
                    ctx.logger.error("[RandomCamera] Error performing camera movement: " + e.getMessage());
                }
            });
        } catch (Exception e) {
            ctx.logger.error("[RandomCamera] Error invoking camera movement: " + e.getMessage());
        }
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
}
