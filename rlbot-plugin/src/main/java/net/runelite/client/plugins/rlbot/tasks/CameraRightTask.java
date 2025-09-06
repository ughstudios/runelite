package net.runelite.client.plugins.rlbot.tasks;

/**
 * Camera right task - rotates the camera to the right.
 */
public class CameraRightTask implements Task {
    private static long lastRotateMs = 0L;
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Allow RL agent to decide when to rotate right
        return true;
    }
    
    @Override
    public void run(TaskContext ctx) {
        long now = System.currentTimeMillis();
        
        // Prevent too frequent rotation
        if (now - lastRotateMs < 800) {
            return;
        }
        
        lastRotateMs = now;
        
        try {
            // Get current camera yaw and rotate right by 45 degrees
            int currentYaw = ctx.client.getCameraYaw();
            int newYaw = (currentYaw + 256) & 2047; // 256 units = ~45 degrees, mask to keep in range
            
            ctx.client.setCameraYawTarget(newYaw);
            
            ctx.logger.info("[CameraRight] Rotated right from " + currentYaw + " to " + newYaw);
            ctx.setBusyForMs(150);
            
        } catch (Exception e) {
            ctx.logger.warn("[CameraRight] Error: " + e.getMessage());
        }
    }
}
