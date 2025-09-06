package net.runelite.client.plugins.rlbot.tasks;

/**
 * Camera left task - rotates the camera to the left.
 */
public class CameraLeftTask implements Task {
    private static long lastRotateMs = 0L;
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Allow RL agent to decide when to rotate left
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
            // Get current camera yaw and rotate left by 45 degrees
            int currentYaw = ctx.client.getCameraYaw();
            int newYaw = (currentYaw - 256) & 2047; // 256 units = ~45 degrees, mask to keep in range
            
            ctx.client.setCameraYawTarget(newYaw);
            
            ctx.logger.info("[CameraLeft] Rotated left from " + currentYaw + " to " + newYaw);
            ctx.setBusyForMs(150);
            
        } catch (Exception e) {
            ctx.logger.warn("[CameraLeft] Error: " + e.getMessage());
        }
    }
}
