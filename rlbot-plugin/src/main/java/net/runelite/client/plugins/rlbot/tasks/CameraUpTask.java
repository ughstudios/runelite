package net.runelite.client.plugins.rlbot.tasks;

/**
 * Camera up task - tilts the camera upward.
 */
public class CameraUpTask implements Task {
    private static long lastTiltMs = 0L;
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Allow RL agent to decide when to tilt up
        return true;
    }
    
    @Override
    public void run(TaskContext ctx) {
        long now = System.currentTimeMillis();
        
        // Prevent too frequent tilting
        if (now - lastTiltMs < 800) {
            return;
        }
        
        lastTiltMs = now;
        
        try {
            // Get current camera pitch and tilt up
            int currentPitch = ctx.client.getCameraPitch();
            int newPitch = Math.max(128, currentPitch - 64); // Tilt up, min pitch is 128
            
            ctx.client.setCameraPitchTarget(newPitch);
            
            ctx.logger.info("[CameraUp] Tilted up from " + currentPitch + " to " + newPitch);
            ctx.setBusyForMs(150);
            
        } catch (Exception e) {
            ctx.logger.warn("[CameraUp] Error: " + e.getMessage());
        }
    }
}
