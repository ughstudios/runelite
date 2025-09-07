package net.runelite.client.plugins.rlbot.tasks;

/**
 * Camera down task - tilts the camera downward.
 */
public class CameraDownTask implements Task {
    private static long lastTiltMs = 0L;
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Allow RL agent to decide when to tilt down
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
            // Get current camera pitch and tilt down
            int currentPitch = ctx.client.getCameraPitch();
            int newPitch = Math.min(512, currentPitch + 64); // Tilt down, max pitch is 512
            
            ctx.client.setCameraPitchTarget(newPitch);
            
            ctx.logger.info("[CameraDown] Tilted down from " + currentPitch + " to " + newPitch);
            ctx.setBusyForMs(150);
            
        } catch (Exception e) {
            ctx.logger.warn("[CameraDown] Error: " + e.getMessage());
        }
    }
}
