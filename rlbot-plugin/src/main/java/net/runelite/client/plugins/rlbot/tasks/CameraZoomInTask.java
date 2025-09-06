package net.runelite.client.plugins.rlbot.tasks;

/**
 * Camera zoom in task - zooms the camera closer to the player.
 */
public class CameraZoomInTask implements Task {
    private static long lastZoomMs = 0L;
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        // Allow RL agent to decide when to zoom in
        return true;
    }
    
    @Override
    public void run(TaskContext ctx) {
        long now = System.currentTimeMillis();
        
        // Prevent too frequent zooming
        if (now - lastZoomMs < 1000) {
            return;
        }
        
        lastZoomMs = now;
        
        try {
            // Zoom in by scrolling up
            ctx.client.getCanvas().dispatchEvent(new java.awt.event.MouseWheelEvent(
                ctx.client.getCanvas(),
                java.awt.event.MouseEvent.MOUSE_WHEEL,
                System.currentTimeMillis(),
                0,
                ctx.client.getCanvas().getWidth() / 2,
                ctx.client.getCanvas().getHeight() / 2,
                1,
                false,
                java.awt.event.MouseWheelEvent.WHEEL_UNIT_SCROLL,
                1,
                -1 // Negative for zoom in
            ));
            
            ctx.logger.info("[CameraZoomIn] Zoomed in");
            ctx.setBusyForMs(150);
            
        } catch (Exception e) {
            ctx.logger.warn("[CameraZoomIn] Error: " + e.getMessage());
        }
    }
}
