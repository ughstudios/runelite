package net.runelite.client.plugins.rlbot.tasks;

/**
 * Single small camera step in a specified direction.
 */
public class CameraRotateStepTask implements Task {
    public enum Direction { LEFT, RIGHT, UP, DOWN, NONE }

    private final Direction direction;

    public CameraRotateStepTask(Direction direction) {
        this.direction = direction != null ? direction : Direction.LEFT;
    }

    @Override
    public boolean shouldRun(TaskContext ctx) {
        return true;
    }

    @Override
    public void run(TaskContext ctx) {
        try {
            ctx.client.setCameraPitchRelaxerEnabled(true);
            int yaw = ctx.client.getCameraYawTarget();
            int pitch = ctx.client.getCameraPitchTarget();
            int dyaw = 0;
            int dpitch = 0;
            switch (direction) {
                case LEFT: dyaw = -96; break;    // ~17 degrees
                case RIGHT: dyaw = 96; break;
                case UP: dpitch = -16; break;    // small tilt up
                case DOWN: dpitch = 16; break;   // small tilt down
                case NONE: default: break;
            }
            int newYaw = (yaw + dyaw) & 0x7FF; // 0..2047
            int newPitch = Math.max(128, Math.min(512, pitch + dpitch));
            ctx.client.setCameraYawTarget(newYaw);
            ctx.client.setCameraPitchTarget(newPitch);
            ctx.setBusyForMs(150);
        } catch (Throwable t) {
            int dx = 0, dy = 0;
            switch (direction) {
                case LEFT: dx = -120; break;
                case RIGHT: dx = 120; break;
                case UP: dy = -40; break;
                case DOWN: dy = 40; break;
                case NONE: default: break;
            }
            ctx.input.rotateCameraSafe(dx, dy);
            ctx.setBusyForMs(150);
        }
    }
}

