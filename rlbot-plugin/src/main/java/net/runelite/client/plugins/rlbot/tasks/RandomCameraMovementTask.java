package net.runelite.client.plugins.rlbot.tasks;

/**
 * Random small camera movement to vary the viewpoint.
 */
public class RandomCameraMovementTask implements Task {
    private static long lastMoveMs = 0L;

    @Override
    public boolean shouldRun(TaskContext ctx) {
        return true;
    }

    @Override
    public void run(TaskContext ctx) {
        long now = System.currentTimeMillis();
        if (now - lastMoveMs < 1200) return; // cooldown
        lastMoveMs = now;

        java.util.Random rng = ctx.rng();
        int pick = rng.nextInt(6);
        switch (pick) {
            case 0: ctx.input.rotateCameraLeftSmall(); break;
            case 1: ctx.input.rotateCameraRightSmall(); break;
            case 2: ctx.input.tiltCameraUpSmall(); break;
            case 3: ctx.input.tiltCameraDownSmall(); break;
            case 4: ctx.input.zoomInSmall(); break;
            default: ctx.input.zoomOutSmall(); break;
        }
        ctx.setBusyForMs(150);
    }
}

