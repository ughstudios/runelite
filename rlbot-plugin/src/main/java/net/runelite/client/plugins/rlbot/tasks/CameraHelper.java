package net.runelite.client.plugins.rlbot.tasks;

import java.util.function.BooleanSupplier;

final class CameraHelper {
    private CameraHelper() {}

    static boolean sweepUntilVisible(TaskContext ctx, BooleanSupplier isVisible, int maxSteps) {
        // Prefer native camera target setters; fallback to input drag if unavailable
        int steps = Math.max(1, maxSteps);
        // Deterministic deltas
        for (int i = 0; i < steps; i++) {
            if (isVisible.getAsBoolean()) {
                return true;
            }

            final int stepIndex = i;
            try {
                ctx.clientThread.invoke(() -> {
                    try {
                        // Enable relaxed pitch so we can tilt freely
                        ctx.client.setCameraPitchRelaxerEnabled(true);

                        int yawTarget = ctx.client.getCameraYawTarget();
                        int pitchTarget = ctx.client.getCameraPitchTarget();

                        // Human-like randomized nudges
                        int yawDelta = ((stepIndex % 2 == 0) ? 60 : -60);
                        int newYaw = (yawTarget + yawDelta) & 0x7FF; // wrap 0..2047

                        int pitchDelta = (stepIndex % 3 == 2) ? -12 : 0;
                        // Clamp pitch to a safe range (higher value = lower camera)
                        int newPitch = Math.max(128, Math.min(512, pitchTarget + pitchDelta));

                        ctx.client.setCameraYawTarget(newYaw);
                        ctx.client.setCameraPitchTarget(newPitch);
                    } catch (Throwable t) {
                        // Fallback: simulate middle-mouse drag if direct setters fail
                        int dx = ((stepIndex % 2 == 0) ? 36 : -36);
                        int dy = (stepIndex % 3 == 2) ? -12 : 0;
                        ctx.input.rotateCameraDrag(dx, dy);
                    }
                });
            } catch (Exception ignored) {
                int dx = ((stepIndex % 2 == 0) ? 36 : -36);
                int dy = (stepIndex % 3 == 2) ? -12 : 0;
                ctx.input.rotateCameraSafe(dx, dy);
            }

            ctx.setBusyForMs(140);
        }
        return isVisible.getAsBoolean();
    }

    static void alignNorth(TaskContext ctx, int maxSteps) {
        int steps = Math.max(1, maxSteps);
        for (int i = 0; i < steps; i++) {
            int mapAngle = ctx.client.getMapAngle(); // 0..2047
            double deg = (mapAngle * 360.0) / 2048.0;
            double leftDist = (deg + 360.0) % 360.0;
            double rightDist = (360.0 - leftDist) % 360.0;
            if (leftDist < 6.0 || rightDist < 6.0) {
                return;
            }
            if (leftDist < rightDist) {
                ctx.input.rotateCameraLeftSmall();
            } else {
                ctx.input.rotateCameraRightSmall();
            }
            ctx.setBusyForMs(80);
        }
    }

    static void sweepYawSmall(TaskContext ctx, int steps) {
        int s = Math.max(1, steps);
        for (int i = 0; i < s; i++) {
            final int stepIndex = i;
            try {
                ctx.clientThread.invoke(() -> {
                    try {
                        int yawTarget = ctx.client.getCameraYawTarget();
                        int delta = (stepIndex % 2 == 0) ? 32 : -32;
                        int newYaw = (yawTarget + delta) & 0x7FF;
                        ctx.client.setCameraYawTarget(newYaw);
                    } catch (Throwable t) {
                        int dx = (stepIndex % 2 == 0) ? 24 : -24;
                        ctx.input.rotateCameraDrag(dx, 0);
                    }
                });
            } catch (Exception ignored) {
                int dx = (stepIndex % 2 == 0) ? 24 : -24;
                ctx.input.rotateCameraSafe(dx, 0);
                ctx.input.rotateCameraSafe(dx, 0);
            }
            ctx.setBusyForMs(80);
        }
    }
}


