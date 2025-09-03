package net.runelite.client.plugins.rlbot.tasks;
import java.util.Random;
import net.runelite.api.Client;
import net.runelite.api.coords.WorldPoint;

/**
 * Minimap movement utilities. Computes a click point along the bearing from the
 * player to the target, compensating for minimap rotation.
 */
final class MinimapPathing {
    private MinimapPathing() {}
    private static WorldPoint lastStepWp = null;

    static void stepTowards(TaskContext ctx, WorldPoint target) {
        stepTowards(ctx, target, 0.0);
    }

    static void stepTowards(TaskContext ctx, WorldPoint target, double headingJitterRadians) {
        try {
            Client client = ctx.client;
            WorldPoint me = client.getLocalPlayer() != null ? client.getLocalPlayer().getWorldLocation() : null;
            if (me == null || target == null) {
                ctx.logger.warn("[Pathing] No player/target for minimap stepTowards");
                return;
            }

            int dxTiles = target.getX() - me.getX();
            int dyTiles = target.getY() - me.getY();
            if (dxTiles == 0 && dyTiles == 0) {
                ctx.logger.info("[Pathing] Already at target world point " + target.getX() + "," + target.getY());
                return;
            }

            // Choose a randomized step length for variability
            int stepLen = 9; // fixed length

            // Use angle + jitter to avoid repeating exact headings
            double baseAngle = Math.atan2(dyTiles, dxTiles);
            double jitter = headingJitterRadians;
            double angle = baseAngle + jitter;

            int stepDx = (int)Math.round(Math.cos(angle) * stepLen);
            int stepDy = (int)Math.round(Math.sin(angle) * stepLen);
            // Ensure we move at least 1 tile if possible
            if (stepDx == 0 && dxTiles != 0) stepDx = (dxTiles > 0 ? 1 : -1);
            if (stepDy == 0 && dyTiles != 0) stepDy = (dyTiles > 0 ? 1 : -1);

            WorldPoint stepTarget = new WorldPoint(me.getX() + stepDx, me.getY() + stepDy, me.getPlane());
            // Avoid repeating the exact same step target consecutively; nudge perpendicular
            if (lastStepWp != null && stepTarget.equals(lastStepWp)) {
                int altDx = -stepDy;
                int altDy = stepDx;
                // fixed perpendicular nudge
                stepTarget = new WorldPoint(me.getX() + altDx, me.getY() + altDy, me.getPlane());
            }
            ctx.logger.info("[Pathing] World step: stepTarget=(" + stepTarget.getX() + "," + stepTarget.getY() + ") from (" + me.getX() + "," + me.getY() + ")");

            final WorldPoint stepTargetFinal = stepTarget;
            boolean clicked = WorldPathing.clickStepToward(ctx, stepTargetFinal, Math.max(3, Math.min(stepLen, 12)));
            lastStepWp = stepTargetFinal;
            if (!clicked) {
                // If world click fails (projection/obstruction), nudge camera and retry a shorter step
                CameraHelper.sweepUntilVisible(ctx, () -> WorldPathing.clickStepToward(ctx, stepTargetFinal, 0), 3);
                WorldPathing.clickStepToward(ctx, new WorldPoint(me.getX() + stepDx / 2, me.getY() + stepDy / 2, me.getPlane()), Math.max(3, stepLen / 2));
            }
        } catch (Exception e) {
            ctx.logger.error("[Pathing] Error in stepTowards: " + e.getMessage());
        }
    }

    // Removed unused minimap widget probing helper
}


