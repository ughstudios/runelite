package net.runelite.client.plugins.rlbot.tasks;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import net.runelite.api.Client;
import net.runelite.api.Player;
import net.runelite.api.coords.WorldPoint;
 

/**
 * Base navigation task that steps toward a target hotspot on the minimap.
 */
public abstract class NavigateToHotspotTask implements Task {
    protected abstract List<WorldPoint> hotspots(TaskContext ctx);
    protected abstract boolean shouldBeActive(TaskContext ctx);

    @Override
    public boolean shouldRun(TaskContext context) {
        return shouldBeActive(context);
    }

    @Override
    public void run(TaskContext ctx) {
        UiHelper.closeObstructions(ctx);
        if (ctx.isBusy() && !ctx.timedOutSince(1500)) {
            return;
        }
        Client client = ctx.client;
        Player me = client.getLocalPlayer();
        if (me == null) return;

        // Choose nearest hotspot, then bias to a viewport-visible nearby tile to avoid walls
        WorldPoint myWp = me.getWorldLocation();
        List<WorldPoint> availableHotspots = hotspots(ctx);
        
        if (availableHotspots.isEmpty()) {
            // No discovered hotspots - explore randomly to find objects
            ctx.logger.info("[Nav] No hotspots discovered yet, exploring randomly");
            exploreRandomly(ctx, myWp);
            return;
        }
        
        final WorldPoint rawTarget = chooseNearest(myWp, availableHotspots);
        // Validate pathability using collision before committing
        WorldPoint target = ensurePathable(ctx, rawTarget);
        if (target == null) {
            ctx.logger.warn("[Nav] Raw target not pathable via collision; attempting nearby offsets");
            target = pickViewportReachableNear(ctx, rawTarget);
        }
        if (target == null) {
            ctx.logger.warn("[Nav] No reachable hotspots found");
            return;
        }

        int distance = myWp.distanceTo(target);
        ctx.updateNavProgress(distance);
        if (ctx.telemetry != null) {
            ctx.telemetry.setTargetName("(" + target.getX() + "," + target.getY() + ")");
            ctx.telemetry.setDistanceTiles(distance);
        }
        ctx.logger.info("[Nav] targetWP=(" + target.getX() + "," + target.getY() + ") distTiles=" + distance +
            " navNoProgressCount=" + ctx.getNavNoProgressCount());
        if (distance <= Math.max(5, ctx.config.nearHotspotTiles())) {
            // Try a world click step when near to increase precision
            boolean clicked = WorldPathing.clickStepToward(ctx, target, Math.max(2, Math.min(6, distance)));
            if (clicked) {
                ctx.setBusyForMs(500);
                return;
            }
            ctx.logger.info("[Nav] Near target (" + distance + " tiles). Precise world click failed, trying minimap.");
            MinimapPathing.stepTowards(ctx, target, 0.0);
            ctx.setBusyForMs(500);
            return;
        }

        // Rotate/tilt camera periodically to seek target
        final WorldPoint sweepTarget = target;
        CameraHelper.sweepUntilVisible(ctx, () -> {
            // consider visible if a world step toward target would be accepted
            return WorldPathing.clickStepToward(ctx, sweepTarget, 0);
        }, 4);
        // Turn run on when moving between hotspots
        RunHelper.ensureRunOn(ctx);

        // Compute jitter and step
        double jitter = Math.toRadians(ThreadLocalRandom.current().nextDouble(-12.0, 12.0));
        // Door handling removed with DoorHelper cleanup
        // Try a mid-distance world step before minimap
        boolean worldClicked = WorldPathing.clickStepToward(ctx, target, 6);
        if (!worldClicked) {
            MinimapPathing.stepTowards(ctx, target, jitter);
        }

        int busyMs = ThreadLocalRandom.current().nextInt(
            Math.max(200, ctx.config.navMinimapClickMsMin()),
            Math.max(ctx.config.navMinimapClickMsMin() + 1, ctx.config.navMinimapClickMsMax() + 1)
        );
        ctx.setBusyForMs(busyMs);

        // Stuck detection and recovery: after several no-progress windows, try different routes
        if (ctx.getNavNoProgressCount() >= Math.max(1, ctx.config.stuckRetries())) {
            ctx.logger.warn("[Nav] No progress for " + ctx.getNavNoProgressCount() + " attempts, performing recovery step");
            
            // Try multiple alternative routes to get around obstacles
            WorldPoint[] recoveryTargets = {
                new WorldPoint(target.getX() + 8, target.getY(), target.getPlane()),
                new WorldPoint(target.getX() - 8, target.getY(), target.getPlane()),
                new WorldPoint(target.getX(), target.getY() + 8, target.getPlane()),
                new WorldPoint(target.getX(), target.getY() - 8, target.getPlane()),
                new WorldPoint(myWp.getX() + 15, myWp.getY(), myWp.getPlane()), // Move away from current stuck position
                new WorldPoint(myWp.getX() - 15, myWp.getY(), myWp.getPlane()),
                new WorldPoint(myWp.getX(), myWp.getY() + 15, myWp.getPlane()),
                new WorldPoint(myWp.getX(), myWp.getY() - 15, myWp.getPlane())
            };
            
            // Try each recovery target
            for (WorldPoint recoveryTarget : recoveryTargets) {
                boolean recoveryClicked = WorldPathing.clickStepToward(ctx, recoveryTarget, 8);
                if (recoveryClicked) {
                    ctx.logger.info("[Nav] Using recovery route via: " + recoveryTarget);
                    ctx.setBusyForMs(800);
                    return;
                }
            }
            
            // If still stuck, try minimap with larger jitter
            ctx.logger.warn("[Nav] All recovery routes failed, trying minimap with large jitter");
            double largeJitter = Math.toRadians(ThreadLocalRandom.current().nextDouble(-45.0, 45.0));
            MinimapPathing.stepTowards(ctx, target, largeJitter);
            ctx.setBusyForMs(800);
            WorldPathing.clickStepToward(ctx, new WorldPoint(target.getX(), target.getY() + 2, target.getPlane()), 3);
            ctx.setBusyForMs(400);
            ctx.resetNavProgress();
        }
    }

    private static WorldPoint chooseNearest(WorldPoint from, List<WorldPoint> candidates) {
        if (from == null || candidates == null || candidates.isEmpty()) return null;
        WorldPoint best = null;
        int bestDist = Integer.MAX_VALUE;
        for (WorldPoint wp : candidates) {
            if (wp == null) continue;
            int d = from.distanceTo(wp);
            if (d >= 0 && d < bestDist) {
                bestDist = d;
                best = wp;
            }
        }
        return best;
    }

    private static WorldPoint pickViewportReachableNear(TaskContext ctx, WorldPoint target) {
        if (target == null) return null;
        // Probe offsets and pick the first whose canvas projection lands inside the viewport
        int[][] offsets = new int[][] { {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1}, {2,0}, {-2,0}, {1,1}, {1,-1}, {-1,1}, {-1,-1} };
        for (int[] off : offsets) {
            WorldPoint cand = new WorldPoint(target.getX() + off[0], target.getY() + off[1], target.getPlane());
            net.runelite.api.coords.LocalPoint lp = net.runelite.api.coords.LocalPoint.fromWorld(ctx.client, cand);
            if (lp == null) continue;
            net.runelite.api.Point p = net.runelite.api.Perspective.localToCanvas(ctx.client, lp, ctx.client.getPlane());
            if (p == null) continue;
            int vx = ctx.client.getViewportXOffset();
            int vy = ctx.client.getViewportYOffset();
            int vw = ctx.client.getViewportWidth();
            int vh = ctx.client.getViewportHeight();
            if (p.getX() >= vx && p.getY() >= vy && p.getX() < (vx + vw) && p.getY() < (vy + vh)) {
                return cand;
            }
        }
        return target;
    }

    static WorldPoint ensurePathable(TaskContext ctx, WorldPoint target) {
        try {
            if (target == null || ctx.client.getLocalPlayer() == null) return null;
            net.runelite.api.WorldView wv = ctx.client.getTopLevelWorldView();
            if (wv == null) return target; // fallback
            net.runelite.api.coords.WorldPoint me = ctx.client.getLocalPlayer().getWorldLocation();
            net.runelite.api.coords.WorldArea from = new net.runelite.api.coords.WorldArea(me, 1, 1);
            net.runelite.api.coords.WorldArea to = new net.runelite.api.coords.WorldArea(target, 1, 1);
            // Simple AABB direction step heuristic: if at least one step toward target is possible, accept
            int dx = Integer.signum(target.getX() - me.getX());
            int dy = Integer.signum(target.getY() - me.getY());
            if (from.canTravelInDirection(wv, dx, dy) || from.canTravelInDirection(wv, dx, 0) || from.canTravelInDirection(wv, 0, dy)) {
                return target;
            }
            // Try small offsets around the target to find a tile we can approach
            int[][] offsets = new int[][] { {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1}, {2,0}, {-2,0}, {0,2}, {0,-2} };
            for (int[] off : offsets) {
                net.runelite.api.coords.WorldPoint cand = new net.runelite.api.coords.WorldPoint(target.getX()+off[0], target.getY()+off[1], target.getPlane());
                dx = Integer.signum(cand.getX() - me.getX());
                dy = Integer.signum(cand.getY() - me.getY());
                if (from.canTravelInDirection(wv, dx, dy) || from.canTravelInDirection(wv, dx, 0) || from.canTravelInDirection(wv, 0, dy)) {
                    return cand;
                }
            }
            return null;
        } catch (Exception e) {
            return target; // be permissive on error
        }
    }

    private static void exploreRandomly(TaskContext ctx, WorldPoint myWp) {
        // Generate a random exploration target within a reasonable radius
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        int exploreRadius = 20; // tiles to explore around current position
        
        int offsetX = rng.nextInt(-exploreRadius, exploreRadius + 1);
        int offsetY = rng.nextInt(-exploreRadius, exploreRadius + 1);
        
        WorldPoint exploreTarget = new WorldPoint(
            myWp.getX() + offsetX,
            myWp.getY() + offsetY,
            myWp.getPlane()
        );
        
        ctx.logger.info("[Nav] Exploring toward (" + exploreTarget.getX() + "," + exploreTarget.getY() + ")");
        
        // Use the same navigation logic as normal hotspot navigation
        RunHelper.ensureRunOn(ctx);
        
        // Try world click first for short distances
        boolean worldClicked = WorldPathing.clickStepToward(ctx, exploreTarget, 6);
        if (!worldClicked) {
            // Fall back to minimap for longer distances
            double jitter = Math.toRadians(ThreadLocalRandom.current().nextDouble(-15.0, 15.0));
            MinimapPathing.stepTowards(ctx, exploreTarget, jitter);
        }
        
        int busyMs = ThreadLocalRandom.current().nextInt(
            Math.max(200, ctx.config.navMinimapClickMsMin()),
            Math.max(ctx.config.navMinimapClickMsMin() + 1, ctx.config.navMinimapClickMsMax() + 1)
        );
        ctx.setBusyForMs(busyMs);
    }

    
}


