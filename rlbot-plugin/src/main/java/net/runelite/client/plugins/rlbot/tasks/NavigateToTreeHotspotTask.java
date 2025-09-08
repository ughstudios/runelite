package net.runelite.client.plugins.rlbot.tasks;

import java.util.List;
import net.runelite.api.coords.WorldPoint;

public class NavigateToTreeHotspotTask extends NavigateToHotspotTask {
    @Override
    protected List<WorldPoint> hotspots(TaskContext ctx) {
        // Scan for trees on every call to keep discovering new ones
        TreeDiscovery.scanAndDiscoverTrees(ctx);
        
        // Balance: prefer close/in-scene trees to avoid walking away from nearby targets.
        int wc = 1; try { wc = ctx.client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
        List<WorldPoint> availableAll = TreeDiscovery.getAvailableTrees();
        List<WorldPoint> available = new java.util.ArrayList<>();
        for (WorldPoint wp : availableAll) {
            if (wp != null && !TreeDiscovery.isDepleted(wp)) available.add(wp);
        }

        // Split by in-scene presence for higher reliability
        List<WorldPoint> inScene = new java.util.ArrayList<>();
        for (WorldPoint t : available) {
            if (net.runelite.api.coords.LocalPoint.fromWorld(ctx.client, t) != null) inScene.add(t);
        }
        if (!inScene.isEmpty()) {
            return inScene;
        }

        // If nothing in-scene, compare best-tier against any available, but don't ignore a very close non-best option
        List<WorldPoint> bestTier = TreeDiscovery.getBestAvailableTreesForLevel(wc);
        if (bestTier != null && !bestTier.isEmpty()) {
            WorldPoint me = ctx.client.getLocalPlayer() != null ? ctx.client.getLocalPlayer().getWorldLocation() : null;
            if (me != null && !available.isEmpty()) {
                int nearestAny = Integer.MAX_VALUE;
                for (WorldPoint a : available) { int d = me.distanceTo(a); if (d >= 0 && d < nearestAny) nearestAny = d; }
                int nearestBest = Integer.MAX_VALUE;
                for (WorldPoint b : bestTier) { int d = me.distanceTo(b); if (d >= 0 && d < nearestBest) nearestBest = d; }
                // If a non-best tree is very close and best is far, prefer the close option
                if (nearestAny <= 12 && nearestBest > 18) {
                    ctx.logger.info("[NavigateToTree] Preferring close tree (" + nearestAny + " tiles) over far best-tier (" + nearestBest + ")");
                    return available;
                }
            }
            return bestTier;
        }

        if (!available.isEmpty()) {
            return available;
        }
        // Explore locally to bootstrap tree discovery
        WorldPoint me = ctx.client.getLocalPlayer() != null ? ctx.client.getLocalPlayer().getWorldLocation() : null;
        if (me != null) {
            return java.util.Arrays.asList(
                new WorldPoint(me.getX()+8, me.getY()+2, me.getPlane()),
                new WorldPoint(me.getX()-8, me.getY()-2, me.getPlane()),
                new WorldPoint(me.getX()+2, me.getY()+8, me.getPlane()),
                new WorldPoint(me.getX()-2, me.getY()-8, me.getPlane())
            );
        }
        return List.of();
    }

    @Override
    protected boolean shouldBeActive(TaskContext ctx) {
        if (ctx.isInventoryNearFull()) {
            return false;
        }
        try {
            // If any choppable tree is visible or very near, do not navigate (let Chop task handle it)
            int wc = 1; try { wc = ctx.client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
            String[] allowed = TreeDiscovery.allowedTreeNamesForLevel(wc);
            net.runelite.api.GameObject vis = ObjectFinder.findNearestByNames(ctx, allowed, "Chop down");
            if (vis != null) {
                java.awt.Point proj = ObjectFinder.projectToCanvas(ctx, vis);
                if (proj != null) {
                    ctx.logger.info("[NavigateToTree] Visible tree found; skipping navigation");
                    return false;
                }
                // Not visible; if very near, also skip navigation
                net.runelite.api.Player me = ctx.client.getLocalPlayer();
                if (me != null) {
                    int dist = me.getWorldLocation().distanceTo(vis.getWorldLocation());
                    if (dist >= 0 && dist <= 8) {
                        ctx.logger.info("[NavigateToTree] Near tree (" + dist + " tiles); skipping navigation");
                        return false;
                    }
                }
            }
            // Fallback distance check using discovered trees
            net.runelite.api.Player me = ctx.client.getLocalPlayer();
            if (me != null) {
                net.runelite.api.coords.WorldPoint myWp = me.getWorldLocation();
                java.util.List<net.runelite.api.coords.WorldPoint> available = TreeDiscovery.getAvailableTrees();
                if (myWp != null && !available.isEmpty()) {
                    int nearest = Integer.MAX_VALUE;
                    for (net.runelite.api.coords.WorldPoint wp : available) {
                        if (wp == null) continue;
                        int d = myWp.distanceTo(wp);
                        if (d >= 0 && d < nearest) nearest = d;
                    }
                    if (nearest <= 15) {
                        ctx.logger.info("[NavigateToTree] Within " + nearest + " tiles of a tree; skipping navigation task");
                        return false;
                    }
                }
            }
        } catch (Exception ignored) {}
        return true;
    }
}

