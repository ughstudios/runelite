package net.runelite.client.plugins.rlbot.tasks;

import java.util.List;
import net.runelite.api.coords.WorldPoint;

public class NavigateToTreeHotspotTask extends NavigateToHotspotTask {
    @Override
    protected List<WorldPoint> hotspots(TaskContext ctx) {
        // Scan for trees on every call to keep discovering new ones
        TreeDiscovery.scanAndDiscoverTrees(ctx);
        
        // Prefer best-tier trees for current WC level when navigating long distances
        int wc = 1; try { wc = ctx.client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); } catch (Exception ignored) {}
        List<WorldPoint> bestTier = TreeDiscovery.getBestAvailableTreesForLevel(wc);
        if (bestTier != null && !bestTier.isEmpty()) {
            return bestTier;
        }
        
        // Fallback: any available trees
        List<WorldPoint> discovered = TreeDiscovery.getAvailableTrees();
        if (!discovered.isEmpty()) {
            // Double-check that none of the trees are actually depleted
            List<WorldPoint> available = new java.util.ArrayList<>();
            for (WorldPoint tree : discovered) {
                if (!TreeDiscovery.isDepleted(tree)) {
                    available.add(tree);
                } else {
                    ctx.logger.info("[NavigateToTree] Skipping depleted tree at " + tree);
                }
            }
            if (!available.isEmpty()) {
                return available;
            }
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
        return !ctx.isInventoryNearFull();
    }
}


