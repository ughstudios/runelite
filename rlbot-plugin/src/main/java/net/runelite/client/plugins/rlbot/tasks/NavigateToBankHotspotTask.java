package net.runelite.client.plugins.rlbot.tasks;

import java.util.List;
import java.util.ArrayList;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;
import net.runelite.api.coords.WorldPoint;

public class NavigateToBankHotspotTask extends NavigateToHotspotTask {
    @Override
    protected List<WorldPoint> hotspots(TaskContext ctx) {
        // Scan for banks on every call to keep discovering new ones
        BankDiscovery.scanAndDiscoverBanks(ctx);
        
        List<WorldPoint> discovered = BankDiscovery.getDiscoveredBanks();
        if (!discovered.isEmpty()) {
            // Prefer banks currently within the loaded scene (LocalPoint != null),
            // not strictly requiring on-screen visibility. This avoids over-filtering.
            List<WorldPoint> inSceneBanks = new ArrayList<>();
            for (WorldPoint bankPos : discovered) {
                if (BankDiscovery.isBlacklisted(bankPos)) continue;
                net.runelite.api.coords.LocalPoint lp = net.runelite.api.coords.LocalPoint.fromWorld(ctx.client, bankPos);
                if (lp != null) {
                    inSceneBanks.add(bankPos);
                }
            }

            if (!inSceneBanks.isEmpty()) {
                return inSceneBanks;
            }

            // If none are in-scene (likely far), pursue the nearest discovered bank.
            WorldPoint me = ctx.client.getLocalPlayer() != null ? ctx.client.getLocalPlayer().getWorldLocation() : null;
            if (me != null) {
                WorldPoint nearest = null;
                int best = Integer.MAX_VALUE;
                for (WorldPoint wp : discovered) {
                    if (BankDiscovery.isBlacklisted(wp)) continue;
                    int d = me.distanceTo(wp);
                    if (d >= 0 && d < best) { best = d; nearest = wp; }
                }
                if (nearest != null) {
                    ctx.logger.info("[BankNav] No in-scene banks; navigating toward nearest discovered bank at " + nearest);
                    return java.util.Arrays.asList(nearest);
                }
            }
            ctx.logger.info("[BankNav] No suitable discovered banks selected; will explore to find new ones");
        }
        
        // If nothing discovered yet or all banks unreachable, bias exploration toward spawn-ish offsets to escape stalls
        WorldPoint me = ctx.client.getLocalPlayer() != null ? ctx.client.getLocalPlayer().getWorldLocation() : null;
        if (me != null) {
            return java.util.Arrays.asList(
                new WorldPoint(me.getX()+10, me.getY(), me.getPlane()),
                new WorldPoint(me.getX()-10, me.getY(), me.getPlane()),
                new WorldPoint(me.getX(), me.getY()+10, me.getPlane()),
                new WorldPoint(me.getX(), me.getY()-10, me.getPlane())
            );
        }
        return List.of();
    }

    // Note: Reachability precheck removed; NavigateToHotspotTask.ensurePathable handles collision checks.

    // Activation gating removed for RL-driven exploration (always eligible).

    private static boolean isBankOpen(TaskContext ctx) {
        try {
            Widget bank = ctx.client.getWidget(WidgetInfo.BANK_CONTAINER);
            return bank != null && !bank.isHidden();
        } catch (Exception e) {
            return false;
        }
    }
}
