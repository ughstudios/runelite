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

    @Override
    protected boolean shouldBeActive(TaskContext ctx) {
        boolean inventoryFull = ctx.isInventoryFull();
        boolean inventoryNearFull = ctx.isInventoryNearFull();
        boolean bankOpen = isBankOpen(ctx);
        int freeSlots = ctx.getInventoryFreeSlots();
        
        ctx.logger.info("[BankNav] shouldBeActive: full=" + inventoryFull + 
                       " nearFull=" + inventoryNearFull + " freeSlots=" + freeSlots + " bankOpen=" + bankOpen);
        
        // Be more aggressive - navigate to bank when inventory is full OR nearly full
        if (!inventoryFull && !inventoryNearFull) {
            ctx.logger.info("[BankNav] Inventory has plenty of space, not navigating to bank");
            return false;
        }

        // If a usable, non-blacklisted bank object is currently visible, let BankDepositTask handle it
        try {
            final net.runelite.api.GameObject visibleBank = ObjectFinder.findNearestBankInteractable(ctx);
            if (visibleBank != null) {
                java.awt.Point p = ObjectFinder.projectToCanvas(ctx, visibleBank);
                boolean blacklisted = BankDiscovery.isBlacklisted(visibleBank.getWorldLocation());
                if (p != null && !blacklisted) {
                    // Check if we're close enough to the bank to interact with it
                    net.runelite.api.Player me = ctx.client.getLocalPlayer();
                    if (me != null) {
                        int distance = me.getWorldLocation().distanceTo(visibleBank.getWorldLocation());
                        if (distance <= 2) {
                            ctx.logger.info("[BankNav] Very close to usable bank; deferring to BankDepositTask");
                            return false;
                        }
                    }
                    ctx.logger.info("[BankNav] Usable bank visible on-screen but not close enough; continuing navigation");
                }
            }
        } catch (Exception ignored) {}

        // Always navigate to bank when inventory is full/near full and bank is not open
        return !bankOpen;
    }

    private static boolean isBankOpen(TaskContext ctx) {
        try {
            Widget bank = ctx.client.getWidget(WidgetInfo.BANK_CONTAINER);
            return bank != null && !bank.isHidden();
        } catch (Exception e) {
            return false;
        }
    }
}
