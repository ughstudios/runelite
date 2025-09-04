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
            // Filter out unreachable bank positions (visible/in-scene), but if none are reachable,
            // still pursue the nearest discovered bank instead of aimless exploration.
            List<WorldPoint> reachableBanks = new ArrayList<>();
            for (WorldPoint bankPos : discovered) {
                if (isBankReachable(ctx, bankPos)) {
                    reachableBanks.add(bankPos);
                } else {
                    ctx.logger.info("[BankNav] Filtering out unreachable bank at " + bankPos);
                }
            }

            if (!reachableBanks.isEmpty()) {
                return reachableBanks;
            } else {
                // Choose the nearest discovered bank (non-blacklisted) and head toward it
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
                        ctx.logger.warn("[BankNav] No on-screen banks; navigating toward nearest discovered bank at " + nearest);
                        return java.util.Arrays.asList(nearest);
                    }
                }
                ctx.logger.warn("[BankNav] All discovered banks are unreachable and none selected; will explore to find new ones");
            }
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

    private boolean isBankReachable(TaskContext ctx, WorldPoint bankPos) {
        // Simplified reachability check - just check if we can see the bank
        try {
            // Convert WorldPoint to LocalPoint, then project to canvas
            net.runelite.api.coords.LocalPoint localPoint = net.runelite.api.coords.LocalPoint.fromWorld(ctx.client, bankPos);
            if (localPoint == null) {
                return false; // Position not in current scene
            }
            
            net.runelite.api.Point proj = net.runelite.api.Perspective.localToCanvas(ctx.client, localPoint, bankPos.getPlane());
            if (proj == null) {
                return false; // Position not visible on screen
            }
            
            // Check if the projected point is in the chat area
            if (isInChatArea(proj)) {
                return false;
            }
            
            // Check if there's actually a bank object at this specific location
            // Look for any bank object within 1 tile of the expected position
            net.runelite.api.GameObject bankObj = ObjectFinder.findNearestByAction(ctx, "Bank");
            if (bankObj != null) {
                WorldPoint bankLocation = bankObj.getWorldLocation();
                if (bankLocation.distanceTo(bankPos) <= 1) {
                    // Found a bank object close to the expected position
                    return true;
                }
            }
            
            // If no bank found at expected position, still allow navigation if we can see the position
            // This allows the agent to learn from trying to reach the position
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    private boolean isInChatArea(net.runelite.api.Point point) {
        // More reasonable chat area check - only the very bottom where chat input is
        return point.getY() > 480; // Only the very bottom chat input area
    }

    @Override
    protected boolean shouldBeActive(TaskContext ctx) {
        boolean inventoryFull = ctx.isInventoryFull();
        boolean bankOpen = isBankOpen(ctx);
        int freeSlots = ctx.getInventoryFreeSlots();
        
        ctx.logger.info("[BankNav] shouldBeActive check: inventoryFull=" + inventoryFull + 
                       " freeSlots=" + freeSlots + " bankOpen=" + bankOpen);
        
        if (!inventoryFull) return false;

        // If a usable, non-blacklisted bank object is currently visible, let BankDepositTask handle it
        try {
            final net.runelite.api.GameObject visibleBank = ObjectFinder.findNearestByAction(ctx, "Bank");
            if (visibleBank != null) {
                java.awt.Point p = ObjectFinder.projectToCanvas(ctx, visibleBank);
                boolean blacklisted = BankDiscovery.isBlacklisted(visibleBank.getWorldLocation());
                if (p != null && !blacklisted) {
                    ctx.logger.info("[BankNav] Usable bank visible on-screen; deferring to BankDepositTask");
                    return false;
                }
            }
        } catch (Exception ignored) {}

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


