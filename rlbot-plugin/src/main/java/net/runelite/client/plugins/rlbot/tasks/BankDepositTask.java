package net.runelite.client.plugins.rlbot.tasks;

import java.awt.event.KeyEvent;
import net.runelite.api.MenuAction;
import net.runelite.api.Client;
import net.runelite.api.Player;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;

/**
 * Open nearest bank booth and deposit inventory via widget if bank is open.
 * Very naive: clicks a nearby object named Bank booth/Bank chest.
 */
public class BankDepositTask implements Task {
    @Override
    public boolean shouldRun(TaskContext context) {
        // If a bank/deposit UI is open, run if inventory is full OR if we need to close the bank
        if (isBankOpen(context) || isDepositBoxOpen(context)) {
            boolean inventoryFull = context.isInventoryFull();
            if (inventoryFull || context.getInventoryFreeSlots() == 28) {
                context.logger.info("[BankDeposit] Bank UI open, inventoryFull=" + inventoryFull + ", freeSlots=" + context.getInventoryFreeSlots());
            }
            return inventoryFull || context.getInventoryFreeSlots() == 28; // Close bank if empty
        }
        if (!context.isInventoryFull()) {
            if (context.getInventoryFreeSlots() <= 5) {
                context.logger.info("[BankDeposit] shouldRun() = false (inventory not near full, free slots: " + context.getInventoryFreeSlots() + ")");
            }
            return false;
        }
        // Actively discover banks like Chop task does for trees
        BankDiscovery.scanAndDiscoverBanks(context);
        // Prefer objects with exact Bank action and not blacklisted
        net.runelite.api.GameObject cand = ObjectFinder.findNearestByAction(context, "Bank");
        boolean canSeeBank = false;
        if (cand != null) {
            java.awt.Point proj = ObjectFinder.projectToCanvas(context, (net.runelite.api.GameObject) cand);
            if (proj != null) {
                net.runelite.api.coords.WorldPoint wp = cand.getWorldLocation();
                boolean blacklisted = BankDiscovery.isBlacklisted(wp);
                boolean hasBankAction = hasExactBankAction(context, cand);
                canSeeBank = !blacklisted && hasBankAction && context.isInventoryFull();
                if (canSeeBank) {
                    context.logger.info("[BankDeposit] Found usable bank: blacklisted=" + blacklisted + ", hasBankAction=" + hasBankAction + ", inventoryFull=" + context.isInventoryFull());
                }
            }
        } else {
            context.logger.info("[BankDeposit] No bank objects found in scene");
        }
        if (!canSeeBank) {
            // Allow run if we have discovered banks so navigation can proceed
            boolean hasDiscovered = !BankDiscovery.getDiscoveredBanks().isEmpty();
            context.logger.info("[BankDeposit] shouldRun() - hasDiscoveredBanks: " + hasDiscovered + " (count=" + BankDiscovery.getDiscoveredBanks().size() + ")");
            return hasDiscovered;
        }
        return true;
    }

    private static boolean hasExactBankAction(TaskContext context, net.runelite.api.GameObject obj) {
        try {
            net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(obj.getId());
            if (comp == null) {
                context.logger.warn("[BankDeposit] No object composition for id=" + obj.getId());
                return false;
            }
            String[] actions = comp.getActions();
            if (actions == null) {
                context.logger.warn("[BankDeposit] No actions for object id=" + obj.getId() + ", name='" + comp.getName() + "'");
                return false;
            }
            context.logger.info("[BankDeposit] Object actions for '" + comp.getName() + "' (id=" + obj.getId() + "): " + java.util.Arrays.toString(actions));
            for (String a : actions) {
                if (a != null && "Bank".equals(a)) {
                    context.logger.info("[BankDeposit] Found exact 'Bank' action!");
                    return true;
                }
            }
            context.logger.warn("[BankDeposit] No exact 'Bank' action found for object '" + comp.getName() + "' (id=" + obj.getId() + ")");
        } catch (Exception e) {
            context.logger.warn("[BankDeposit] Exception checking bank action: " + e.getMessage());
        }
        return false;
    }

    @Override
    public void run(TaskContext context) {
        UiHelper.closeObstructions(context);
        if (context.isBusy() && !context.timedOutSince(4000)) {
            return;
        }
        Client client = context.client;
        
        // Check if bank is open
        boolean bankOpen = isBankOpen(context);
        
        if (bankOpen) {
            // If inventory is empty, close the bank
            if (context.getInventoryFreeSlots() == 28) {
                context.logger.info("[BankDeposit] Inventory is empty, closing bank");
                if (!UiHelper.clickCloseIfVisible(context)) {
                    context.input.pressKey(KeyEvent.VK_ESCAPE);
                    context.setBusyForMs(200);
                }
                return;
            }
            
            // Check for deposit inventory widget first
            Widget depositInv = client.getWidget(WidgetInfo.BANK_DEPOSIT_INVENTORY);
            if (depositInv != null && !depositInv.isHidden()) {
                context.logger.info("[BankDeposit] Found deposit inventory widget, clicking it");
                int widgetId = depositInv.getId();
                context.clientThread.invoke(() -> {
                    try {
                        client.menuAction(-1, widgetId, MenuAction.CC_OP, 1, -1, "Deposit inventory", "");
                    } catch (Exception ignored) {}
                });
                context.setBusyForMs(500);
                if (context.telemetry != null) context.telemetry.addReward(10);
                return;
            }
            
            Widget depositButton = findDepositButton(context);
            if (depositButton != null) {
                context.logger.info("[BankDeposit] Found deposit button via scanning, clicking it");
                int widgetId = depositButton.getId();
                context.clientThread.invoke(() -> {
                    try {
                        client.menuAction(-1, widgetId, MenuAction.CC_OP, 1, -1, "Deposit", "");
                    } catch (Exception ignored) {}
                });
                context.setBusyForMs(500);
                if (context.telemetry != null) context.telemetry.addReward(10);
                return;
            }
            
            // Fallback: try clicking deposit area
            context.logger.info("[BankDeposit] No deposit button found, trying to click deposit area");
            java.awt.Point depositArea = new java.awt.Point(400, 300); // Approximate location
            context.input.smoothMouseMove(depositArea);
            context.setBusyForMs(200);
            context.input.clickAt(depositArea);
            context.setBusyForMs(500);
            if (context.telemetry != null) context.telemetry.addReward(5);
            return;
        }
        
        // Deposit box handling (unchanged)
        if (isDepositBoxOpen(context) && !context.isInventoryFull()) {
            if (!UiHelper.clickCloseIfVisible(context)) {
                context.input.pressKey(KeyEvent.VK_ESCAPE);
                context.setBusyForMs(200);
            }
            return;
        }
        Widget depositBoxButton = client.getWidget(192, 29);
        if (depositBoxButton != null && !depositBoxButton.isHidden() && depositBoxButton.getBounds() != null && depositBoxButton.getBounds().width > 0) {
            context.logger.info("[BankDeposit] Found deposit box, clicking Deposit inventory");
            int widgetId = depositBoxButton.getId();
            context.clientThread.invoke(() -> {
                try {
                    client.menuAction(-1, widgetId, MenuAction.CC_OP, 1, -1, "Deposit inventory", "");
                } catch (Exception ignored) {}
            });
            context.setBusyForMs(500);
            if (context.telemetry != null) context.telemetry.addReward(10);
            if (!UiHelper.clickCloseIfVisible(context)) {
                context.input.pressKey(KeyEvent.VK_ESCAPE);
                context.setBusyForMs(250);
            }
            return;
        }

        // Otherwise, find a nearby bank object and click it if on-screen
        Player me = context.client.getLocalPlayer();
        if (me == null) return;
        
        // Search for objects with "Bank" action specifically
        final net.runelite.api.GameObject best = ObjectFinder.findNearestByAction(context, "Bank");

        if (best != null) {
            context.logger.info("[BankDeposit] Found bank object: " + best.getId() + " at " + best.getWorldLocation());
            
            if (!hasExactBankAction(context, best)) {
                context.logger.warn("[BankDeposit] Skipping non-bankable object: " + best.getId() + " at " + best.getWorldLocation());
                return;
            }
            if (BankDiscovery.isBlacklisted(best.getWorldLocation())) {
                context.logger.warn("[BankDeposit] Skipping blacklisted bank at " + best.getWorldLocation());
                return;
            }
            // Project bank to canvas and perform validated click like Chop task
            java.awt.Point proj = ObjectFinder.projectToCanvas(context, best);
            if (proj == null) {
                boolean visible = CameraHelper.sweepUntilVisible(context, () -> ObjectFinder.projectToCanvas(context, (net.runelite.api.GameObject) best) != null, 6);
                if (!visible) {
                    WorldPathing.clickStepToward(context, best.getWorldLocation(), 6);
                    context.setBusyForMs(600);
                }
                return;
            }
            // Move and click with validation on the bank action
            boolean clicked = context.input.moveAndClickWithValidation(new java.awt.Point(proj.x, proj.y), "Bank");
            if (!clicked) {
                context.logger.warn("[BankDeposit] Click validation failed on bank; stepping closer");
                WorldPathing.clickStepToward(context, best.getWorldLocation(), 4);
                context.setBusyForMs(400);
                return;
            }
            context.setBusyForMs(250);
            // If widget not open after interaction, retry via menuAction fallback
            if (!isBankOpen(context)) {
                context.logger.info("[BankDeposit] Bank not open after validated click; invoking menu action");
                context.input.interactWithGameObject(best, "Bank");
                context.setBusyForMs(300);
            }
            if (!isBankOpen(context)) {
                context.logger.warn("[BankDeposit] Bank still not open; backing off (chat handler may blacklist)");
                context.setBusyForMs(300);
            }
            return;
        }
        
        // No bank in scene: navigate toward nearest discovered bank or explore
        java.util.List<net.runelite.api.coords.WorldPoint> discovered = BankDiscovery.getDiscoveredBanks();
        if (!discovered.isEmpty()) {
            net.runelite.api.coords.WorldPoint meWp = me.getWorldLocation();
            net.runelite.api.coords.WorldPoint nearest = null;
            int bestDist = Integer.MAX_VALUE;
            for (net.runelite.api.coords.WorldPoint wp : discovered) {
                if (BankDiscovery.isBlacklisted(wp)) continue;
                int d = meWp.distanceTo(wp);
                if (d >= 0 && d < bestDist) { bestDist = d; nearest = wp; }
            }
            if (nearest != null) {
                context.logger.info("[BankDeposit] Navigating toward nearest discovered bank at " + nearest);
                boolean worldClicked = WorldPathing.clickStepToward(context, nearest, 6);
                if (!worldClicked) {
                    boolean visible = WorldPathing.clickStepToward(context, nearest, 0);
                    if (!visible) {
                        MinimapPathing.stepTowards(context, nearest, 0.0);
                    }
                }
                context.setBusyForMs(600);
                return;
            }
        }
        // If nothing discovered, trigger bank discovery exploration via NavigateToBankHotspotTask hotspots logic
        context.logger.info("[BankDeposit] No discovered banks to navigate to; exploring to find bank");
        NavigateToBankHotspotTask nav = new NavigateToBankHotspotTask();
        java.util.List<net.runelite.api.coords.WorldPoint> hs = nav.hotspots(context);
        if (!hs.isEmpty()) {
            net.runelite.api.coords.WorldPoint target = hs.get(0);
            boolean worldClicked = WorldPathing.clickStepToward(context, target, 6);
            if (!worldClicked) {
                MinimapPathing.stepTowards(context, target, 0.0);
            }
            context.setBusyForMs(600);
        }
    }

    private static boolean isBankOpen(TaskContext context) {
        Widget bank = context.client.getWidget(WidgetInfo.BANK_CONTAINER);
        return bank != null && !bank.isHidden();
    }

    private static boolean isDepositBoxOpen(TaskContext context) {
        try {
            // Deposit box group is 192; child 29 is the Deposit inventory button
            Widget w = context.client.getWidget(192, 29);
            if (w != null && !w.isHidden()) {
                java.awt.Rectangle b = w.getBounds();
                return b != null && b.width > 0 && b.height > 0;
            }
        } catch (Exception ignored) {}
        return false;
    }

    /**
     * Scan all visible widgets recursively using DFS to find deposit-related buttons.
     * This is much more comprehensive than checking specific widget IDs.
     */
    private static Widget findDepositButton(TaskContext context) {
        try {
            // DFS through all widget groups (0-1000+ to be thorough)
            for (int groupId = 0; groupId < 2000; groupId++) {
                Widget group = context.client.getWidget(groupId, 0);
                if (group == null || group.isHidden()) continue;
                
                // DFS through this group's children recursively
                Widget found = dfsFindDepositButton(context, group, 0);
                if (found != null) {
                    return found;
                }
            }
        } catch (Exception e) {
            context.logger.warn("[BankDeposit] Error scanning for deposit button: " + e.getMessage());
        }
        return null;
    }
    
    /**
     * Recursive DFS to find deposit button in widget tree.
     */
    private static Widget dfsFindDepositButton(TaskContext context, Widget widget, int depth) {
        if (widget == null || widget.isHidden() || depth > 10) return null; // Prevent infinite recursion
        
        try {
            // Check this widget's text first
            String text = widget.getText();
            if (text != null && !text.isEmpty()) {
                String lowerText = text.toLowerCase();
                if (lowerText.contains("deposit") || lowerText.contains("deposit inventory")) {
                    context.logger.info("[BankDeposit] Found deposit button via DFS (text): group=" + widget.getParentId() + 
                                     ", child=" + widget.getId() + ", text='" + text + "', depth=" + depth);
                    return widget;
                }
            }
            
            // Check for clickable widgets with sprites (buttons without text)
            if (widget.hasListener() && widget.getSpriteId() > 0) {
                // Log potential button candidates for debugging
                if (depth <= 3) { // Only log top-level candidates to avoid spam
                    context.logger.info("[BankDeposit] Found clickable widget: id=" + widget.getId() + 
                                     ", parent=" + widget.getParentId() + ", sprite=" + widget.getSpriteId() + 
                                     ", bounds=" + widget.getBounds() + ", depth=" + depth);
                }
                
                // Check if this looks like a deposit button based on sprite ID or position
                // Common deposit button sprites or positions
                if (widget.getSpriteId() == 170 || // From your screenshot
                    (widget.getBounds() != null && widget.getBounds().width > 50 && widget.getBounds().height > 20)) {
                    context.logger.info("[BankDeposit] Found potential deposit button (sprite): id=" + widget.getId() + 
                                     ", sprite=" + widget.getSpriteId() + ", bounds=" + widget.getBounds());
                    return widget;
                }
            }
            
            // DFS through children
            Widget[] children = widget.getChildren();
            if (children != null) {
                for (Widget child : children) {
                    Widget found = dfsFindDepositButton(context, child, depth + 1);
                    if (found != null) {
                        return found;
                    }
                }
            }
            
            // DFS through dynamic children if different from regular children
            Widget[] dynamicChildren = widget.getDynamicChildren();
            if (dynamicChildren != null) {
                for (Widget child : dynamicChildren) {
                    Widget found = dfsFindDepositButton(context, child, depth + 1);
                    if (found != null) {
                        return found;
                    }
                }
            }
            
        } catch (Exception e) {
            context.logger.warn("[BankDeposit] Error in DFS at depth " + depth + ": " + e.getMessage());
        }
        
        return null;
    }
    /**
     * Dump all visible widgets for debugging purposes.
     */
    private static void dumpAllVisibleWidgets(TaskContext context) {
        try {
            context.logger.info("[BankDeposit] === WIDGET DUMP START ===");
            
            // Scan through widget groups and dump visible ones
            for (int groupId = 0; groupId < 100; groupId++) { // Limit to first 100 groups to avoid spam
                Widget group = context.client.getWidget(groupId, 0);
                if (group == null || group.isHidden()) continue;
                
                context.logger.info("[BankDeposit] Group " + groupId + ": " + 
                                 "id=" + group.getId() + 
                                 ", text='" + group.getText() + "'" +
                                 ", sprite=" + group.getSpriteId() +
                                 ", hasListener=" + group.hasListener() +
                                 ", bounds=" + group.getBounds());
                
                // Dump children too
                Widget[] children = group.getChildren();
                if (children != null) {
                    for (Widget child : children) {
                        if (child == null || child.isHidden()) continue;
                        context.logger.info("[BankDeposit]   Child: id=" + child.getId() + 
                                         ", text='" + child.getText() + "'" +
                                         ", sprite=" + child.getSpriteId() +
                                         ", hasListener=" + child.hasListener() +
                                         ", bounds=" + child.getBounds());
                    }
                }
            }
            
            context.logger.info("[BankDeposit] === WIDGET DUMP END ===");
        } catch (Exception e) {
            context.logger.warn("[BankDeposit] Error dumping widgets: " + e.getMessage());
        }
    }
}


