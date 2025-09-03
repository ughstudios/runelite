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
            boolean inventoryFull = context.isInventoryNearFull();
            // Run if inventory is full (to deposit) OR if inventory is empty (to close bank)
            if (inventoryFull || context.getInventoryFreeSlots() == 28) {
                context.logger.info("[BankDeposit] Bank UI open, inventoryFull=" + inventoryFull + ", freeSlots=" + context.getInventoryFreeSlots());
            }
            return inventoryFull || context.getInventoryFreeSlots() == 28; // Close bank if empty
        }
        if (!context.isInventoryNearFull()) {
            // Only log when inventory is getting full but not quite there
            if (context.getInventoryFreeSlots() <= 5) {
                context.logger.info("[BankDeposit] shouldRun() = false (inventory not near full, free slots: " + context.getInventoryFreeSlots() + ")");
            }
            return false;
        }
        // Only when a bank object is visible on screen, not blacklisted, and has exact "Bank" action
        net.runelite.api.GameObject cand = ObjectFinder.findNearestByNames(context, new String[]{"bank booth", "bank chest"}, null);
        boolean canSeeBank = false;
        if (cand != null) {
            java.awt.Point proj = ObjectFinder.projectToCanvas(context, (net.runelite.api.GameObject) cand);
            if (proj != null) {
                net.runelite.api.coords.WorldPoint wp = cand.getWorldLocation();
                boolean blacklisted = BankDiscovery.isBlacklisted(wp);
                boolean hasBankAction = hasExactBankAction(context, cand);
                canSeeBank = !blacklisted && hasBankAction && context.isInventoryNearFull() && proj != null;
                // Only log when we actually find a usable bank
                if (canSeeBank) {
                    context.logger.info("[BankDeposit] Found usable bank: blacklisted=" + blacklisted + ", hasBankAction=" + hasBankAction + ", inventoryFull=" + context.isInventoryNearFull());
                }
            } else {
                context.logger.info("[BankDeposit] Bank object found but not projectable to canvas");
            }
        } else {
            // Only log when inventory is full and we can't find a bank
            if (context.isInventoryNearFull()) {
                context.logger.info("[BankDeposit] No bank objects found in scene");
            }
        }
        
        // If we found a bank, add it to our discovered banks
        if (canSeeBank) {
            BankDiscovery.scanAndDiscoverBanks(context);
        }
        
        return canSeeBank;
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
            
            // Also check for the "Deposit" button (which might be a different widget)
            // Try comprehensive widget scanning first
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
            
            // If no deposit button found, dump all visible widgets for debugging
            context.logger.info("[BankDeposit] No deposit button found, dumping all visible widgets for debugging");
            dumpAllVisibleWidgets(context);
            
            // Fallback: try common widget IDs for deposit buttons
            Widget[] depositWidgets = {
                client.getWidget(12, 9),  // Common bank deposit button location
                client.getWidget(12, 10), // Alternative location
                client.getWidget(12, 11), // Another possible location
                client.getWidget(12, 12)  // Yet another possible location
            };
            
            for (Widget depositWidget : depositWidgets) {
                if (depositWidget != null && !depositWidget.isHidden() && 
                    depositWidget.getBounds() != null && depositWidget.getBounds().width > 0) {
                    
                    String widgetText = depositWidget.getText();
                    if (widgetText != null && (widgetText.contains("Deposit") || widgetText.contains("deposit"))) {
                        context.logger.info("[BankDeposit] Found 'Deposit' button with text: '" + widgetText + "', clicking it");
                        int widgetId = depositWidget.getId();
                        context.clientThread.invoke(() -> {
                            try {
                                client.menuAction(-1, widgetId, MenuAction.CC_OP, 1, -1, "Deposit", "");
                            } catch (Exception ignored) {}
                        });
                        context.setBusyForMs(500);
                        if (context.telemetry != null) context.telemetry.addReward(10);
                        return;
                    }
                }
            }
            
            // If no deposit button found, try clicking on the deposit area directly
            context.logger.info("[BankDeposit] No deposit button found, trying to click deposit area");
            // Try clicking in the general deposit area (right side of bank interface)
            java.awt.Point depositArea = new java.awt.Point(400, 300); // Approximate location
            context.input.smoothMouseMove(depositArea);
            context.setBusyForMs(200);
            context.input.clickAt(depositArea);
            context.setBusyForMs(500);
            if (context.telemetry != null) context.telemetry.addReward(5);
            return;
        }
        
        // If deposit box widget is open, use its Deposit inventory button (gid=192, child=29)
        if (isDepositBoxOpen(context) && !context.isInventoryNearFull()) {
            // Close deposit box if nothing to deposit
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
            // Close deposit box: try close button scan else ESC
            if (!UiHelper.clickCloseIfVisible(context)) {
                context.input.pressKey(KeyEvent.VK_ESCAPE);
                context.setBusyForMs(250);
            }
            return;
        }

        // If bank widget is open, click deposit inventory (only if inventory is full)
        if (isBankOpen(context) && !context.isInventoryNearFull()) {
            if (!UiHelper.clickCloseIfVisible(context)) {
                context.input.pressKey(KeyEvent.VK_ESCAPE);
                context.setBusyForMs(200);
            }
            return;
        }
        Widget depositInv = client.getWidget(WidgetInfo.BANK_DEPOSIT_INVENTORY);
        if (depositInv != null && !depositInv.isHidden()) {
            context.logger.info("[BankDeposit] Found bank deposit widget, clicking it");
            int widgetId = depositInv.getId();
            context.clientThread.invoke(() -> {
                try {
                    client.menuAction(-1, widgetId, MenuAction.CC_OP, 1, -1, "Deposit inventory", "");
                } catch (Exception ignored) {}
            });
            context.setBusyForMs(500);
            if (context.telemetry != null) context.telemetry.addReward(10);
            // Close bank: try close button scan else ESC
            if (!UiHelper.clickCloseIfVisible(context)) {
                context.input.pressKey(KeyEvent.VK_ESCAPE);
                context.setBusyForMs(250);
            }
            return;
        }
        
        // Otherwise, find a nearby bank object and click it if on-screen
        Player me = context.client.getLocalPlayer();
        if (me == null) return;
        
        // Search for objects with "Bank" action specifically - only real bank booths
        final net.runelite.api.GameObject best = ObjectFinder.findNearestByAction(context, "Bank");

        if (best != null) {
            context.logger.info("[BankDeposit] Found bank object: " + best.getId() + " at " + best.getWorldLocation());
            
            // Enforce exact Bank action; if absent, blacklist and skip
            if (!hasExactBankAction(context, best)) {
                BankDiscovery.setLastTargetedBank(best.getWorldLocation());
                BankDiscovery.blacklistLastTargetedBank();
                context.logger.warn("[BankDeposit] Blacklisting non-bankable object: " + best.getId() + " at " + best.getWorldLocation());
                return;
            }
            // Ensure unobstructed; rotate camera if needed
            java.awt.Point proj = ObjectFinder.projectToCanvas(context, best);
            if (proj == null) {
                // Off-screen: sweep camera then try a world step toward bank
                boolean visible = CameraHelper.sweepUntilVisible(context, () -> ObjectFinder.projectToCanvas(context, (net.runelite.api.GameObject) best) != null, 6);
                if (!visible) {
                    WorldPathing.clickStepToward(context, best.getWorldLocation(), 6);
                    context.setBusyForMs(600);
                }
                return;
            }
            
            // Use direct mouse movement and clicking for more reliable interaction
            BankDiscovery.setLastTargetedBank(best.getWorldLocation());
            context.logger.info("[BankDeposit] Clicking bank object at " + best.getWorldLocation());
            
            try {
                // Move mouse to bank and click
                context.input.smoothMouseMove(proj);
                
                // Wait a bit for mouse movement to complete, then click
                context.setBusyForMs(300);
                
                // Click on the bank mesh
                context.input.click();
                context.setBusyForMs(200); // Reduced from 1200ms
                
                // Retry open once if widget not visible soon
                if (!isBankOpen(context)) {
                    context.setBusyForMs(300);
                    context.input.click();
                    context.setBusyForMs(200); // Reduced from 1200ms
                }
                
            } catch (Exception e) {
                context.logger.warn("[BankDeposit] Failed to interact with bank: " + e.getMessage());
            }
        } else {
            // No bank in scene: let NavigateToBankHotspotTask handle navigation
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


