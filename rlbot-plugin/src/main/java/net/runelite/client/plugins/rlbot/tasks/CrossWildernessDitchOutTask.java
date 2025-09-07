package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.api.MenuAction;
import net.runelite.api.ObjectComposition;
import net.runelite.api.widgets.Widget;
import net.runelite.api.coords.WorldPoint;
import net.runelite.client.plugins.rlbot.RLBotLogger;
import net.runelite.client.plugins.rlbot.input.RLBotInputHandler;

import java.awt.event.KeyEvent;
import java.util.LinkedList;
import java.util.Queue;

public class CrossWildernessDitchOutTask implements Task {
    private static final String DITCH_NAME = "Wilderness Ditch";
    private static final int WILDERNESS_Y_THRESHOLD = 3523; // Y coordinate that indicates wilderness
    
    private static long lastFailureMs = 0;
    private static final long FAILURE_COOLDOWN_MS = 5000; // 5 seconds
    private static int consecutiveFailures = 0;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    
    @Override
    public boolean shouldRun(TaskContext ctx) {
        ctx.logger.info("[WildernessDitchOut] shouldRun() - Starting eligibility check");
        
        if (ctx.isBusy() && !ctx.timedOutSince(600)) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = false (busy and not timed out)");
            return false;
        }
        if (ctx.client.getLocalPlayer() == null) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = false (no local player)");
            return false;
        }
        
        // Check if wilderness warning dialog is open
        if (isWildernessDialogOpen(ctx)) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = true (wilderness dialog is open)");
            return true; // Handle the dialog
        }
        
        // Only run if we're IN the wilderness
        WorldPoint playerPos = ctx.client.getLocalPlayer().getWorldLocation();
        if (playerPos != null && playerPos.getY() < WILDERNESS_Y_THRESHOLD) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = false (not in wilderness at " + playerPos + ")");
            return false; // Not in wilderness, no need to cross out
        }
        
        // Check failure cooldown
        long now = System.currentTimeMillis();
        if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            if (now - lastFailureMs < FAILURE_COOLDOWN_MS * 2) {
                ctx.logger.info("[WildernessDitchOut] shouldRun() = false (extended cooldown after max failures)");
                return false; // Extended cooldown after max failures
            } else {
                consecutiveFailures = 0; // Reset after extended cooldown
            }
        } else if (now - lastFailureMs < FAILURE_COOLDOWN_MS) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = false (regular cooldown after failure)");
            return false; // Regular cooldown after failure
        }

        net.runelite.api.GameObject ditch = ObjectFinder.findNearestByNames(ctx, new String[]{DITCH_NAME, "Ditch", "Wilderness Ditch", "Ditch (Wilderness)", "wilderness ditch", "ditch"}, "Cross");
        if (ditch == null) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = false (no ditch found)");
            return false;
        }
        
        // Only attempt if we're very close to the ditch (within 2 tiles)
        net.runelite.api.coords.WorldPoint ditchPos = ditch.getWorldLocation();
        if (playerPos != null && playerPos.distanceTo(ditchPos) > 2) {
            ctx.logger.info("[WildernessDitchOut] shouldRun() = false (too far from ditch: " + playerPos.distanceTo(ditchPos) + " tiles)");
            return false; // Too far from ditch
        }
        
        boolean canProject = ObjectFinder.projectToCanvas(ctx, ditch) != null;
        ctx.logger.info("[WildernessDitchOut] shouldRun() = " + canProject + " (can project to canvas: " + canProject + ")");
        return canProject;
    }
    
    @Override
    public void run(TaskContext ctx) {
        ctx.logger.info("[WildernessDitchOut] Starting wilderness ditch crossing (OUT)");
        
        if (ctx.isBusy() && !ctx.timedOutSince(600)) return;
        
        net.runelite.api.GameObject ditch = ObjectFinder.findNearestByNames(ctx, new String[]{DITCH_NAME, "Ditch", "Wilderness Ditch", "Ditch (Wilderness)", "wilderness ditch", "ditch"}, "Cross");
        if (ditch == null) {
            ctx.logger.warn("[WildernessDitchOut] Ditch object not found");
            recordFailure();
            return;
        }
        
        net.runelite.api.Client client = ctx.client;
        net.runelite.api.coords.WorldPoint initialPos = ctx.client.getLocalPlayer().getWorldLocation();
        
        // Get the ditch's local position
        net.runelite.api.coords.LocalPoint lp = ditch.getLocalLocation();
        if (lp == null) {
            ctx.logger.warn("[WildernessDitchOut] Could not get ditch local position");
            recordFailure();
            return;
        }
        
        // Invoke the explicit "Cross" menu option
        ObjectComposition comp = client.getObjectDefinition(ditch.getId());
        int crossIdx = -1;
        String label = "Cross";
        if (comp != null && comp.getActions() != null) {
            String[] acts = comp.getActions();
            for (int i = 0; i < acts.length; i++) {
                String a = acts[i];
                if (a != null && a.equalsIgnoreCase("Cross")) { crossIdx = i; label = a; break; }
            }
        }
        if (crossIdx < 0) {
            ctx.logger.warn("[WildernessDitchOut] Cross action not found on ditch object");
            recordFailure();
            // If the object lost the Cross option, just step across to try again later
            WorldPoint target = ditch.getWorldLocation();
            if (!WorldPathing.clickStepToward(ctx, target, 6)) MinimapPathing.stepTowards(ctx, target, 0.0);
            ctx.setBusyForMs(600);
            return;
        }
        
        // Move cursor overlay to the ditch's canvas projection so overlay matches action point
        final net.runelite.api.Point cp = net.runelite.api.Perspective.localToCanvas(client, lp, 0);
        if (cp != null) {
            try { 
                ctx.input.smoothMouseMove(new java.awt.Point(cp.getX(), cp.getY())); 
            } catch (Exception e) {
                ctx.logger.warn("[WildernessDitchOut] Failed to move mouse: " + e.getMessage());
            }
        }
        
        // Try direct canvas click first, then fallback to menu action
        final WorldPoint finalInitialPos = initialPos;
        final int finalCrossIdx = crossIdx;
        final int finalObjectId = ditch.getId();
        final String finalOpt = label;
        final String finalObjectName = comp != null ? comp.getName() : "Wilderness Ditch";
        
        ctx.clientThread.invoke(() -> {
            try {
                // First try direct canvas click
                if (cp != null) {
                    ctx.logger.info("[WildernessDitchOut] Trying direct canvas click at " + cp.getX() + "," + cp.getY());
                    ctx.input.click();
                }
                
                // Then try menu action
                ctx.logger.info("[WildernessDitchOut] Crossing via menuAction=GAME_OBJECT_FIRST_OPTION at scene(" + lp.getSceneX() + "," + lp.getSceneY() + ") with option '" + finalOpt + "' on '" + finalObjectName + "'");
                client.menuAction(finalCrossIdx, finalObjectId, MenuAction.GAME_OBJECT_FIRST_OPTION, lp.getSceneX(), lp.getSceneY(), finalOpt, finalObjectName);
                ctx.logger.info("[WildernessDitchOut] MenuAction invoked successfully");
                
                // Schedule dialog check and success check after a delay using a separate thread
                new Thread(() -> {
                    try {
                        // Set busy for dialog check delay
                        ctx.setBusyForMs(1000);
                        
                        // Check if wilderness dialog is open (must be on client thread)
                        ctx.clientThread.invoke(() -> {
                            if (isWildernessDialogOpen(ctx)) {
                                ctx.logger.info("[WildernessDitchOut] Wilderness dialog detected, handling it");
                                handleWildernessDialog(ctx);
                            } else {
                                ctx.logger.info("[WildernessDitchOut] No wilderness dialog detected");
                            }
                        });
                        
                        // Set busy for crossing completion delay
                        ctx.setBusyForMs(2000);
                        
                        // Check if we successfully crossed out
                        checkCrossingOutSuccess(ctx, finalInitialPos);
                        
                    } catch (Exception e) {
                        ctx.logger.error("[WildernessDitchOut] Thread error: " + e.getMessage());
                    }
                }).start();
                
            } catch (Exception e) {
                ctx.logger.error("[WildernessDitchOut] Failed to cross ditch: " + e.getMessage());
                recordFailure();
            }
        });
        
        ctx.setBusyForMs(1000); // Reduced from 3000ms
    }
    
    private void recordFailure() {
        consecutiveFailures++;
        lastFailureMs = System.currentTimeMillis();
    }
    
    private void checkCrossingOutSuccess(TaskContext ctx, WorldPoint initialPos) {
        if (ctx.client.getLocalPlayer() == null) return;
        
        WorldPoint currentPos = ctx.client.getLocalPlayer().getWorldLocation();
        if (currentPos == null) return;
        
        // Check if we moved and are now OUT of wilderness
        if (!currentPos.equals(initialPos) && currentPos.getY() < WILDERNESS_Y_THRESHOLD) {
            ctx.logger.info("[WildernessDitchOut] Successfully crossed out of wilderness! Initial: " + initialPos + ", Current: " + currentPos);
            consecutiveFailures = 0; // Reset failures on success
        } else if (!currentPos.equals(initialPos)) {
            ctx.logger.warn("[WildernessDitchOut] Moved but still in wilderness, counting as failure. Initial: " + initialPos + ", Current: " + currentPos);
            recordFailure();
        } else {
            ctx.logger.warn("[WildernessDitchOut] Failed to cross out - no movement detected. Position: " + currentPos);
            recordFailure();
        }
    }
    
    private boolean isWildernessDialogOpen(TaskContext ctx) {
        ctx.logger.info("[WildernessDitchOut] Checking for wilderness dialog...");
        
        // First check for the specific wilderness warning screen widget (475)
        Widget wildernessWidget = ctx.client.getWidget(475, 0);
        if (wildernessWidget != null && !wildernessWidget.isHidden()) {
            ctx.logger.info("[WildernessDitchOut] Found wilderness warning screen widget 475");
            return true;
        }
        
        // Check for common wilderness warning dialog widgets
        Widget[] dialogWidgets = {
            ctx.client.getWidget(219, 1), // Common dialog widget
            ctx.client.getWidget(229, 1), // Another dialog widget
            ctx.client.getWidget(193, 2), // Options dialog
            ctx.client.getWidget(217, 5), // Dialog options
            ctx.client.getWidget(162, 44), // NPC dialog
            ctx.client.getWidget(162, 45), // Player dialog
            ctx.client.getWidget(11, 2),   // Chat dialog
            ctx.client.getWidget(11, 3),   // Chat options
            ctx.client.getWidget(11, 4),   // Chat continue
            ctx.client.getWidget(11, 5),   // Chat options
            ctx.client.getWidget(162, 0),  // Root dialog
            ctx.client.getWidget(162, 1),  // Dialog text
            ctx.client.getWidget(162, 2),  // Dialog options
            ctx.client.getWidget(162, 3),  // Dialog continue
            ctx.client.getWidget(162, 4),  // Dialog options
            ctx.client.getWidget(162, 5),  // Dialog continue
            ctx.client.getWidget(162, 6),  // Dialog options
            ctx.client.getWidget(162, 7),  // Dialog continue
            ctx.client.getWidget(162, 8),  // Dialog options
            ctx.client.getWidget(162, 9),  // Dialog continue
            ctx.client.getWidget(162, 10), // Dialog options
            ctx.client.getWidget(162, 11), // Dialog continue
            ctx.client.getWidget(162, 12), // Dialog options
            ctx.client.getWidget(162, 13), // Dialog continue
            ctx.client.getWidget(162, 14), // Dialog options
            ctx.client.getWidget(162, 15), // Dialog continue
            ctx.client.getWidget(162, 16), // Dialog options
            ctx.client.getWidget(162, 17), // Dialog continue
            ctx.client.getWidget(162, 18), // Dialog options
            ctx.client.getWidget(162, 19), // Dialog continue
            ctx.client.getWidget(162, 20), // Dialog options
            ctx.client.getWidget(162, 21), // Dialog continue
            ctx.client.getWidget(162, 22), // Dialog options
            ctx.client.getWidget(162, 23), // Dialog continue
            ctx.client.getWidget(162, 24), // Dialog options
            ctx.client.getWidget(162, 25), // Dialog continue
            ctx.client.getWidget(162, 26), // Dialog options
            ctx.client.getWidget(162, 27), // Dialog continue
            ctx.client.getWidget(162, 28), // Dialog options
            ctx.client.getWidget(162, 29), // Dialog continue
            ctx.client.getWidget(162, 30), // Dialog options
            ctx.client.getWidget(162, 31), // Dialog continue
            ctx.client.getWidget(162, 32), // Dialog options
            ctx.client.getWidget(162, 33), // Dialog continue
            ctx.client.getWidget(162, 34), // Dialog options
            ctx.client.getWidget(162, 35), // Dialog continue
            ctx.client.getWidget(162, 36), // Dialog options
            ctx.client.getWidget(162, 37), // Dialog continue
            ctx.client.getWidget(162, 38), // Dialog options
            ctx.client.getWidget(162, 39), // Dialog continue
            ctx.client.getWidget(162, 40), // Dialog options
            ctx.client.getWidget(162, 41), // Dialog continue
            ctx.client.getWidget(162, 42), // Dialog options
            ctx.client.getWidget(162, 43), // Dialog continue
            ctx.client.getWidget(162, 44), // Dialog options
            ctx.client.getWidget(162, 45), // Dialog continue
            ctx.client.getWidget(162, 46), // Dialog options
            ctx.client.getWidget(162, 47), // Dialog continue
            ctx.client.getWidget(162, 48), // Dialog options
            ctx.client.getWidget(162, 49), // Dialog continue
            ctx.client.getWidget(162, 50)  // Dialog options
        };
        
        for (Widget widget : dialogWidgets) {
            if (widget != null && !widget.isHidden()) {
                String text = widget.getText();
                if (text != null && (text.toLowerCase().contains("wilderness") || 
                                   text.toLowerCase().contains("enter") ||
                                   text.toLowerCase().contains("warning") ||
                                   text.toLowerCase().contains("dangerous") ||
                                   text.toLowerCase().contains("cross"))) {
                    ctx.logger.info("[WildernessDitchOut] Found wilderness dialog widget: " + text);
                    return true;
                }
            }
        }
        
        // Also check all visible widgets using BFS
        Widget foundWidget = findWidgetByTextBFS(ctx, new String[]{"wilderness", "enter", "warning", "dangerous", "cross"});
        if (foundWidget != null) {
            ctx.logger.info("[WildernessDitchOut] Found wilderness dialog widget via BFS: " + foundWidget.getText());
            return true;
        }
        
        ctx.logger.info("[WildernessDitchOut] No wilderness dialog found");
        return false;
    }
    
    private void handleWildernessDialog(TaskContext ctx) {
        ctx.logger.info("[WildernessDitchOut] Handling wilderness warning dialog");
        
        // First try to find the specific "Enter Wilderness" button widget (475.11[0])
        Widget enterButtonParent = ctx.client.getWidget(475, 11);
        if (enterButtonParent != null && !enterButtonParent.isHidden()) {
            ctx.logger.info("[WildernessDitchOut] Found wilderness dialog widget 475.11: " + enterButtonParent.getText());
            
            // Look for child widgets that contain "Enter Wilderness" text
            Widget[] children = enterButtonParent.getChildren();
            if (children != null) {
                for (int i = 0; i < children.length; i++) {
                    Widget child = children[i];
                    if (child != null && !child.isHidden()) {
                        String childText = child.getText();
                        if (childText != null && childText.contains("Enter Wilderness")) {
                            ctx.logger.info("[WildernessDitchOut] Found Enter Wilderness child widget 475.11[" + i + "]: " + childText);
                            
                            // Click the enter wilderness button using menu action
                            ctx.clientThread.invoke(() -> {
                                try {
                                    ctx.client.menuAction(-1, child.getId(), MenuAction.CC_OP, 1, -1, "Enter Wilderness", "");
                                    ctx.logger.info("[WildernessDitchOut] Clicked enter wilderness child button via menu action");
                                } catch (Exception e) {
                                    ctx.logger.error("[WildernessDitchOut] Failed to click enter wilderness child button: " + e.getMessage());
                                }
                            });
                            
                            ctx.setBusyForMs(2000);
                            return;
                        }
                    }
                }
            }
            
            // If no child found, try clicking the parent widget itself
            ctx.logger.info("[WildernessDitchOut] No Enter Wilderness child found, trying parent widget");
            ctx.clientThread.invoke(() -> {
                try {
                    ctx.client.menuAction(-1, enterButtonParent.getId(), MenuAction.CC_OP, 1, -1, "Enter Wilderness", "");
                    ctx.logger.info("[WildernessDitchOut] Clicked enter wilderness parent button via menu action");
                } catch (Exception e) {
                    ctx.logger.error("[WildernessDitchOut] Failed to click enter wilderness parent button: " + e.getMessage());
                }
            });
            
            ctx.setBusyForMs(2000);
            return;
        }
        
        // Fallback: Find "Enter wilderness" or similar option using BFS
        Widget fallbackButton = findWidgetByTextBFS(ctx, new String[]{"Enter wilderness", "Enter", "Yes", "Continue"});
        
        if (fallbackButton != null) {
            ctx.logger.info("[WildernessDitchOut] Found enter wilderness button via BFS: " + fallbackButton.getText());
            
            // Click the enter wilderness button
            ctx.clientThread.invoke(() -> {
                try {
                    // Use menu action to click the widget
                    ctx.client.menuAction(-1, fallbackButton.getId(), MenuAction.CC_OP, 1, -1, fallbackButton.getText(), "");
                    ctx.logger.info("[WildernessDitchOut] Clicked enter wilderness button via BFS");
                } catch (Exception e) {
                    ctx.logger.error("[WildernessDitchOut] Failed to click enter wilderness button: " + e.getMessage());
                }
            });
            
            ctx.setBusyForMs(2000);
        } else {
            ctx.logger.warn("[WildernessDitchOut] Could not find enter wilderness button");
            // Try pressing space or enter as fallback
            ctx.clientThread.invoke(() -> {
                try {
                    ctx.input.pressKey(KeyEvent.VK_SPACE);
                    ctx.logger.info("[WildernessDitchOut] Pressed space key as fallback");
                } catch (Exception e) {
                    ctx.logger.error("[WildernessDitchOut] Failed to send space key: " + e.getMessage());
                }
            });
            ctx.setBusyForMs(1000);
        }
    }
    
    private Widget findWidgetByTextBFS(TaskContext ctx, String[] searchTexts) {
        Queue<Widget> queue = new LinkedList<>();
        
        // Start BFS from common dialog widget groups
        int[] dialogGroups = {219, 229, 193, 217, 162, 231, 233};
        
        for (int groupId : dialogGroups) {
            Widget rootWidget = ctx.client.getWidget(groupId, 0);
            if (rootWidget != null) {
                queue.offer(rootWidget);
            }
        }
        
        // BFS through all widgets
        while (!queue.isEmpty()) {
            Widget current = queue.poll();
            
            if (current == null || current.isHidden()) continue;
            
            // Check if this widget contains our target text
            String widgetText = current.getText();
            if (widgetText != null) {
                for (String searchText : searchTexts) {
                    if (widgetText.toLowerCase().contains(searchText.toLowerCase())) {
                        ctx.logger.info("[WildernessDitchOut] Found widget with text: '" + widgetText + "'");
                        return current;
                    }
                }
            }
            
            // Add children to queue
            Widget[] children = current.getChildren();
            if (children != null) {
                for (Widget child : children) {
                    if (child != null) {
                        queue.offer(child);
                    }
                }
            }
        }
        
        return null;
    }
}
