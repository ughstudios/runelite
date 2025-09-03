package net.runelite.client.plugins.rlbot.tasks;

import java.util.List;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.Perspective;
import net.runelite.api.Player;
import net.runelite.api.MenuAction;
import net.runelite.api.MenuEntry;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.gameval.InventoryID;
import net.runelite.api.EquipmentInventorySlot;
import net.runelite.api.ItemContainer;
import net.runelite.api.Item;

/**
 * Click the nearest visible tree with a Chop action.
 */
public class ChopNearestTreeTask implements Task {
    // Track recent failed invocations to trigger recovery (camera/step) instead of spamming
    private static long lastInvokeMs = 0L;
    private static int recentInvokeFailures = 0;
    private static int lastTargetObjectId = -1;
    @Override
    public boolean shouldRun(TaskContext context) {
        if (context.isInventoryFull()) {
            context.logger.info("[ChopTask] shouldRun() = false (inventory full)");
            return false;
        }
        
        // Don't run if already woodcutting - let the player continue woodcutting
        if (context.isWoodcuttingAnim()) {
            context.logger.info("[ChopTask] shouldRun() = false (already woodcutting)");
            return false;
        }
        
        // CRITICAL FIX: Scan and mark depleted trees BEFORE searching for trees
        TreeDiscovery.scanAndDiscoverTrees(context);
        
        // Try multiple action names for trees - some might be "Chop", "Cut", "Chop down", etc.
        net.runelite.api.GameObject candidate = null;
        String[] actionNames = {"Chop down", "Chop", "Cut", "Cut down"};
        
        for (String actionName : actionNames) {
            candidate = ObjectFinder.findNearestByNames(context, new String[]{"tree", "oak", "willow", "yew", "maple"}, actionName);
            if (candidate != null) {
                // Double-check this isn't a stump
                try {
                    net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(candidate.getId());
                    if (comp != null) {
                        String name = comp.getName();
                        if (name != null && name.toLowerCase().contains("stump")) {
                            context.logger.info("[ChopTask] Found stump, skipping: " + name);
                            TreeDiscovery.markDepleted(candidate.getWorldLocation());
                            candidate = null;
                            continue;
                        }
                    }
                } catch (Exception e) {
                    context.logger.warn("[ChopTask] Error checking if tree is stump: " + e.getMessage());
                }
                
                if (candidate != null) {
                    context.logger.info("[ChopTask] shouldRun() - Found candidate tree with action '" + actionName + "': " + (candidate != null));
                    break;
                }
            }
        }
        
        if (candidate == null) {
            context.logger.info("[ChopTask] shouldRun() - No trees found with any chop actions");
            // Let's also log what trees we can see and their actions for debugging
            try {
                net.runelite.api.Scene scene = context.client.getScene();
                if (scene != null) {
                    net.runelite.api.Tile[][][] tiles = scene.getTiles();
                    for (int z = 0; z < tiles.length; z++) {
                        for (int x = 0; x < tiles[z].length; x++) {
                            for (int y = 0; y < tiles[z][x].length; y++) {
                                net.runelite.api.Tile tile = tiles[z][x][y];
                                if (tile == null) continue;
                                for (net.runelite.api.GameObject go : tile.getGameObjects()) {
                                    if (go == null) continue;
                                    net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(go.getId());
                                    if (comp == null) continue;
                                    String name = comp.getName();
                                    if (name == null) continue;
                                    String lowerName = name.toLowerCase();
                                    if (lowerName.contains("tree") || lowerName.contains("oak") || lowerName.contains("willow") || lowerName.contains("yew") || lowerName.contains("maple")) {
                                        context.logger.info("[ChopTask] Found tree: " + name + " at " + go.getWorldLocation());
                                        String[] actions = comp.getActions();
                                        if (actions != null) {
                                            StringBuilder sb = new StringBuilder();
                                            sb.append("[ChopTask] Tree actions: ");
                                            for (int i = 0; i < actions.length; i++) {
                                                String action = actions[i];
                                                sb.append("[").append(i).append("] ").append(action != null ? action : "null").append("; ");
                                            }
                                            context.logger.info(sb.toString());
                                        } else {
                                            context.logger.info("[ChopTask] Tree has no actions");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (Exception e) {
                context.logger.warn("[ChopTask] Error scanning for tree actions: " + e.getMessage());
            }
        }
        
        if (candidate != null) {
            // Double-check that this tree actually has a chop action
            try {
                net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(candidate.getId());
                if (comp != null && comp.getActions() != null) {
                    boolean hasChopAction = false;
                    for (String action : comp.getActions()) {
                        if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                            hasChopAction = true;
                            break;
                        }
                    }
                    if (!hasChopAction) {
                        context.logger.info("[ChopTask] Tree at " + candidate.getWorldLocation() + " has no chop action, marking as depleted");
                        TreeDiscovery.markDepleted(candidate.getWorldLocation());
                        return false;
                    }
                }
            } catch (Exception e) {
                context.logger.warn("[ChopTask] Error checking tree actions in shouldRun: " + e.getMessage());
                TreeDiscovery.markDepleted(candidate.getWorldLocation());
                return false;
            }
        }
        
        boolean canSeeTree = candidate != null && ObjectFinder.projectToCanvas(context, candidate) != null;
        context.logger.info("[ChopTask] shouldRun() - canSeeTree: " + canSeeTree);
        
        if (canSeeTree) {
            // Check if the tree is in wilderness and we're not
            net.runelite.api.coords.WorldPoint treePos = candidate.getWorldLocation();
            net.runelite.api.coords.WorldPoint playerPos = context.client.getLocalPlayer() != null ? 
                context.client.getLocalPlayer().getWorldLocation() : null;
                
            if (treePos != null && playerPos != null) {
                boolean treeInWilderness = treePos.getY() > 3523;
                boolean playerInWilderness = playerPos.getY() > 3523;
                
                context.logger.info("[ChopTask] Tree at " + treePos + " (wilderness: " + treeInWilderness + "), Player at " + playerPos + " (wilderness: " + playerInWilderness + ")");
                
                if (treeInWilderness && !playerInWilderness) {
                    context.logger.info("[ChopTask] Tree is in wilderness but player is not. Need to cross wilderness ditch first.");
                    return false; // Let CrossWildernessDitchTask handle this
                }
            }
            
            TreeDiscovery.scanAndDiscoverTrees(context);
            context.logger.info("[ChopTask] shouldRun() = true (can see tree and no wilderness issues)");
            return true;
        }
        // If no visible tree, still allow run so the task can trigger navigation fallback
        // when there are discovered trees or we can explore to find some
        boolean hasDiscovered = !TreeDiscovery.getDiscoveredTrees().isEmpty();
        if (!hasDiscovered) {
            // Try a quick scan to populate
            TreeDiscovery.scanAndDiscoverTrees(context);
            hasDiscovered = !TreeDiscovery.getDiscoveredTrees().isEmpty();
        }
        context.logger.info("[ChopTask] shouldRun() - hasDiscovered: " + hasDiscovered + " (discovered trees: " + TreeDiscovery.getDiscoveredTrees().size() + ")");
        return hasDiscovered;
    }

    @Override
    public void run(TaskContext context) {
        UiHelper.closeObstructions(context);
        if (context.isBusy() && !context.timedOutSince(4000)) {
            return;
        }
        Client client = context.client;
        Player me = client.getLocalPlayer();
        if (me == null) return;

        // Restrict target trees based on player's woodcutting level
        int wcLevel = 1;
        try {
            wcLevel = context.client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING);
        } catch (Exception ignored) {}
        String[] allowed = TreeDiscovery.allowedTreeNamesForLevel(wcLevel);
        
        // Only log woodcutting level once per session
        context.logger.info("[ChopTask] Player woodcutting level: " + wcLevel + ", allowed trees: " + java.util.Arrays.toString(allowed));
        
        // First, try to find a non-depleted tree nearby with chop action
        GameObject found = null;
        String[] actionNames = {"Chop down", "Chop", "Cut", "Cut down"};
        
        for (int attempts = 0; attempts < 3; attempts++) {
            for (String actionName : actionNames) {
                found = ObjectFinder.findNearestByNames(context, allowed, actionName);
                if (found != null && !TreeDiscovery.isDepleted(found.getWorldLocation())) {
                    context.logger.info("[ChopTask] Found tree with action '" + actionName + "' at " + found.getWorldLocation());
                    // Double-check that this tree actually has a chop action
                    try {
                        net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(found.getId());
                                                    if (comp != null && comp.getActions() != null) {
                                boolean hasChopAction = false;
                                for (String action : comp.getActions()) {
                                    if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                                        hasChopAction = true;
                                        break;
                                    }
                                }
                                if (!hasChopAction) {
                                    context.logger.info("[ChopTask] Tree at " + found.getWorldLocation() + " has no chop action, marking as depleted");
                                    TreeDiscovery.markDepleted(found.getWorldLocation());
                                    found = null; // Reset to find another
                                    continue;
                                }
                                // Log if we found a willow tree specifically
                                if (comp.getName() != null && comp.getName().toLowerCase().contains("willow")) {
                                    context.logger.warn("[ChopTask] WARNING: Found willow tree (ID: " + found.getId() + ") but player level is " + wcLevel + " (requires 30)");
                                }
                            }
                    } catch (Exception e) {
                        context.logger.warn("[ChopTask] Error checking tree actions: " + e.getMessage());
                        TreeDiscovery.markDepleted(found.getWorldLocation());
                        found = null;
                        continue;
                    }
                    break; // Found a valid non-depleted tree with chop action
                } else if (found != null) {
                    context.logger.info("[ChopTask] Found depleted tree at " + found.getWorldLocation() + ", trying to find another");
                    // Mark this tree as depleted so we don't find it again
                    TreeDiscovery.markDepleted(found.getWorldLocation());
                    found = null; // Reset to find another
                }
            }
            if (found != null) break; // Found a valid tree, exit the attempts loop
        }
        
        if (found == null || TreeDiscovery.isDepleted(found.getWorldLocation())) {
            context.logger.info("[Task] No available trees nearby, exploring to find new ones");
            // Navigate to find more trees
            net.runelite.api.coords.WorldPoint myWp0 = context.client.getLocalPlayer() != null ? context.client.getLocalPlayer().getWorldLocation() : null;
            if (myWp0 != null) {
                // Try to find any available tree, not just the nearest
                List<net.runelite.api.coords.WorldPoint> availableTrees = TreeDiscovery.getAvailableTrees();
                if (!availableTrees.isEmpty()) {
                    // Pick a random available tree to navigate to
                    net.runelite.api.coords.WorldPoint targetTree = availableTrees.get((int)(Math.random() * availableTrees.size()));
                    context.logger.info("[Task] Navigating to available tree at " + targetTree);
                    WorldPathing.clickStepToward(context, targetTree, 6);
                    context.setBusyForMs(500);
                } else {
                    // No available trees, explore to find new ones
                    context.logger.info("[Task] No available trees found, exploring to discover new ones");
                    // Trigger exploration by moving in a random direction
                    int dx = (int)(Math.random() * 20) - 10;
                    int dy = (int)(Math.random() * 20) - 10;
                    net.runelite.api.coords.WorldPoint explorePoint = new net.runelite.api.coords.WorldPoint(
                        myWp0.getX() + dx, myWp0.getY() + dy, myWp0.getPlane());
                    context.logger.info("[Task] Exploring to " + explorePoint + " to find new trees");
                    WorldPathing.clickStepToward(context, explorePoint, 6);
                    context.setBusyForMs(500);
                }
            }
            return;
        }

        if (found != null) {
            final GameObject best = found;
            
            // Debug: Check if this tree is depleted
            boolean isDepleted = TreeDiscovery.isDepleted(best.getWorldLocation());
            context.logger.info("[Task] Found tree at " + best.getWorldLocation() + ", depleted: " + isDepleted);
            
            // Debug: Log the tree's object composition and actions
            try {
                net.runelite.api.ObjectComposition comp = client.getObjectDefinition(best.getId());
                if (comp != null) {
                    context.logger.info("[Task] Tree object: " + comp.getName() + " (ID: " + best.getId() + ")");
                    String[] actions = comp.getActions();
                    if (actions != null) {
                        StringBuilder sb = new StringBuilder();
                        sb.append("[Task] Tree actions: ");
                        for (int i = 0; i < actions.length; i++) {
                            String action = actions[i];
                            sb.append("[").append(i).append("] ").append(action != null ? action : "null").append("; ");
                        }
                        context.logger.info(sb.toString());
                    } else {
                        context.logger.info("[Task] Tree has no actions");
                    }
                }
            } catch (Exception e) {
                context.logger.warn("[Task] Error getting tree composition: " + e.getMessage());
            }
            
            // Log all available menu entries to see what options we have on the focused object
            MenuEntry[] entries = client.getMenuEntries();
            if (entries != null && entries.length > 0) {
                StringBuilder sb = new StringBuilder();
                sb.append("[Task] Current menu entries (top->bottom): ");
                for (int i = 0; i < entries.length; i++) {
                    MenuEntry e = entries[i];
                    sb.append("[").append(i).append("] ")
                      .append(e.getType()).append(" | ")
                      .append(e.getOption()).append(" -> ")
                      .append(e.getTarget()).append("; ");
                }
                context.logger.info(sb.toString());
            } else {
                context.logger.info("[Task] No menu entries available right now");
            }

            // Prefer direct menuAction for the object's "Chop down" option instead of generic walk
            String[] actions = new String[0];
            String targetName = "Tree";
            try {
                net.runelite.api.ObjectComposition comp = client.getObjectDefinition(best.getId());
                if (comp != null) {
                    if (comp.getActions() != null) actions = comp.getActions();
                    if (comp.getName() != null && !comp.getName().isEmpty()) targetName = comp.getName();
                }
            } catch (Exception ignored) {}
            int chopIdx = -1;
            String chopLabel = null;
            if (actions != null) {
                for (int i = 0; i < actions.length; i++) {
                    String a = actions[i];
                    if (a == null) continue;
                    String al = a.toLowerCase();
                    if (al.contains("chop down") || al.contains("chop")) {
                        chopIdx = i;
                        chopLabel = a;
                        break;
                    }
                }
            }

            if (chopIdx >= 0) {
                // Avoid dead/sapling trees by name (but allow evergreen trees)
                try {
                    net.runelite.api.ObjectComposition comp = client.getObjectDefinition(best.getId());
                    String nm = comp != null ? comp.getName() : null;
                    String ln = nm != null ? nm.toLowerCase() : "";
                    if (ln.contains("dead") || ln.contains("burnt") || ln.contains("sapling")) {
                        context.logger.info("[Task] Skipping invalid tree type: " + nm);
                        TreeDiscovery.markDepleted(best.getWorldLocation());
                        return;
                    }
                } catch (Exception ignored) {}
                
                // Move cursor overlay to the object's canvas projection so overlay matches action point
                net.runelite.api.Point cp = Perspective.localToCanvas(client, best.getLocalLocation(), 0);
                if (cp != null) {
                    try { 
                        // Move mouse to tree and click with validation in one step
                        boolean clickSuccess = context.input.moveAndClickWithValidation(new java.awt.Point(cp.getX(), cp.getY()), "Chop");
                        if (!clickSuccess) {
                            context.logger.warn("[Task] Click validation failed - target may not have chop action");
                            TreeDiscovery.markDepleted(best.getWorldLocation());
                            context.setBusyForMs(100);
                            return;
                        }
                        context.logger.info("[Task] Successfully clicked on tree at canvas point: (" + cp.getX() + "," + cp.getY() + ") with validation");
                        
                        // Set busy immediately to prevent overlapping actions
                        context.setBusyForMs(200); // Reduced from 500ms
                        
                        // Mark the tree as depleted since we successfully clicked it
                        TreeDiscovery.markDepleted(best.getWorldLocation());
                        context.logger.info("[Task] Marked tree as depleted: " + best.getWorldLocation());
                        
                        // Log current animation for debugging
                        Player player = context.client.getLocalPlayer();
                        if (player != null) {
                            int currentAnim = player.getAnimation();
                            context.logger.info("[Task] Current player animation: " + currentAnim + " (woodcutting: " + context.isWoodcuttingAnim() + ")");
                            
                            // Also log if we have an axe equipped
                            try {
                                // Check the actual equipment inventory, not just the visual appearance
                                ItemContainer equipment = context.client.getItemContainer(InventoryID.WORN);
                                boolean hasAxe = false;
                                if (equipment != null) {
                                    // Check weapon slot (index 3) for axes
                                    Item weapon = equipment.getItem(EquipmentInventorySlot.WEAPON.getSlotIdx());
                                    if (weapon != null && weapon.getId() > 0) {
                                        String itemName = context.client.getItemDefinition(weapon.getId()).getName();
                                        if (itemName != null && itemName.toLowerCase().contains("axe")) {
                                            hasAxe = true;
                                            context.logger.info("[Task] Found axe in weapon slot: " + itemName);
                                        }
                                    }
                                    
                                    // Also check if we have any axe in inventory as fallback
                                    if (!hasAxe) {
                                        ItemContainer inventory = context.client.getItemContainer(InventoryID.INV);
                                        if (inventory != null) {
                                            for (Item item : inventory.getItems()) {
                                                if (item != null && item.getId() > 0) {
                                                    String itemName = context.client.getItemDefinition(item.getId()).getName();
                                                    if (itemName != null && itemName.toLowerCase().contains("axe")) {
                                                        hasAxe = true;
                                                        context.logger.info("[Task] Found axe in inventory: " + itemName);
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                if (!hasAxe) {
                                    context.logger.warn("[Task] No axe found in equipment or inventory - this might be why chopping isn't working!");
                                }
                            } catch (Exception e) {
                                context.logger.warn("[Task] Could not check equipment: " + e.getMessage());
                            }
                            
                            // If we're not woodcutting after the interaction, mark as depleted and move on
                            if (!context.isWoodcuttingAnim()) {
                                context.logger.info("[Task] Not woodcutting after interaction, but continuing anyway - axe check passed");
                                // Don't mark as depleted immediately - give it a chance to start woodcutting
                                context.setBusyForMs(50); // Reduced from 100ms
                            } else {
                                context.logger.info("[Task] Successfully started woodcutting!");
                                context.setBusyForMs(500); // Reduced from 1000ms - only longer delay when actually woodcutting
                            }
                        }
                        
                    } catch (Exception e) {
                        context.logger.warn("[Task] Failed to move mouse or click tree: " + e.getMessage());
                        TreeDiscovery.markDepleted(best.getWorldLocation());
                        context.setBusyForMs(100); // Reduced from 500ms
                    }
                } else {
                    context.logger.warn("[Task] Could not project tree to canvas");
                    TreeDiscovery.markDepleted(best.getWorldLocation());
                    context.setBusyForMs(50); // Reduced from 200ms
                }
                return;
            } else {
                // No chop action found - mark as depleted and skip immediately
                context.logger.warn("[Task] Tree at " + best.getWorldLocation() + " has no chop action, marking as depleted");
                TreeDiscovery.markDepleted(best.getWorldLocation());
                context.setBusyForMs(50); // Reduced from 200ms
                return;
            }
        } else {
            context.logger.info("[Task] No chop-able tree found; will navigate toward tree hotspot");
            // Fallback: trigger a navigation step toward nearest discovered tree if available
            net.runelite.api.coords.WorldPoint myWp = context.client.getLocalPlayer() != null ? context.client.getLocalPlayer().getWorldLocation() : null;
            if (myWp != null) {
                net.runelite.api.coords.WorldPoint nearest = TreeDiscovery.getNearestDiscoveredTree(myWp);
                if (nearest != null) {
                    WorldPathing.clickStepToward(context, nearest, 6);
                    context.setBusyForMs(500); // Reduced from 2000ms
                }
            }
        }

        // Recovery: if we've spammed invocations without entering woodcutting, adjust camera or step
        if (!context.isWoodcuttingAnim()) {
            long now = System.currentTimeMillis();
            if (now - lastInvokeMs < 4500L) {
                recentInvokeFailures++;
            } else {
                recentInvokeFailures = 0;
            }
            if (recentInvokeFailures >= 3) {
                context.logger.info("[Task] Chop attempts failing; sweeping camera and nudging position");
                CameraHelper.sweepYawSmall(context, 12);
                net.runelite.api.coords.WorldPoint meWp = context.client.getLocalPlayer() != null ? context.client.getLocalPlayer().getWorldLocation() : null;
                if (meWp != null) {
                    WorldPathing.clickStepToward(context, new net.runelite.api.coords.WorldPoint(meWp.getX()+1, meWp.getY(), meWp.getPlane()), 4);
                    context.setBusyForMs(200); // Reduced from 500ms
                }
                recentInvokeFailures = 0;
            }
        }
    }
}


