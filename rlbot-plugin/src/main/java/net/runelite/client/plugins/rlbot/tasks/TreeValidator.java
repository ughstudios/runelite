package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.GameObject;
import net.runelite.api.coords.WorldPoint;

/**
 * Validates trees for woodcutting operations.
 */
public class TreeValidator {
    
    /**
     * Validates if a tree can be chopped based on level requirements and actions.
     */
    public static boolean isValidTree(TaskContext context, GameObject tree, int wcLevel) {
        if (tree == null) {
            context.logger.info("[TreeValidator] Tree is null");
            return false;
        }
        
        try {
            net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(tree.getId());
            if (comp == null) {
                context.logger.info("[TreeValidator] No object composition for tree ID: " + tree.getId());
                return false;
            }
            
            String name = comp.getName();
            if (name == null) {
                context.logger.info("[TreeValidator] Tree has no name");
                return false;
            }
            
            if (name.toLowerCase().contains("stump")) {
                context.logger.info("[TreeValidator] Tree is a stump: " + name);
                return false;
            }
            
            if (!TreeDiscovery.isTreeAllowedForLevel(name, wcLevel)) {
                context.logger.info("[TreeValidator] Tree '" + name + "' requires higher level than " + wcLevel);
                return false;
            }
            
            String[] actions = comp.getActions();
            if (actions == null) {
                context.logger.info("[TreeValidator] Tree has no actions");
                return false;
            }
            
            boolean hasChopAction = false;
            for (String action : actions) {
                if (action != null && (action.toLowerCase().contains("chop") || action.toLowerCase().contains("cut"))) {
                    hasChopAction = true;
                    context.logger.info("[TreeValidator] Found valid chop action: " + action);
                    break;
                }
            }
            
            if (!hasChopAction) {
                context.logger.info("[TreeValidator] Tree has no chop action");
                return false;
            }
            
            return true;
            
        } catch (Exception e) {
            context.logger.warn("[TreeValidator] Error validating tree: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Checks if tree is in wilderness when player is not.
     */
    public static boolean isWildernessConflict(TaskContext context, GameObject tree) {
        if (tree == null) return false;
        
        WorldPoint treePos = tree.getWorldLocation();
        WorldPoint playerPos = context.client.getLocalPlayer() != null ? 
            context.client.getLocalPlayer().getWorldLocation() : null;
            
        if (treePos == null || playerPos == null) return false;
        
        boolean treeInWilderness = treePos.getY() > 3523;
        boolean playerInWilderness = playerPos.getY() > 3523;
        
        if (treeInWilderness && !playerInWilderness) {
            context.logger.info("[TreeValidator] Wilderness conflict: tree at " + treePos + ", player at " + playerPos);
            return true;
        }
        
        return false;
    }
    
    /**
     * Checks if player has an axe equipped or in inventory.
     */
    public static boolean hasAxe(TaskContext context) {
        try {
            net.runelite.api.ItemContainer equipment = context.client.getItemContainer(net.runelite.api.gameval.InventoryID.WORN);
            if (equipment != null) {
                net.runelite.api.Item weapon = equipment.getItem(net.runelite.api.EquipmentInventorySlot.WEAPON.getSlotIdx());
                if (weapon != null && weapon.getId() > 0) {
                    String itemName = context.client.getItemDefinition(weapon.getId()).getName();
                    if (itemName != null && itemName.toLowerCase().contains("axe")) {
                        context.logger.info("[TreeValidator] Found axe in weapon slot: " + itemName);
                        return true;
                    }
                }
            }
            
            net.runelite.api.ItemContainer inventory = context.client.getItemContainer(net.runelite.api.gameval.InventoryID.INV);
            if (inventory != null) {
                for (net.runelite.api.Item item : inventory.getItems()) {
                    if (item != null && item.getId() > 0) {
                        String itemName = context.client.getItemDefinition(item.getId()).getName();
                        if (itemName != null && itemName.toLowerCase().contains("axe")) {
                            context.logger.info("[TreeValidator] Found axe in inventory: " + itemName);
                            return true;
                        }
                    }
                }
            }
            
            context.logger.warn("[TreeValidator] No axe found in equipment or inventory");
            return false;
            
        } catch (Exception e) {
            context.logger.warn("[TreeValidator] Error checking for axe: " + e.getMessage());
            return false;
        }
    }
}
