package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.ObjectComposition;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.api.TileObject;
import net.runelite.api.coords.WorldPoint;
import net.runelite.client.plugins.rlbot.config.RLBotConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for dynamically discovering and persisting bank locations.
 * Scans the game world for bank objects and saves their locations to a consolidated config file.
 */
public class BankDiscovery {
    private static final Logger logger = LoggerFactory.getLogger(BankDiscovery.class);
    private static final String[] BANK_NAMES = {"bank booth", "bank chest", "bank", "deposit box", "bank deposit box"};
    
    private static volatile WorldPoint lastTargetedBank = null;
    
    /**
     * Scan the current scene for bank objects and add any new discoveries
     */
    public static void scanAndDiscoverBanks(TaskContext ctx) {
        try {
            Scene scene = ctx.client.getScene();
            if (scene == null) return;
            
            boolean foundNewBanks = false;
            Tile[][][] tiles = scene.getTiles();
            
            for (int z = 0; z < tiles.length; z++) {
                for (int x = 0; x < tiles[z].length; x++) {
                    for (int y = 0; y < tiles[z][x].length; y++) {
                        Tile tile = tiles[z][x][y];
                        if (tile == null) continue;
                        
                        for (TileObject to : tile.getGameObjects()) {
                            if (to == null) continue;
                            
                            ObjectComposition comp = ctx.client.getObjectDefinition(to.getId());
                            if (comp == null) continue;
                            
                            String name = comp.getName();
                            if (name == null) continue;
                            
                            String lowerName = name.toLowerCase();
                            boolean isBank = false;
                            for (String bankName : BANK_NAMES) {
                                if (lowerName.contains(bankName)) {
                                    isBank = true;
                                    break;
                                }
                            }
                            
                            if (isBank) {
                                // Skip bank tables (decorative objects)
                                if (isBankTable(comp)) {
                                    continue;
                                }
                                
                                WorldPoint bankLocation = to.getWorldLocation();
                                if (!RLBotConfigManager.hasBank(bankLocation)) {
                                    RLBotConfigManager.addBank(bankLocation, name);
                                    foundNewBanks = true;
                                    logger.info("[BankDiscovery] Found new bank: {} at {}", name, bankLocation);
                                }
                            }
                        }
                    }
                }
            }
            
            if (foundNewBanks) {
                logger.debug("[BankDiscovery] Found new banks during scan");
            }
        } catch (Exception e) {
            logger.warn("[BankDiscovery] Error scanning for banks: {}", e.getMessage());
        }
    }
    
    /**
     * Get all discovered bank locations (excluding blacklisted ones)
     */
    public static List<WorldPoint> getDiscoveredBanks() {
        List<RLBotConfigManager.BankLocation> banks = RLBotConfigManager.getBanks();
        List<WorldPoint> result = new ArrayList<>();
        
        for (RLBotConfigManager.BankLocation bank : banks) {
            WorldPoint wp = bank.toWorldPoint();
            if (!RLBotConfigManager.isBankBlacklisted(wp)) {
                result.add(wp);
            }
        }
        
        return result;
    }
    
    /**
     * Get the nearest discovered bank to a given location
     */
    public static WorldPoint getNearestDiscoveredBank(WorldPoint from) {
        if (from == null) return null;
        
        List<WorldPoint> banks = getDiscoveredBanks();
        if (banks.isEmpty()) return null;
        
        WorldPoint nearest = null;
        int nearestDist = Integer.MAX_VALUE;
        
        for (WorldPoint bank : banks) {
            int dist = from.distanceTo(bank);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearest = bank;
            }
        }
        
        return nearest;
    }
    
    /**
     * Add a manually discovered bank location
     */
    public static void addDiscoveredBank(WorldPoint location, String name) {
        if (location != null && !RLBotConfigManager.hasBank(location)) {
            RLBotConfigManager.addBank(location, name);
            logger.info("[BankDiscovery] Manually added bank: {} at {}", name, location);
        }
    }

    public static void setLastTargetedBank(WorldPoint location) {
        lastTargetedBank = location;
    }

    public static void blacklistLastTargetedBank() {
        if (lastTargetedBank != null) {
            RLBotConfigManager.blacklistBank(lastTargetedBank);
            logger.warn("[BankDiscovery] Blacklisted unreachable bank at {}", lastTargetedBank);
        }
    }

    public static boolean isBlacklisted(WorldPoint location) {
        return location != null && RLBotConfigManager.isBankBlacklisted(location);
    }
    
    /**
     * Remove a bank from the blacklist
     */
    public static void removeFromBlacklist(WorldPoint location) {
        if (location != null) {
            RLBotConfigManager.removeBankFromBlacklist(location);
            logger.info("[BankDiscovery] Removed bank from blacklist: {}", location);
        }
    }
    
    /**
     * Check if an object is a bank table (decorative object that looks like a bank but has no banking functionality)
     */
    private static boolean isBankTable(ObjectComposition comp) {
        if (comp == null || comp.getName() == null) return false;
        
        String name = comp.getName().toLowerCase();
        
        // Bank tables are typically named things like "Bank table", "Table", etc.
        // but don't have actual banking actions
        if (name.contains("table") && name.contains("bank")) {
            return true;
        }
        
        // Also check for common bank table object IDs
        int id = comp.getId();
        // These are common bank table object IDs - add more as needed
        if (id == 34810) { // Common bank table ID that has "Bank" action but is decorative
            return true;
        }
        
        // Check if the name is null or empty (often indicates decorative objects)
        if (name.equals("null") || name.isEmpty()) {
            // Additional check: if it has a "Bank" action but the name is null/empty, 
            // it's likely a decorative bank table
            if (comp.getActions() != null) {
                for (String action : comp.getActions()) {
                    if (action != null && action.equals("Bank")) {
                        return true; // Null/empty name with Bank action = likely bank table
                    }
                }
            }
        }
        
        return false;
    }
}
