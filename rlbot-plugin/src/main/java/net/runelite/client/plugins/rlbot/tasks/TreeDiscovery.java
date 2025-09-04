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
 * Utility class for dynamically discovering and persisting tree locations.
 * Scans the game world for choppable tree objects and saves their locations to a consolidated config file.
 */
public class TreeDiscovery {
    private static final Logger logger = LoggerFactory.getLogger(TreeDiscovery.class);
    private static final String[] TREE_NAMES = {"tree", "oak", "willow", "maple", "yew", "magic", "teak", "mahogany"};
    private static final String CHOP_ACTION = "chop down";
    
    private static final long DEPLETION_COOLDOWN_MS = 5_000L; // 5s respawn window - short but not permanent
    private static volatile WorldPoint lastTargetedTree = null;
    
    /**
     * Scan the current scene for choppable tree objects and add any new discoveries
     */
    public static void scanAndDiscoverTrees(TaskContext ctx) {
        try {
            Scene scene = ctx.client.getScene();
            if (scene == null) return;
            
            boolean foundNewTrees = false;
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
                            boolean isTree = false;
                            for (String treeName : TREE_NAMES) {
                                if (lowerName.contains(treeName)) {
                                    isTree = true;
                                    break;
                                }
                            }
                            
                            if (isTree) {
                                // Check if it has a chop action
                                boolean hasChopAction = false;
                                String[] actions = comp.getActions();
                                if (actions != null) {
                                    for (String action : actions) {
                                        if (action != null && action.toLowerCase().contains(CHOP_ACTION)) {
                                            hasChopAction = true;
                                            break;
                                        }
                                    }
                                }
                                
                                if (hasChopAction) {
                                    WorldPoint treeLocation = to.getWorldLocation();
                                    if (!RLBotConfigManager.hasTree(treeLocation)) {
                                        RLBotConfigManager.addTree(treeLocation, name);
                                        foundNewTrees = true;
                                    }
                                    // If it has Chop now, clear any prior depletion mark
                                    RLBotConfigManager.markTreeDepleted(treeLocation, 0); // Clear depletion
                                } else {
                                    // Tree found but no chop action - likely a stump, mark as depleted
                                    WorldPoint treeLocation = to.getWorldLocation();
                                    markDepleted(treeLocation);
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
        }
    }
    
    /**
     * Get all discovered tree locations
     */
    public static List<WorldPoint> getDiscoveredTrees() {
        List<RLBotConfigManager.TreeLocation> trees = RLBotConfigManager.getTrees();
        List<WorldPoint> worldPoints = new ArrayList<>();
        for (RLBotConfigManager.TreeLocation tree : trees) {
            worldPoints.add(tree.toWorldPoint());
        }
        return worldPoints;
    }

    /**
     * Get only currently available trees (not marked depleted).
     */
    public static List<WorldPoint> getAvailableTrees() {
        List<WorldPoint> out = new ArrayList<>();
        List<RLBotConfigManager.TreeLocation> trees = RLBotConfigManager.getTrees();
        
        for (RLBotConfigManager.TreeLocation tree : trees) {
            WorldPoint wp = tree.toWorldPoint();
            if (!RLBotConfigManager.isTreeDepleted(wp)) {
                out.add(wp);
            }
        }
        
        return out;
    }
    
    /**
     * Get the nearest discovered tree to a given location
     */
    public static WorldPoint getNearestDiscoveredTree(WorldPoint from) {
        if (from == null) return null;
        List<WorldPoint> avail = getAvailableTrees();
        if (avail.isEmpty()) return null;
        WorldPoint nearest = null;
        int nearestDist = Integer.MAX_VALUE;
        for (WorldPoint tree : avail) {
            int dist = from.distanceTo(tree);
            if (dist < nearestDist) { 
                nearestDist = dist; 
                nearest = tree; 
            }
        }
        return nearest;
    }

    /** Mark a tree location as depleted (temporarily unavailable). */
    public static void markDepleted(WorldPoint location) {
        if (location == null) return;
        long until = System.currentTimeMillis() + DEPLETION_COOLDOWN_MS;
        RLBotConfigManager.markTreeDepleted(location, until);
    }

    /** Track the last tree the agent attempted to interact with. */
    public static void setLastTargetedTree(WorldPoint location) {
        lastTargetedTree = location;
    }

    /** Blacklist the last targeted tree by marking it depleted for an extended window. */
    public static void blacklistLastTargetedTree() {
        if (lastTargetedTree != null) {
            long extended = System.currentTimeMillis() + Math.max(120_000L, DEPLETION_COOLDOWN_MS);
            RLBotConfigManager.markTreeDepleted(lastTargetedTree, extended);
        }
    }

    /** Check whether a location is currently marked as depleted. */
    public static boolean isDepleted(WorldPoint location) {
        if (location == null) return false;
        return RLBotConfigManager.isTreeDepleted(location);
    }
    
    /**
     * Return allowed tree name filters based on a player's woodcutting level.
     * Names are lower-case substrings matched against object names.
     */
    public static String[] allowedTreeNamesForLevel(int woodcuttingLevel) {
        List<String> names = new ArrayList<>();
        // Always allow generic "tree"
        names.add("tree");
        if (woodcuttingLevel >= 15) names.add("oak");
        if (woodcuttingLevel >= 30) names.add("willow");
        if (woodcuttingLevel >= 35) names.add("teak");
        if (woodcuttingLevel >= 45) names.add("maple");
        if (woodcuttingLevel >= 50) names.add("mahogany");
        if (woodcuttingLevel >= 60) names.add("yew");
        if (woodcuttingLevel >= 75) names.add("magic");
        return names.toArray(new String[0]);
    }
    
    /**
     * Add a manually discovered tree location
     */
    public static void addDiscoveredTree(WorldPoint location, String name) {
        if (location != null && !RLBotConfigManager.hasTree(location)) {
            RLBotConfigManager.addTree(location, name);
        }
    }
    
    /**
     * Clean up expired depleted trees
     */
    public static void cleanupExpiredDepletedTrees() {
        RLBotConfigManager.cleanupExpiredDepletedTrees();
    }
}
