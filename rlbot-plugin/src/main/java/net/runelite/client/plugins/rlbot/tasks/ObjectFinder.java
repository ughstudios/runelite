package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.ObjectComposition;
import net.runelite.api.Player;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.Perspective;
import java.awt.Shape;
import java.awt.Rectangle;

public final class ObjectFinder {
    private ObjectFinder() {}

    public static GameObject findNearestByAction(TaskContext ctx, String requiredAction) {
        if (ctx == null || requiredAction == null) return null;
        Player me = ctx.client.getLocalPlayer();
        if (me == null) return null;
        WorldPoint myWp = me.getWorldLocation();
        if (myWp == null) return null;
        
        GameObject best = null;
        int bestDist = Integer.MAX_VALUE;
        
        Scene scene = ctx.client.getScene();
        if (scene != null) {
            Tile[][][] tiles = scene.getTiles();
            for (int z = 0; z < tiles.length; z++) {
                for (int x = 0; x < tiles[z].length; x++) {
                    for (int y = 0; y < tiles[z][x].length; y++) {
                        Tile tile = tiles[z][x][y];
                        if (tile == null) continue;
                        
                        for (GameObject go : tile.getGameObjects()) {
                            if (go == null) continue;
                            
                            // Check if this object has the required action
                            ObjectComposition comp = ctx.client.getObjectDefinition(go.getId());
                            if (comp == null || comp.getActions() == null) continue;
                            
                            // Skip bank tables - they're decorative and don't have real banking functionality
                            if (isBankTable(comp)) {
                                ctx.logger.info("[ObjectFinder] Skipping bank table: id=" + go.getId() + ", name='" + comp.getName() + "'");
                                continue;
                            }
                            // If we're looking for a bank action, skip blacklisted bank world points entirely
                            if (requiredAction != null && requiredAction.equalsIgnoreCase("Bank")) {
                                WorldPoint wp = go.getWorldLocation();
                                if (wp != null && BankDiscovery.isBlacklisted(wp)) {
                                    ctx.logger.info("[ObjectFinder] Skipping blacklisted bank at " + wp);
                                    continue;
                                }
                            }
                            
                            boolean hasAction = false;
                            for (String action : comp.getActions()) {
                                if (action != null && action.equals(requiredAction)) {
                                    hasAction = true;
                                    break;
                                }
                            }
                            
                            if (hasAction) {
                                ctx.logger.info("[ObjectFinder] Found object with '" + requiredAction + "' action: id=" + go.getId() + ", name='" + comp.getName() + "' at " + tile.getWorldLocation());
                                // Debug: check if this object is actually projectable to canvas
                                java.awt.Point proj = projectToCanvas(ctx, go);
                                if (proj != null) {
                                    ctx.logger.info("[ObjectFinder] Object is projectable to canvas at " + proj);
                                } else {
                                    ctx.logger.info("[ObjectFinder] Object is NOT projectable to canvas");
                                }
                                int d = tile.getWorldLocation().distanceTo(myWp);
                                if (d >= 0 && d < bestDist) { 
                                    bestDist = d; 
                                    best = go; 
                                }
                            }
                        }
                    }
                }
            }
        }
        return best;
    }

    public static GameObject findNearestByNames(TaskContext ctx, String[] nameSubstrings, String requiredAction) {
        Client client = ctx.client;
        Player me = client.getLocalPlayer();
        if (me == null) return null;
        WorldPoint myWp = me.getWorldLocation();
        Scene scene = client.getScene();
        Tile[][][] tiles = scene.getTiles();
        int bestDist = Integer.MAX_VALUE;
        GameObject best = null;
        for (int z = 0; z < tiles.length; z++) {
            for (int x = 0; x < tiles[z].length; x++) {
                for (int y = 0; y < tiles[z][x].length; y++) {
                    Tile tile = tiles[z][x][y];
                    if (tile == null) continue;
                    for (GameObject go : tile.getGameObjects()) {
                        if (go == null) continue;
                        ObjectComposition comp = client.getObjectDefinition(go.getId());
                        if (comp == null) continue;
                        String name = comp.getName();
                        if (name == null) continue;
                        String lower = name.toLowerCase();
                        boolean nameOk = false;
                        for (String s : nameSubstrings) {
                            if (lower.contains(s)) { 
                                nameOk = true; 
                                // Only log when we actually find a valid target, not every object
                                break; 
                            }
                        }
                        if (!nameOk) continue;
                        
                        // Check if this tree is depleted (stump)
                        if (TreeDiscovery.isDepleted(tile.getWorldLocation())) {
                            ctx.logger.info("[ObjectFinder] Skipping depleted tree at " + tile.getWorldLocation());
                            continue;
                        }
                        // If searching for banks, skip blacklisted world points
                        if (requiredAction != null && requiredAction.equalsIgnoreCase("Bank")) {
                            WorldPoint wp = go.getWorldLocation();
                            if (wp != null && BankDiscovery.isBlacklisted(wp)) {
                                ctx.logger.info("[ObjectFinder] Skipping blacklisted bank at " + wp);
                                continue;
                            }
                        }
                        
                        if (requiredAction != null) {
                            boolean ok = false;
                            String[] acts = comp.getActions();
                            if (acts != null) {
                                for (String a : acts) {
                                    if (a != null && (a.equalsIgnoreCase(requiredAction) || a.toLowerCase().contains(requiredAction.toLowerCase()))) { 
                                        ok = true; 
                                        break; 
                                    }
                                }
                            }
                            if (!ok) continue;
                        }
                        int d = tile.getWorldLocation().distanceTo(myWp);
                        if (d >= 0 && d < bestDist) { bestDist = d; best = go; }
                    }
                }
            }
        }
        return best;
    }

    public static java.awt.Point projectToCanvas(TaskContext ctx, GameObject go) {
        if (go == null) return null;
        LocalPoint lp = go.getLocalLocation();
        if (lp == null) return null;
        net.runelite.api.Point p = Perspective.localToCanvas(ctx.client, lp, ctx.client.getPlane());
        if (p == null) return null;
        int cx = p.getX(), cy = p.getY();
        // Use viewport bounds to avoid clicking UI chrome
        int vx = ctx.client.getViewportXOffset();
        int vy = ctx.client.getViewportYOffset();
        int vw = ctx.client.getViewportWidth();
        int vh = ctx.client.getViewportHeight();
        boolean inViewport = cx >= vx && cy >= vy && cx < (vx + vw) && cy < (vy + vh);
        if (!inViewport) return null;
        return new java.awt.Point(cx, cy);
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


