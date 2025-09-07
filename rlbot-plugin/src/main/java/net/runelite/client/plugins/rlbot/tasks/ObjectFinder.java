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
                            
                            // Check if this object is banned
                            if (net.runelite.client.plugins.rlbot.config.RLBotConfigManager.isObjectBanned(go.getId())) {
                                ctx.logger.info("[ObjectFinder] Skipping banned object: id=" + go.getId() + ", name='" + comp.getName() + "'");
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
                                // Prefer objects whose convex hull is on-screen
                                java.awt.Point proj = projectToCanvas(ctx, go);
                                if (proj == null) continue;
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
                            // FIXED: More precise tree name matching to prevent "tree" from matching "yew tree"
                            if (s.equals("tree")) {
                                // For generic "tree", only match if it's exactly "tree" or "dead tree", not "yew tree", "oak tree", etc.
                                if (lower.equals("tree") || lower.equals("dead tree")) {
                                    nameOk = true;
                                    break;
                                }
                            } else {
                                // For specific trees like "oak", "willow", "yew", match if the name contains the substring
                                if (lower.contains(s)) { 
                                    nameOk = true; 
                                    break; 
                                }
                            }
                        }
                        if (!nameOk) continue;
                        
                        // Check if this object is banned
                        if (net.runelite.client.plugins.rlbot.config.RLBotConfigManager.isObjectBanned(go.getId())) {
                            ctx.logger.info("[ObjectFinder] Skipping banned object: id=" + go.getId() + ", name='" + comp.getName() + "'");
                            continue;
                        }
                        
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

    /**
     * Find nearest bank-interactable object (booth, chest, deposit box).
     * Accepts actions such as "Bank", "Use", "Open", "Deposit".
     */
    public static GameObject findNearestBankInteractable(TaskContext ctx) {
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
                        // Name filter
                        String name = comp.getName();
                        if (name == null) continue;
                        String lower = name.toLowerCase();
                        boolean looksBank = lower.contains("bank booth") || lower.contains("bank chest") || lower.contains("bank deposit") || lower.equals("bank") || lower.contains("deposit box");
                        if (!looksBank) continue;
                        if (isBankTable(comp)) continue;
                        // Action filter
                        boolean actionOk = false;
                        String[] actions = comp.getActions();
                        if (actions != null) {
                            for (String a : actions) {
                                if (a == null) continue;
                                String al = a.toLowerCase();
                                if (al.contains("bank") || al.contains("use") || al.contains("open") || al.contains("deposit")) { actionOk = true; break; }
                            }
                        }
                        if (!actionOk) continue;
                        // Projection/viewport gate
                        java.awt.Point proj = projectToCanvas(ctx, go);
                        if (proj == null) continue;
                        // Distance gate & blacklist
                        WorldPoint wp = go.getWorldLocation();
                        if (wp != null && BankDiscovery.isBlacklisted(wp)) continue;
                        int d = tile.getWorldLocation().distanceTo(myWp);
                        if (d >= 0 && d < bestDist) { bestDist = d; best = go; }
                    }
                }
            }
        }
        return best;
    }

    public static java.awt.Point projectToCanvas(TaskContext ctx, GameObject go) {
        return projectToCanvasWithDiagnostics(ctx, go, false);
    }
    
    /**
     * Project a GameObject to canvas coordinates with optional detailed logging.
     * 
     * @param ctx Task context
     * @param go GameObject to project
     * @param enableDiagnostics If true, logs detailed diagnostic information
     * @return Canvas point or null if projection fails
     */
    public static java.awt.Point projectToCanvasWithDiagnostics(TaskContext ctx, GameObject go, boolean enableDiagnostics) {
        if (go == null) {
            if (enableDiagnostics) {
                ctx.logger.warn("[PROJECTION] Failed: GameObject is null");
            }
            return null;
        }
        
        LocalPoint lp = go.getLocalLocation();
        if (lp == null) {
            if (enableDiagnostics) {
                ctx.logger.warn("[PROJECTION] Failed: LocalPoint is null for object at " + go.getWorldLocation());
            }
            return null;
        }
        
        if (enableDiagnostics) {
            ctx.logger.info("[PROJECTION] Object world location: " + go.getWorldLocation());
            ctx.logger.info("[PROJECTION] Object local location: " + lp);
        }
        
        net.runelite.api.Point p = Perspective.localToCanvas(ctx.client, lp, ctx.client.getPlane());
        if (p == null) {
            if (enableDiagnostics) {
                ctx.logger.warn("[PROJECTION] Failed: Perspective.localToCanvas returned null");
                ctx.logger.info("[PROJECTION] This usually means the object is:");
                ctx.logger.info("[PROJECTION] - Behind the camera");
                ctx.logger.info("[PROJECTION] - Too far away to render");
                ctx.logger.info("[PROJECTION] - At an extreme angle outside the camera's field of view");
                
                // Log camera information
                try {
                    ctx.logger.info("[PROJECTION] Camera yaw: " + ctx.client.getCameraYawTarget());
                    ctx.logger.info("[PROJECTION] Camera pitch: " + ctx.client.getCameraPitchTarget());
                    ctx.logger.info("[PROJECTION] Camera zoom: " + ctx.client.get3dZoom());
                } catch (Exception e) {
                    ctx.logger.warn("[PROJECTION] Could not get camera info: " + e.getMessage());
                }
            }
            return null;
        }
        
        int cx = p.getX(), cy = p.getY();
        
        // Get viewport bounds
        int vx = ctx.client.getViewportXOffset();
        int vy = ctx.client.getViewportYOffset();
        int vw = ctx.client.getViewportWidth();
        int vh = ctx.client.getViewportHeight();
        
        boolean inViewport = cx >= vx && cy >= vy && cx < (vx + vw) && cy < (vy + vh);
        
        if (enableDiagnostics) {
            ctx.logger.info("[PROJECTION] Projected canvas point: (" + cx + ", " + cy + ")");
            ctx.logger.info("[PROJECTION] Viewport bounds: x=" + vx + ", y=" + vy + ", w=" + vw + ", h=" + vh);
            ctx.logger.info("[PROJECTION] Viewport range: x=[" + vx + "-" + (vx + vw) + "], y=[" + vy + "-" + (vy + vh) + "]");
            ctx.logger.info("[PROJECTION] Point in viewport: " + inViewport);
            
            if (!inViewport) {
                ctx.logger.warn("[PROJECTION] Failed: Point is outside viewport bounds");
                if (cx < vx) ctx.logger.info("[PROJECTION] - Point is too far LEFT (cx=" + cx + " < vx=" + vx + ")");
                if (cx >= vx + vw) ctx.logger.info("[PROJECTION] - Point is too far RIGHT (cx=" + cx + " >= " + (vx + vw) + ")");
                if (cy < vy) ctx.logger.info("[PROJECTION] - Point is too far UP (cy=" + cy + " < vy=" + vy + ")");
                if (cy >= vy + vh) ctx.logger.info("[PROJECTION] - Point is too far DOWN (cy=" + cy + " >= " + (vy + vh) + ")");
            } else {
                ctx.logger.info("[PROJECTION] Success: Point is within viewport bounds");
            }
        }
        
        if (!inViewport) return null;
        return new java.awt.Point(cx, cy);
    }

    /**
     * Choose a stable clickable point for a GameObject, preferring its convex hull to avoid
     * targeting NPCs in front of the object (e.g., bankers). Falls back to tile center.
     */
    public static java.awt.Point projectToClickablePoint(TaskContext ctx, GameObject go) {
        if (go == null) return null;
        Shape hull = go.getConvexHull();
        if (hull != null) {
            Rectangle b = hull.getBounds();
            if (b != null && b.width > 1 && b.height > 1) {
                ctx.logger.info("[PROJECTION] Convex hull bounds: x=" + b.x + ", y=" + b.y + ", w=" + b.width + ", h=" + b.height);
                // Aim toward the upper-middle of the hull to reduce NPC occlusion
                int x = b.x + (b.width / 2);
                int y = b.y + Math.max(1, b.height / 3);
                java.awt.Point candidate = new java.awt.Point(x, y);
                ctx.logger.info("[PROJECTION] Candidate click point: (" + candidate.x + "," + candidate.y + ")");
                if (hull.contains(candidate)) {
                    // Also ensure within viewport
                    int vx = ctx.client.getViewportXOffset();
                    int vy = ctx.client.getViewportYOffset();
                    int vw = ctx.client.getViewportWidth();
                    int vh = ctx.client.getViewportHeight();
                    if (candidate.x >= vx && candidate.y >= vy && candidate.x < (vx + vw) && candidate.y < (vy + vh)) {
                        ctx.logger.info("[PROJECTION] Using candidate point (within viewport and hull)");
                        return candidate;
                    } else {
                        ctx.logger.warn("[PROJECTION] Candidate point outside viewport");
                    }
                } else {
                    ctx.logger.warn("[PROJECTION] Candidate point not contained in hull");
                }
                // Fallback to hull center
                java.awt.Point center = new java.awt.Point(b.x + b.width / 2, b.y + b.height / 2);
                ctx.logger.info("[PROJECTION] Using hull center fallback: (" + center.x + "," + center.y + ")");
                return center;
            } else {
                ctx.logger.warn("[PROJECTION] Hull bounds invalid: " + (b == null ? "null" : "w=" + b.width + " h=" + b.height));
            }
        } else {
            ctx.logger.warn("[PROJECTION] No convex hull available, falling back to simple projection");
        }
        return projectToCanvas(ctx, go);
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


