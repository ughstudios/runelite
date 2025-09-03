package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.api.Player;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.MenuAction;

final class WorldPathing {
    private WorldPathing() {}

    /**
     * Clicks a world tile a few tiles towards target if its canvas projection is on screen.
     * Returns true if a click was performed.
     */
    static boolean clickStepToward(TaskContext ctx, WorldPoint target, int stepTiles) {
        try {
            Client client = ctx.client;
            Player me = client.getLocalPlayer();
            if (me == null || target == null) return false;
            WorldPoint meWp = me.getWorldLocation();
            int dx = target.getX() - meWp.getX();
            int dy = target.getY() - meWp.getY();
            double len = Math.hypot(dx, dy);
            if (len < 1e-3) return false;

            // If validation-only, probe actual target tile projection and viewport inclusion
            if (stepTiles <= 0) {
                LocalPoint lpValidate = LocalPoint.fromWorld(client, target);
                if (lpValidate == null) return false;
                net.runelite.api.Point pv = net.runelite.api.Perspective.localToCanvas(client, lpValidate, client.getPlane());
                if (pv == null) return false;
                int vx = client.getViewportXOffset();
                int vy = client.getViewportYOffset();
                int vw = client.getViewportWidth();
                int vh = client.getViewportHeight();
                return pv.getX() >= vx && pv.getY() >= vy && pv.getX() < (vx + vw) && pv.getY() < (vy + vh);
            }

            double ux = dx / len;
            double uy = dy / len;
            int stepX = meWp.getX() + (int)Math.round(ux * Math.max(0, stepTiles));
            int stepY = meWp.getY() + (int)Math.round(uy * Math.max(0, stepTiles));
            WorldPoint step = new WorldPoint(stepX, stepY, meWp.getPlane());

            LocalPoint lp = LocalPoint.fromWorld(client, step);
            if (lp == null) return false;
            ctx.logger.info("[WorldPath] Step toward world (" + step.getX() + "," + step.getY() + ")");
            // Prefer direct canvas click if projection is inside viewport
            net.runelite.api.Point p = net.runelite.api.Perspective.localToCanvas(client, lp, client.getPlane());
            if (p != null) {
                int vx = client.getViewportXOffset();
                int vy = client.getViewportYOffset();
                int vw = client.getViewportWidth();
                int vh = client.getViewportHeight();
                if (p.getX() >= vx && p.getY() >= vy && p.getX() < (vx + vw) && p.getY() < (vy + vh)) {
                    // Use WALK menuAction to avoid clicking UI overlays
                    int sceneX = lp.getSceneX();
                    int sceneY = lp.getSceneY();
                    ctx.logger.info("[WorldPath] WALK to scene (" + sceneX + "," + sceneY + ") for world (" + step.getX() + "," + step.getY() + ") canvas(" + p.getX() + "," + p.getY() + ")");
                    
                    // Move mouse first, then click
                    try { 
                        ctx.input.smoothMouseMove(new java.awt.Point(p.getX(), p.getY())); 
                        ctx.logger.info("[WorldPath] Mouse moved to canvas point, now executing walk click");
                        
                        // Set busy for mouse movement delay
                        ctx.setBusyForMs(50);
                        
                        // Perform the actual click
                        ctx.input.click(); // Perform the actual click
                        ctx.logger.info("[WorldPath] Click executed at canvas point (" + p.getX() + "," + p.getY() + ")");
                        
                    } catch (Exception e) {
                        ctx.logger.error("[WorldPath] Mouse move/click failed: " + e.getMessage());
                        
                        // Fallback to menuAction if direct click fails
                        ctx.clientThread.invoke(() -> {
                            try {
                                client.menuAction(sceneX, sceneY, MenuAction.WALK, 0, 0, "Walk here", "");
                                ctx.logger.info("[WorldPath] Fallback WALK menuAction executed");
                            } catch (Exception e2) {
                                ctx.logger.error("[WorldPath] Fallback WALK menuAction error: " + e2.getMessage());
                            }
                        });
                    }
                    
                    ctx.markMenuWalkClick();
                    ctx.setBusyForMs(800); // Increased timeout
                    return true;
                } else {
                    ctx.logger.info("[WorldPath] Projection (" + p.getX() + "," + p.getY() + ") outside viewport bounds — returning false to allow caller to choose minimap");
                    return false;
                }
            } else {
                ctx.logger.info("[WorldPath] Projection is null for step world (" + step.getX() + "," + step.getY() + ") — returning false to allow caller to choose minimap");
                return false;
            }
            
        } catch (Exception e) {
            ctx.logger.error("[WorldPath] Error clicking world tile: " + e.getMessage());
            return false;
        }
    }
}


