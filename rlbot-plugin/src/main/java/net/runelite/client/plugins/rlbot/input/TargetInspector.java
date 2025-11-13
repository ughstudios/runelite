package net.runelite.client.plugins.rlbot.input;

import java.awt.Point;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.NPC;
import net.runelite.api.NPCComposition;
import net.runelite.api.ObjectComposition;
import net.runelite.api.Perspective;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * Scene/NPC/object inspection utilities for validating click targets.
 */
final class TargetInspector
{
    private final RLBotLogger logger;
    private final Client client;

    TargetInspector(RLBotLogger logger, Client client)
    {
        this.logger = logger;
        this.client = client;
    }

    private String actionKey(String expectedAction)
    {
        if (expectedAction == null) { return ""; }
        String e = expectedAction.toLowerCase();
        if (e.contains("chop")) return "chop";
        if (e.contains("cut")) return "cut";
        if (e.contains("bank")) return "bank";
        if (e.contains("open")) return "open";
        if (e.contains("deposit")) return "deposit";
        if (e.contains("use")) return "use";
        return e;
    }

    boolean validateTargetAtPoint(Point canvasPoint, String expectedAction)
    {
        try
        {
            final String key = actionKey(expectedAction);
            final int plane = client.getPlane();
            Scene scene = client.getScene();
            if (scene == null) { return false; }
            Tile[][] tiles = scene.getTiles()[plane];
            if (tiles == null) { return false; }

            // game objects via convex hull
            for (Tile[] col : tiles)
            {
                if (col == null) { continue; }
                for (Tile tile : col)
                {
                    if (tile == null) { continue; }
                    for (GameObject go : tile.getGameObjects())
                    {
                        if (go == null) { continue; }
                        java.awt.Shape hull = go.getConvexHull();
                        if (hull == null || !hull.contains(canvasPoint)) { continue; }
                        ObjectComposition comp = client.getObjectDefinition(go.getId());
                        if (comp == null) { continue; }
                        String[] actions = comp.getActions();
                        if (actions == null) { continue; }
                        for (String a : actions)
                        {
                            if (a != null && a.toLowerCase().contains(key))
                            {
                                return true;
                            }
                        }
                    }
                }
            }

            // NPC convex/projection check
            for (NPC npc : client.getNpcs())
            {
                if (npc == null) { continue; }
                java.awt.Shape npcHull = npc.getConvexHull();
                if (npcHull != null && npcHull.contains(canvasPoint))
                {
                    NPCComposition nc = npc.getTransformedComposition();
                    if (nc != null)
                    {
                        String[] actions = nc.getActions();
                        if (actions != null)
                        {
                            for (String a : actions)
                            {
                                if (a != null && a.toLowerCase().contains(key))
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                else
                {
                    net.runelite.api.Point proj = Perspective.localToCanvas(client, npc.getLocalLocation(), client.getPlane(), npc.getLogicalHeight());
                    if (proj == null) { continue; }
                    if (Math.abs(proj.getX() - canvasPoint.x) <= 10 && Math.abs(proj.getY() - canvasPoint.y) <= 10)
                    {
                        NPCComposition nc = npc.getTransformedComposition();
                        if (nc == null) { continue; }
                        String[] actions = nc.getActions();
                        if (actions == null) { continue; }
                        for (String a : actions)
                        {
                            if (a != null && a.toLowerCase().contains(key))
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }
        catch (Exception e)
        {
            logger.warn("[TargetInspector] Error validating target: " + e.getMessage());
            return false;
        }
    }

    String describeTargetAt(Point canvasPoint)
    {
        try
        {
            final int plane = client.getPlane();
            Scene scene = client.getScene();
            if (scene == null) { return "No scene"; }
            Tile[][] tiles = scene.getTiles()[plane];
            if (tiles == null) { return "No tiles for plane"; }

            StringBuilder sb = new StringBuilder();
            for (Tile[] col : tiles)
            {
                if (col == null) { continue; }
                for (Tile t : col)
                {
                    if (t == null) { continue; }
                    for (GameObject go : t.getGameObjects())
                    {
                        if (go == null) { continue; }
                        net.runelite.api.coords.LocalPoint lp = go.getLocalLocation();
                        if (lp == null) { continue; }
                        net.runelite.api.Point proj = Perspective.localToCanvas(client, lp, plane);
                        if (proj == null) { continue; }
                        if (Math.abs(proj.getX() - canvasPoint.x) <= 10 && Math.abs(proj.getY() - canvasPoint.y) <= 10)
                        {
                            ObjectComposition comp = client.getObjectDefinition(go.getId());
                            if (comp == null) { continue; }
                            sb.append("GameObject: ").append(comp.getName()).append(" (ID: ").append(go.getId()).append(")");
                            String[] actions = comp.getActions();
                            if (actions != null)
                            {
                                sb.append(" Actions: [");
                                boolean first = true;
                                for (String a : actions)
                                {
                                    if (a != null)
                                    {
                                        if (!first) sb.append(", ");
                                        sb.append(a);
                                        first = false;
                                    }
                                }
                                sb.append("]");
                            }
                            sb.append("; ");
                        }
                    }
                }
            }

            for (NPC npc : client.getNpcs())
            {
                if (npc == null) { continue; }
                net.runelite.api.Point npcP = Perspective.localToCanvas(client, npc.getLocalLocation(), client.getPlane(), npc.getLogicalHeight());
                if (npcP == null) { continue; }
                if (Math.abs(npcP.getX() - canvasPoint.x) <= 10 && Math.abs(npcP.getY() - canvasPoint.y) <= 10)
                {
                    NPCComposition nc = npc.getTransformedComposition();
                    if (nc != null)
                    {
                        sb.append("NPC: ").append(nc.getName());
                        String[] actions = nc.getActions();
                        if (actions != null)
                        {
                            sb.append(" Actions: [");
                            boolean first = true;
                            for (String a : actions)
                            {
                                if (a != null)
                                {
                                    if (!first) sb.append(", ");
                                    sb.append(a);
                                    first = false;
                                }
                            }
                            sb.append("]");
                        }
                        sb.append("; ");
                    }
                }
            }
            String s = sb.toString();
            return s.isEmpty() ? null : s;
        }
        catch (Exception e)
        {
            return "Inspector error: " + e.getMessage();
        }
    }

    GameObject findObjectHullMatching(Point canvasPoint, String expectedAction)
    {
        try
        {
            final String key = actionKey(expectedAction);
            final int plane = client.getPlane();
            Scene scene = client.getScene();
            if (scene == null) { return null; }
            Tile[][] tiles = scene.getTiles()[plane];
            if (tiles == null) { return null; }
            GameObject best = null;
            double bestDist = Double.MAX_VALUE;
            for (Tile[] col : tiles)
            {
                if (col == null) { continue; }
                for (Tile tile : col)
                {
                    if (tile == null) { continue; }
                    for (GameObject go : tile.getGameObjects())
                    {
                        if (go == null) { continue; }
                        java.awt.Shape hull = go.getConvexHull();
                        if (hull == null || !hull.contains(canvasPoint)) { continue; }
                        ObjectComposition comp = client.getObjectDefinition(go.getId());
                        if (comp == null || comp.getActions() == null) { continue; }
                        boolean match = false;
                        for (String a : comp.getActions())
                        {
                            if (a != null && a.toLowerCase().contains(key)) { match = true; break; }
                        }
                        if (!match) { continue; }
                        net.runelite.api.coords.LocalPoint lp = go.getLocalLocation();
                        if (lp == null) { continue; }
                        double dx = lp.getX() - client.getCameraX();
                        double dy = lp.getY() - client.getCameraY();
                        double dist = Math.hypot(dx, dy);
                        if (dist < bestDist) { bestDist = dist; best = go; }
                    }
                }
            }
            return best;
        }
        catch (Exception ignored)
        {
            return null;
        }
    }
}
