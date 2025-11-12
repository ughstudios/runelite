package net.runelite.client.plugins.rlbot.input;

import java.awt.Point;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * UI and geometry occlusion checks and resolution strategies.
 */
final class OcclusionResolver
{
    private final RLBotLogger logger;
    private final Client client;
    private final ChatboxCollisionHandler chatbox;
    private final CameraController camera;

    OcclusionResolver(RLBotLogger logger, Client client, ChatboxCollisionHandler chatbox, CameraController camera)
    {
        this.logger = logger;
        this.client = client;
        this.chatbox = chatbox;
        this.camera = camera;
    }

    boolean isPointInViewport(Point p)
    {
        try
        {
            int vx = client.getViewportXOffset();
            int vy = client.getViewportYOffset();
            int vw = client.getViewportWidth();
            int vh = client.getViewportHeight();
            if (vw <= 0 || vh <= 0) { return true; }
            return p.x >= vx && p.x < vx + vw && p.y >= vy && p.y < vy + vh;
        }
        catch (Exception e)
        {
            return true;
        }
    }

    boolean isOccludedByUI(Point canvasPoint)
    {
        if (!isPointInViewport(canvasPoint)) { return true; }
        if (chatbox != null)
        {
            try
            {
                if (chatbox.isInChatArea(canvasPoint)) { return true; }
            }
            catch (IllegalStateException e)
            {
                logger.error("[Occlusion] Chat widget error: " + e.getMessage());
                return false;
            }
        }
        return false;
    }

    boolean revealPointByCameraUI(Point target)
    {
        if (!isOccludedByUI(target)) { return true; }
        if (chatbox != null)
        {
            try
            {
                if (chatbox.isInChatArea(target))
                {
                    chatbox.handleChatboxCollision(target, null, "reveal");
                    return !isOccludedByUI(target);
                }
            }
            catch (IllegalStateException e)
            {
                logger.error("[Occlusion] Chat widget error: " + e.getMessage());
            }
        }
        // fallback simple camera adjustment
        try
        {
            int vy = client.getViewportYOffset();
            int vh = client.getViewportHeight();
            if (vh > 0 && target.y > vy + vh - 24)
            {
                camera.zoomOutSmall();
            }
            else
            {
                int dx = 140;
                int dy = (target.y < vy + vh / 2) ? -20 : 20;
                camera.rotateCameraSafe(dx, dy);
            }
        }
        catch (Exception ignored) {}
        return false;
    }

    boolean isOccludedByGeometry(Point canvasPoint, GameObject target)
    {
        return isOccludedByGeometryWithDiagnostics(canvasPoint, target, false);
    }

    boolean isOccludedByGeometryWithDiagnostics(Point canvasPoint, GameObject target, boolean diagnostics)
    {
        try
        {
            if (target == null) { return false; }
            net.runelite.api.coords.LocalPoint tlp = target.getLocalLocation();
            if (tlp == null) { return false; }
            double tdx = tlp.getX() - client.getCameraX();
            double tdy = tlp.getY() - client.getCameraY();
            double tdist = Math.hypot(tdx, tdy);

            final int plane = client.getPlane();
            Scene scene = client.getScene();
            if (scene == null) { return false; }
            Tile[][] tiles = scene.getTiles()[plane];
            if (tiles == null) { return false; }

            for (Tile[] col : tiles)
            {
                if (col == null) { continue; }
                for (Tile tile : col)
                {
                    if (tile == null) { continue; }
                    for (GameObject go : tile.getGameObjects())
                    {
                        if (go == null || go == target) { continue; }
                        java.awt.Shape hull = go.getConvexHull();
                        if (hull == null || !hull.contains(canvasPoint)) { continue; }
                        net.runelite.api.coords.LocalPoint lp = go.getLocalLocation();
                        if (lp == null) { continue; }
                        double dx = lp.getX() - client.getCameraX();
                        double dy = lp.getY() - client.getCameraY();
                        double dist = Math.hypot(dx, dy);
                        if (dist + 1.0 < tdist)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
        catch (Exception ignored)
        {
            return false;
        }
    }

    boolean revealPointByCameraGeometry(Point targetPoint, GameObject target, int attempts)
    {
        if (target == null) { return false; }
        for (int i = 0; i < Math.max(1, attempts); i++)
        {
            if (!isOccludedByGeometry(targetPoint, target)) { return true; }
            camera.rotateCameraSafe(120, 0);
        }
        return false;
    }
}

