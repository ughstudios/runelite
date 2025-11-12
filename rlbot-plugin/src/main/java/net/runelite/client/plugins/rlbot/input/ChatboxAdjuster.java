package net.runelite.client.plugins.rlbot.input;

import java.awt.Point;
import java.util.concurrent.TimeUnit;

import java.awt.Point;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.Perspective;
import net.runelite.api.coords.LocalPoint;
import net.runelite.client.plugins.rlbot.RLBotLogger;

final class ChatboxAdjuster {
    private ChatboxAdjuster() {}

    private enum Strategy { ZOOM_OUT, ROTATE_LEFT, ROTATE_RIGHT, TILT_UP, COMBINED }

    static boolean resolve(Client client, RLBotInputHandler input, RLBotLogger log, int consecutiveCollisions,
                           Point canvasPoint, GameObject targetObject, String expectedAction,
                           java.util.function.Function<Point, Boolean> isInChatArea) {
        Strategy strategy = selectStrategy(client, consecutiveCollisions, targetObject);
        log.info("[CHATBOX_COLLISION] Attempting resolution with strategy: " + strategy);
        try {
            switch (strategy) {
                case ZOOM_OUT:
                    for (int i = 0; i < 3; i++) input.zoomOutSmall();
                    break;
                case ROTATE_LEFT:
                    input.rotateCameraSafe(-120, 0);
                    break;
                case ROTATE_RIGHT:
                    input.rotateCameraSafe(120, 0);
                    break;
                case TILT_UP:
                    input.rotateCameraSafe(0, -60);
                    break;
                case COMBINED:
                    input.rotateCameraSafe(90, -30);
                    input.zoomOutSmall();
                    input.zoomOutSmall();
                    break;
            }
            long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(200);
            while (System.nanoTime() < deadline) {
                // busy wait to avoid blocking threads
            }

            Point newProjection = null;
            if (targetObject != null) {
                LocalPoint lp = LocalPoint.fromWorld(client.getTopLevelWorldView(), targetObject.getWorldLocation());
                if (lp != null) {
                    net.runelite.api.Point apiPoint = Perspective.localToCanvas(client, lp, targetObject.getPlane());
                    if (apiPoint != null) newProjection = new Point(apiPoint.getX(), apiPoint.getY());
                }
            }
            if (newProjection != null && !isInChatArea.apply(newProjection)) {
                log.info("[CHATBOX_COLLISION] Camera adjustment successful - object now at " + newProjection);
                return true;
            } else {
                log.warn("[CHATBOX_COLLISION] Camera adjustment did not resolve collision");
                return false;
            }
        } catch (Exception e) {
            log.error("[CHATBOX_COLLISION] Error during camera adjustment: " + e.getMessage());
            return false;
        }
    }

    private static Strategy selectStrategy(Client client, int consecutiveCollisions, GameObject targetObject) {
        switch (consecutiveCollisions) {
            case 1: return Strategy.ZOOM_OUT;
            case 2:
                if (targetObject != null && client.getLocalPlayer() != null && client.getLocalPlayer().getWorldLocation() != null) {
                    int deltaX = targetObject.getWorldLocation().getX() - client.getLocalPlayer().getWorldLocation().getX();
                    return deltaX > 0 ? Strategy.ROTATE_RIGHT : Strategy.ROTATE_LEFT;
                }
                return Strategy.ROTATE_LEFT;
            case 3: return Strategy.TILT_UP;
            default: return Strategy.COMBINED;
        }
    }
}
