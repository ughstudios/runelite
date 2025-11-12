package net.runelite.client.plugins.rlbot.input;

import java.awt.Canvas;
import java.awt.Point;
import net.runelite.api.Client;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * Camera movement and zoom controls.
 */
final class CameraController
{
    private final RLBotLogger logger;
    private final Client client;
    private final ClientThread clientThread;
    private final InputDispatcher dispatch;
    private final KeyboardController keyboard;

    CameraController(RLBotLogger logger, Client client, ClientThread clientThread, InputDispatcher dispatch, KeyboardController keyboard)
    {
        this.logger = logger;
        this.client = client;
        this.clientThread = clientThread;
        this.dispatch = dispatch;
        this.keyboard = keyboard;
    }

    void rotateCameraDrag(int dx, int dy)
    {
        clientThread.invoke(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null) { return; }
            int vx = client.getViewportXOffset();
            int vy = client.getViewportYOffset();
            int vw = client.getViewportWidth();
            int vh = client.getViewportHeight();
            int sx = vx + Math.max(10, vw / 2);
            int sy = vy + Math.max(10, vh / 2);
            dispatch.dispatchMiddleDrag(canvas, new Point(sx, sy), new Point(sx + dx, sy + dy));
        });
    }

    void rotateCameraSafe(int dx, int dy)
    {
        int h = Math.max(1, Math.abs(dx) / 30);
        if (dx > 0) { for (int i = 0; i < h; i++) keyboard.holdKey(java.awt.event.KeyEvent.VK_RIGHT, 120); }
        else if (dx < 0) { for (int i = 0; i < h; i++) keyboard.holdKey(java.awt.event.KeyEvent.VK_LEFT, 120); }

        int v = Math.max(1, Math.abs(dy) / 30);
        if (dy > 0) { for (int i = 0; i < v; i++) keyboard.holdKey(java.awt.event.KeyEvent.VK_DOWN, 120); }
        else if (dy < 0) { for (int i = 0; i < v; i++) keyboard.holdKey(java.awt.event.KeyEvent.VK_UP, 120); }
    }

    void rotateCameraLeftSmall() { rotateCameraSafe(-80, 0); }
    void rotateCameraRightSmall() { rotateCameraSafe(80, 0); }
    void tiltCameraUpSmall() { rotateCameraSafe(0, -60); }
    void tiltCameraDownSmall() { rotateCameraSafe(0, 60); }

    void zoomInSmall() { zoomWheel(-3); }
    void zoomOutSmall() { zoomWheel(+3); }

    private void zoomWheel(int wheelRotation)
    {
        clientThread.invoke(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null) { return; }
            dispatch.dispatchWheel(canvas, wheelRotation);
        });
    }
}

