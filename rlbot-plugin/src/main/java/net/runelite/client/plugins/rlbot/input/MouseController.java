package net.runelite.client.plugins.rlbot.input;

import java.awt.Canvas;
import java.awt.Component;
import java.awt.Point;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import javax.swing.SwingUtilities;
import net.runelite.api.Client;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * High-level mouse behavior: smooth movement, left/right click.
 */
final class MouseController
{
    private final RLBotLogger logger;
    private final Client client;
    private final ClientThread clientThread;
    private final InputDispatcher dispatch;

    private volatile Point lastCanvasMovePoint;

    MouseController(RLBotLogger logger, Client client, ClientThread clientThread, InputDispatcher dispatch)
    {
        this.logger = logger;
        this.client = client;
        this.clientThread = clientThread;
        this.dispatch = dispatch;
    }

    Point getLastCanvasMovePoint()
    {
        return lastCanvasMovePoint;
    }

    void clearLastCanvasMovePoint()
    {
        lastCanvasMovePoint = null;
    }

    void smoothMouseMove(Point canvasPoint)
    {
        logger.debug("[Mouse] smoothMouseMove to " + canvasPoint);
        CountDownLatch finished = new CountDownLatch(1);
        Thread mover = new Thread(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null)
            {
                logger.error("[Mouse] Canvas is null, cannot move");
                finished.countDown();
                return;
            }

            try
            {
                Point current = getCurrentMousePosition(canvas);
                if (current == null)
                {
                    current = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
                }
                Point start = current;
                Point end = new Point(canvasPoint);

                int steps = Math.max(6, Math.min(22, (int) (start.distance(end) / 14.0)));
                for (int i = 1; i <= steps; i++)
                {
                    double t = i / (double) steps; // ease-out-ish
                    t = 1 - Math.pow(1 - t, 2);
                    int x = (int) Math.round(start.x + (end.x - start.x) * t);
                    int y = (int) Math.round(start.y + (end.y - start.y) * t);
                    Point stepPoint = new Point(x, y);
                    dispatch.dispatchMouseMove(canvas, stepPoint);
                    lastCanvasMovePoint = stepPoint;
                }
            }
            finally
            {
                finished.countDown();
            }
        }, "rlbot-mouse-move");
        mover.setDaemon(true);
        mover.start();
        try
        {
            finished.await(800, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException ie)
        {
            Thread.currentThread().interrupt();
        }
        // flush pending events
        if (!SwingUtilities.isEventDispatchThread())
        {
            try { SwingUtilities.invokeAndWait(() -> { /* flush */ }); } catch (Exception ignored) {}
        }
    }

    void moveSync(Point canvasPoint)
    {
        Canvas canvas = dispatch.getCanvas();
        if (canvas == null)
        {
            logger.error("[Mouse] Canvas is null, cannot moveSync");
            return;
        }
        dispatch.dispatchMouseMoveSync(canvas, canvasPoint);
        lastCanvasMovePoint = canvasPoint;
    }

    void click()
    {
        Thread clicker = new Thread(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null)
            {
                logger.error("[Mouse] Canvas is null, cannot click");
                return;
            }
            Point p = lastCanvasMovePoint;
            if (p == null)
            {
                try
                {
                    Point mp = canvas.getMousePosition();
                    if (mp != null) { p = mp; }
                }
                catch (Exception ignored) {}
            }
            if (p == null)
            {
                p = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }
            dispatch.dispatchMouseMove(canvas, p);
            if (!SwingUtilities.isEventDispatchThread())
            {
                try { SwingUtilities.invokeAndWait(() -> { /* flush */ }); } catch (Exception ignored) {}
            }
            dispatch.dispatchMouseClick(canvas, p);
            lastCanvasMovePoint = null;
        }, "rlbot-click");
        clicker.setDaemon(true);
        clicker.start();
    }

    void clickAt(Point canvasPoint)
    {
        Thread clicker = new Thread(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null)
            {
                logger.error("[Mouse] Canvas is null, cannot clickAt");
                return;
            }
            dispatch.dispatchMouseMoveSync(canvas, canvasPoint);
            dispatch.dispatchMouseClickSync(canvas, canvasPoint);
            lastCanvasMovePoint = null;
        }, "rlbot-click");
        clicker.setDaemon(true);
        clicker.start();
    }

    void clickAtSync(Point canvasPoint)
    {
        Canvas canvas = dispatch.getCanvas();
        if (canvas == null)
        {
            logger.error("[Mouse] Canvas is null, cannot clickAtSync");
            return;
        }
        // Only click; use moveSync first if you need to reposition
        dispatch.dispatchMouseClickSync(canvas, canvasPoint);
        lastCanvasMovePoint = null;
    }

    void clickOnlySync(Point canvasPoint)
    {
        Canvas canvas = dispatch.getCanvas();
        if (canvas == null)
        {
            logger.error("[Mouse] Canvas is null, cannot clickOnlySync");
            return;
        }
        dispatch.dispatchMouseClickSync(canvas, canvasPoint);
        lastCanvasMovePoint = null;
    }

    void rightClickCurrent()
    {
        clientThread.invoke(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null) { return; }
            Point canvasPos;
            try
            {
                Point mp = canvas.getMousePosition();
                canvasPos = (mp != null) ? mp : new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }
            catch (Exception e)
            {
                canvasPos = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }
            dispatch.dispatchMouseMove(canvas, canvasPos);
            if (!SwingUtilities.isEventDispatchThread())
            {
                try { SwingUtilities.invokeAndWait(() -> { /* flush */ }); } catch (Exception ignored) {}
            }
            dispatch.dispatchMouseRightClick(canvas, canvasPos);
        });
    }

    private Point getCurrentMousePosition(Component canvas)
    {
        try
        {
            Point p = canvas.getMousePosition();
            if (p != null) { return p; }
        }
        catch (Exception ignored) {}
        return null;
    }
}
