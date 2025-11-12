package net.runelite.client.plugins.rlbot.input;

import java.awt.Canvas;
import java.awt.Component;
import java.awt.Point;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import javax.swing.SwingUtilities;
import net.runelite.api.Client;
import net.runelite.client.input.KeyManager;
import net.runelite.client.input.MouseManager;
import net.runelite.client.plugins.rlbot.RLBotLogger;
import net.runelite.client.plugins.rlbot.RLBotPlugin;

/**
 * Low-level event dispatcher for mouse and keyboard events. Centralizes
 * stretched-coordinate transforms and overlay sync for synthetic pointer moves.
 */
final class InputDispatcher
{
    private final RLBotLogger logger;
    private final Client client;
    private final KeyManager keyManager;
    private final MouseManager mouseManager;
    private volatile RLBotPlugin plugin; // used to update overlay mouse position

    InputDispatcher(RLBotLogger logger, Client client, KeyManager keyManager, MouseManager mouseManager, RLBotPlugin plugin)
    {
        this.logger = logger;
        this.client = client;
        this.keyManager = keyManager;
        this.mouseManager = mouseManager;
        this.plugin = plugin;
    }

    void setPlugin(RLBotPlugin plugin)
    {
        this.plugin = plugin;
    }

    Canvas getCanvas()
    {
        if (client == null)
        {
            logger.error("Client is null, cannot get canvas");
            return null;
        }
        return client.getCanvas();
    }

    Point toStretched(Point real)
    {
        if (!client.isStretchedEnabled())
        {
            return real;
        }
        java.awt.Dimension stretched = client.getStretchedDimensions();
        java.awt.Dimension realDim = client.getRealDimensions();
        if (stretched == null || realDim == null)
        {
            return real;
        }
        double sx = (double) stretched.width / realDim.width;
        double sy = (double) stretched.height / realDim.height;
        int x = (int) Math.round(real.x * sx);
        int y = (int) Math.round(real.y * sy);
        return new Point(x, y);
    }

    void dispatchMouseMove(Component c, Point realCanvasPoint)
    {
        Point p = toStretched(realCanvasPoint);
        long when = System.currentTimeMillis();
        MouseEvent evt = new MouseEvent(
            c,
            MouseEvent.MOUSE_MOVED,
            when,
            0,
            p.x,
            p.y,
            0,
            false,
            0
        );
        if (plugin != null)
        {
            // overlay expects real canvas coordinates
            plugin.updateLastSyntheticMouseLocation(realCanvasPoint);
        }
        SwingUtilities.invokeLater(() -> c.dispatchEvent(evt));
    }

    void dispatchMouseClick(Component c, Point realCanvasPoint)
    {
        Point p = toStretched(realCanvasPoint);
        long when = System.currentTimeMillis();
        int mods = InputEvent.BUTTON1_DOWN_MASK;
        MouseEvent press = new MouseEvent(c, MouseEvent.MOUSE_PRESSED, when, mods, p.x, p.y, 1, false, MouseEvent.BUTTON1);
        MouseEvent release = new MouseEvent(c, MouseEvent.MOUSE_RELEASED, when + 50, mods, p.x, p.y, 1, false, MouseEvent.BUTTON1);
        MouseEvent click = new MouseEvent(c, MouseEvent.MOUSE_CLICKED, when + 51, mods, p.x, p.y, 1, false, MouseEvent.BUTTON1);
        if (plugin != null)
        {
            plugin.updateLastSyntheticMouseLocation(realCanvasPoint);
        }
        SwingUtilities.invokeLater(() -> c.dispatchEvent(press));
        SwingUtilities.invokeLater(() -> c.dispatchEvent(release));
        SwingUtilities.invokeLater(() -> c.dispatchEvent(click));
    }

    void dispatchMouseRightClick(Component c, Point realCanvasPoint)
    {
        Point p = toStretched(realCanvasPoint);
        long when = System.currentTimeMillis();
        int mods = InputEvent.BUTTON3_DOWN_MASK;
        boolean popup = true;
        MouseEvent press = new MouseEvent(c, MouseEvent.MOUSE_PRESSED, when, mods, p.x, p.y, 1, popup, MouseEvent.BUTTON3);
        MouseEvent release = new MouseEvent(c, MouseEvent.MOUSE_RELEASED, when + 50, mods, p.x, p.y, 1, popup, MouseEvent.BUTTON3);
        MouseEvent click = new MouseEvent(c, MouseEvent.MOUSE_CLICKED, when + 51, mods, p.x, p.y, 1, popup, MouseEvent.BUTTON3);
        SwingUtilities.invokeLater(() -> c.dispatchEvent(press));
        SwingUtilities.invokeLater(() -> c.dispatchEvent(release));
        SwingUtilities.invokeLater(() -> c.dispatchEvent(click));
    }

    void dispatchMiddleDrag(Component c, Point realStart, Point realEnd)
    {
        Point s = toStretched(realStart);
        Point e = toStretched(realEnd);
        long when = System.currentTimeMillis();
        int button = MouseEvent.BUTTON2;
        int mods = InputEvent.getMaskForButton(button);

        // ensure a move to start
        dispatchMouseMove(c, realStart);

        MouseEvent press = new MouseEvent(c, MouseEvent.MOUSE_PRESSED, when, mods, s.x, s.y, 1, false, button);
        mouseManager.processMousePressed(press);

        int steps = 6;
        for (int i = 1; i <= steps; i++)
        {
            int x = s.x + (e.x - s.x) * i / steps;
            int y = s.y + (e.y - s.y) * i / steps;
            MouseEvent drag = new MouseEvent(c, MouseEvent.MOUSE_DRAGGED, when + 10L * i, mods, x, y, 1, false, button);
            mouseManager.processMouseDragged(drag);
        }

        MouseEvent release = new MouseEvent(c, MouseEvent.MOUSE_RELEASED, when + 10L * (steps + 2), mods, e.x, e.y, 1, false, button);
        mouseManager.processMouseReleased(release);
    }

    void dispatchWheel(Canvas canvas, int wheelRotation)
    {
        long when = System.currentTimeMillis();
        int cx = Math.max(1, canvas.getWidth() / 2);
        int cy = Math.max(1, canvas.getHeight() / 2);
        // tiny move first so overlay stays in sync
        dispatchMouseMove(canvas, new Point(cx, cy));
        java.awt.event.MouseWheelEvent wheel = new java.awt.event.MouseWheelEvent(
            canvas,
            java.awt.event.MouseEvent.MOUSE_WHEEL,
            when,
            0,
            cx,
            cy,
            0,
            false,
            java.awt.event.MouseWheelEvent.WHEEL_UNIT_SCROLL,
            1,
            wheelRotation
        );
        canvas.requestFocus();
        SwingUtilities.invokeLater(() -> canvas.dispatchEvent(wheel));
        try { mouseManager.processMouseWheelMoved(wheel); } catch (Exception ignored) {}
    }

    void dispatchKeyPressRelease(Component c, int keyCode, long durationMs)
    {
        long when = System.currentTimeMillis();
        if (!c.isFocusOwner())
        {
            c.requestFocusInWindow();
        }
        KeyEvent press = new KeyEvent(c, KeyEvent.KEY_PRESSED, when, 0, keyCode, KeyEvent.CHAR_UNDEFINED);
        SwingUtilities.invokeLater(() -> c.dispatchEvent(press));
        try { keyManager.processKeyPressed(press); } catch (Exception ignored) {}

        Thread releaser = new Thread(() -> {
            try { Thread.sleep(Math.max(1, durationMs)); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            KeyEvent release = new KeyEvent(c, KeyEvent.KEY_RELEASED, System.currentTimeMillis(), 0, keyCode, KeyEvent.CHAR_UNDEFINED);
            SwingUtilities.invokeLater(() -> c.dispatchEvent(release));
            try { keyManager.processKeyReleased(release); } catch (Exception ignored) {}
        }, "rlbot-key-release");
        releaser.setDaemon(true);
        releaser.start();
    }

    void dispatchKeyEvent(Component c, int keyCode)
    {
        long when = System.currentTimeMillis();
        KeyEvent press = new KeyEvent(c, KeyEvent.KEY_PRESSED, when, 0, keyCode, KeyEvent.CHAR_UNDEFINED);
        char typedChar = KeyEvent.CHAR_UNDEFINED;
        boolean sendTyped = false;
        if (keyCode >= KeyEvent.VK_0 && keyCode <= KeyEvent.VK_9)
        {
            typedChar = (char) ('0' + (keyCode - KeyEvent.VK_0));
            sendTyped = true;
        }
        else if (keyCode >= KeyEvent.VK_A && keyCode <= KeyEvent.VK_Z)
        {
            typedChar = (char) ('a' + (keyCode - KeyEvent.VK_A));
            sendTyped = true;
        }
        else if (keyCode == KeyEvent.VK_SPACE)
        {
            typedChar = ' ';
            sendTyped = true;
        }
        KeyEvent typed = null;
        if (sendTyped)
        {
            typed = new KeyEvent(c, KeyEvent.KEY_TYPED, when + 10, 0, KeyEvent.VK_UNDEFINED, typedChar);
        }
        KeyEvent release = new KeyEvent(c, KeyEvent.KEY_RELEASED, when + 50, 0, keyCode, KeyEvent.CHAR_UNDEFINED);
        try
        {
            keyManager.processKeyPressed(press);
            if (typed != null) { keyManager.processKeyTyped(typed); }
            keyManager.processKeyReleased(release);
        }
        catch (Exception e)
        {
            logger.error("Error dispatching key events: " + e.getMessage() + ": " + e.toString());
        }
    }

    void dispatchCharKeyEvent(Component c, int keyCode, char keyChar)
    {
        long when = System.currentTimeMillis();
        KeyEvent press = new KeyEvent(c, KeyEvent.KEY_PRESSED, when, 0, keyCode, KeyEvent.CHAR_UNDEFINED);
        KeyEvent typed = new KeyEvent(c, KeyEvent.KEY_TYPED, when + 10, 0, KeyEvent.VK_UNDEFINED, keyChar);
        KeyEvent release = new KeyEvent(c, KeyEvent.KEY_RELEASED, when + 50, 0, keyCode, KeyEvent.CHAR_UNDEFINED);
        try
        {
            keyManager.processKeyPressed(press);
            keyManager.processKeyTyped(typed);
            keyManager.processKeyReleased(release);
        }
        catch (Exception e)
        {
            logger.error("Error dispatching char key events: " + e.getMessage());
        }
    }

    void dispatchModifierKey(Component c, int keyCode, boolean press)
    {
        long when = System.currentTimeMillis();
        int id = press ? KeyEvent.KEY_PRESSED : KeyEvent.KEY_RELEASED;
        KeyEvent evt = new KeyEvent(c, id, when, 0, keyCode, KeyEvent.CHAR_UNDEFINED);
        if (press)
        {
            try { keyManager.processKeyPressed(evt); } catch (Exception ignored) {}
        }
        else
        {
            try { keyManager.processKeyReleased(evt); } catch (Exception ignored) {}
        }
    }
}
