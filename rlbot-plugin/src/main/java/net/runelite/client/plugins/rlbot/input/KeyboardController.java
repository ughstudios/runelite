package net.runelite.client.plugins.rlbot.input;

import java.awt.Canvas;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * High-level keyboard operations (press, hold, type).
 */
final class KeyboardController
{
    private final RLBotLogger logger;
    private final ClientThread clientThread;
    private final InputDispatcher dispatch;

    KeyboardController(RLBotLogger logger, ClientThread clientThread, InputDispatcher dispatch)
    {
        this.logger = logger;
        this.clientThread = clientThread;
        this.dispatch = dispatch;
    }

    void pressKey(int keyCode)
    {
        clientThread.invoke(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null) { return; }
            try { dispatch.dispatchKeyEvent(canvas, keyCode); }
            catch (Exception e) { logger.error("[Keyboard] pressKey error: " + e.getMessage()); }
        });
    }

    void holdKey(int keyCode, long durationMs)
    {
        clientThread.invoke(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null) { return; }
            try { dispatch.dispatchKeyPressRelease(canvas, keyCode, durationMs); }
            catch (Exception e) { logger.error("[Keyboard] holdKey error: " + e.getMessage()); }
        });
    }

    void typeText(String text)
    {
        clientThread.invoke(() -> {
            Canvas canvas = dispatch.getCanvas();
            if (canvas == null) { return; }
            for (char c : text.toCharArray())
            {
                int code = java.awt.event.KeyEvent.getExtendedKeyCodeForChar(c);
                boolean upper = Character.isUpperCase(c);
                if (upper) { dispatch.dispatchModifierKey(canvas, java.awt.event.KeyEvent.VK_SHIFT, true); }
                dispatch.dispatchCharKeyEvent(canvas, code, c);
                if (upper) { dispatch.dispatchModifierKey(canvas, java.awt.event.KeyEvent.VK_SHIFT, false); }
            }
        });
    }
}

