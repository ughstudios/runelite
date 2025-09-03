package net.runelite.client.plugins.rlbot.tasks;

import java.awt.Point;
import java.awt.Rectangle;
import net.runelite.api.Client;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;

final class RunHelper {
    // Varp 173 is the run toggle (1 = running, 0 = walking)
    private static final int VARP_RUN_TOGGLE = 173;
    // Fallback widget group:child for the run toggle (observed in logs gid=160 child=27)
    private static final int RUN_TOGGLE_GROUP = 160;
    private static final int RUN_TOGGLE_CHILD = 27;
    private static long lastToggleMs = 0L;
    private static final long TOGGLE_COOLDOWN_MS = 3000L;

    private RunHelper() {}

    static boolean isRunning(Client client) {
        try {
            return client.getVarpValue(VARP_RUN_TOGGLE) == 1;
        } catch (Exception e) {
            return false;
        }
    }

    static void ensureRunOn(TaskContext ctx) {
        Client client = ctx.client;
        if (isRunning(client)) return;
        long now = System.currentTimeMillis();
        if (now - lastToggleMs < TOGGLE_COOLDOWN_MS) return;
        clickRunToggle(ctx);
        lastToggleMs = now;
    }

    static void ensureRunOff(TaskContext ctx) {
        Client client = ctx.client;
        if (!isRunning(client)) return;
        long now = System.currentTimeMillis();
        if (now - lastToggleMs < TOGGLE_COOLDOWN_MS) return;
        clickRunToggle(ctx);
        lastToggleMs = now;
    }

    private static void clickRunToggle(TaskContext ctx) {
        try {
            // Prefer a direct widget menu action if available
            Widget orb = ctx.client.getWidget(WidgetInfo.MINIMAP_TOGGLE_RUN_ORB);
            if (orb != null && !orb.isHidden()) {
                int wid = orb.getId();
                ctx.logger.info("[Run] Toggling run via CC_OP on minimap orb widgetId=" + wid);
                ctx.clientThread.invoke(() -> {
                    try {
                        ctx.client.menuAction(-1, wid, net.runelite.api.MenuAction.CC_OP, 1, -1, "Toggle Run", "");
                    } catch (Exception ignored) {}
                });
                ctx.setBusyForMs(200);
                return;
            }
            // Fallback to group:child observed in widget dump
            Widget w = ctx.client.getWidget(RUN_TOGGLE_GROUP, RUN_TOGGLE_CHILD);
            if (w != null && !w.isHidden()) {
                int wid = w.getId();
                ctx.logger.info("[Run] Toggling run via CC_OP on widget 160:27 id=" + wid);
                ctx.clientThread.invoke(() -> {
                    try {
                        ctx.client.menuAction(-1, wid, net.runelite.api.MenuAction.CC_OP, 1, -1, "Toggle Run", "");
                    } catch (Exception ignored) {}
                });
                ctx.setBusyForMs(200);
            }
        } catch (Exception ignored) {
        }
    }
}


