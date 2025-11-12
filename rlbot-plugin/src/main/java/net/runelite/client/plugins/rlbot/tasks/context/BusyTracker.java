package net.runelite.client.plugins.rlbot.tasks.context;

import net.runelite.api.Player;
import net.runelite.api.AnimationID;
import net.runelite.api.coords.WorldPoint;
import net.runelite.client.plugins.rlbot.RLBotTelemetry;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;

public final class BusyTracker {
    private long lastActionMs = 0L;
    private long busyUntilMs = 0L;
    private WorldPoint lastWorldPoint = null;
    private long lastMoveMs = 0L;

    public boolean isBusy(TaskContext ctx) {
        Player p = ctx.client.getLocalPlayer();
        if (p == null) return false;
        int anim = p.getAnimation();
        boolean performing = anim != -1 && anim != AnimationID.IDLE;
        long now = System.currentTimeMillis();
        WorldPoint wp = p.getWorldLocation();
        if (wp != null) {
            if (lastWorldPoint == null || !wp.equals(lastWorldPoint)) {
                lastWorldPoint = wp;
                lastMoveMs = now;
            }
        }
        boolean moving = isPlayerWalking(ctx); // improved walking detection
        boolean inDialog = isDialogOpen(ctx);
        boolean timeLocked = System.currentTimeMillis() < busyUntilMs;
        return performing || moving || inDialog || timeLocked;
    }

    public void refreshMovementFromPlayer(TaskContext ctx) {
        try {
            Player p = ctx.client.getLocalPlayer();
            if (p == null) return;
            WorldPoint wp = p.getWorldLocation();
            if (wp == null) return;
            if (lastWorldPoint == null || !wp.equals(lastWorldPoint)) {
                lastWorldPoint = wp;
                lastMoveMs = System.currentTimeMillis();
            }
        } catch (Exception ignored) {}
    }

    public void setBusyForMs(TaskContext ctx, long durationMs) {
        long now = System.currentTimeMillis();
        long proposed = now + Math.max(25, durationMs);
        if (busyUntilMs > 0 && busyUntilMs - now > 3000L) {
            busyUntilMs = now + 1000L;
        } else {
            busyUntilMs = Math.max(busyUntilMs, proposed);
        }
        lastActionMs = now;
        RLBotTelemetry t = ctx.telemetry;
        if (t != null) {
            t.setBusyRemainingMs(durationMs);
            t.incEpisodeSteps();
        }
    }

    public void clearBusyLock() {
        busyUntilMs = 0L;
        lastActionMs = 0L;
    }

    public boolean timedOutSince(long sinceMs) {
        return System.currentTimeMillis() - lastActionMs > sinceMs;
    }

    public boolean isPlayerMovingRecent(long withinMs) {
        return System.currentTimeMillis() - lastMoveMs < withinMs;
    }

    public boolean isPlayerWalking(TaskContext ctx) {
        Player p = ctx.client.getLocalPlayer();
        if (p == null) return false;
        long now = System.currentTimeMillis();
        boolean positionChanged = now - lastMoveMs < 600;
        int anim = p.getAnimation();
        boolean isWalkingAnim = (anim == 819 || anim == 820 || anim == 821 || anim == 822 || anim == 824);
        return positionChanged || isWalkingAnim;
    }

    public long getLastMoveMs() { return lastMoveMs; }

    public long getBusyRemainingMsEstimate() {
        long now = System.currentTimeMillis();
        return Math.max(0L, busyUntilMs - now);
    }

    public String getBusyDebugString(TaskContext ctx) {
        Player p = ctx.client.getLocalPlayer();
        int anim = p != null ? p.getAnimation() : -1;
        long now = System.currentTimeMillis();
        boolean timeLocked = now < busyUntilMs;
        boolean recentlyMoved = now - lastMoveMs < 600;
        return "anim=" + anim + ", moving=" + recentlyMoved + ", dialog=" + isDialogOpen(ctx) + ", busyMsLeft=" + getBusyRemainingMsEstimate();
    }

    private boolean isDialogOpen(TaskContext ctx) {
        net.runelite.api.widgets.Widget w = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.DIALOG_NPC_TEXT);
        if (w != null && !w.isHidden()) return true;
        net.runelite.api.widgets.Widget c = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.DIALOG_PLAYER_TEXT);
        return c != null && !c.isHidden();
    }
}

