package net.runelite.client.plugins.rlbot.tasks.context;

import net.runelite.client.plugins.rlbot.tasks.TaskContext;

public final class NavTracker {
    private Integer lastNavDistance = null;
    private long lastNavUpdateMs = 0L;
    private int navNoProgressCount = 0;

    public void updateNavProgress(TaskContext ctx, int currentDistanceTiles) {
        long now = System.currentTimeMillis();
        if (lastNavDistance == null) {
            lastNavDistance = currentDistanceTiles;
            lastNavUpdateMs = now;
            navNoProgressCount = 0;
            return;
        }
        if (currentDistanceTiles <= lastNavDistance - 3) {
            navNoProgressCount = 0;
            lastNavDistance = currentDistanceTiles;
            lastNavUpdateMs = now;
        } else {
            if (now - lastNavUpdateMs > Math.max(3000, net.runelite.client.plugins.rlbot.RLBotConstants.STUCK_NO_PROGRESS_WINDOW_MS)) {
                navNoProgressCount++;
                lastNavUpdateMs = now;
            }
        }
    }

    public int getNavNoProgressCount() {
        return navNoProgressCount;
    }

    public void resetNavProgress() {
        lastNavDistance = null;
        lastNavUpdateMs = 0L;
        navNoProgressCount = 0;
    }
}

