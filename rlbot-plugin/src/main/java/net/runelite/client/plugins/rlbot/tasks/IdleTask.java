package net.runelite.client.plugins.rlbot.tasks;

/**
 * A no-op task that intentionally waits without blocking threads.
 * It just marks the context busy for a short window so the agent can "do nothing".
 */
public class IdleTask implements Task {
    @Override
    public boolean shouldRun(TaskContext context) {
        // Always eligible; caller/policy should gate when appropriate
        return true;
    }

    @Override
    public void run(TaskContext context) {
        // If currently woodcutting animation is playing, prefer a longer idle
        boolean cutting = context.isWoodcuttingAnim();
        int ms = cutting ? 600 : 250;
        context.logger.info("[Idle] Setting busy for " + ms + "ms");
        context.setBusyForMs(ms);
    }
}


