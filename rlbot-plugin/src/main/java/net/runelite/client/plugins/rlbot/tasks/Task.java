package net.runelite.client.plugins.rlbot.tasks;

/**
 * A single, atomic behavior the agent can perform.
 */
public interface Task {

    /**
     * Execute one step of this task. Implementations should be idempotent per tick
     * and return quickly; they can rely on repeated calls across ticks to make progress.
     */
    void run(TaskContext context);
}

