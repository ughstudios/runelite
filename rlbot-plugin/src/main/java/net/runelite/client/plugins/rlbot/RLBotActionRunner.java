package net.runelite.client.plugins.rlbot;

import java.util.List;
import net.runelite.client.plugins.rlbot.tasks.Task;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;

/**
 * Executes a requested macro action with optional eligibility gating.
 */
final class RLBotActionRunner
{
    private RLBotActionRunner() {}

    static final class Result
    {
        final int executedIndex;
        final boolean eligible;
        final String actionName;
        Result(int idx, boolean eligible, String actionName)
        {
            this.executedIndex = idx;
            this.eligible = eligible;
            this.actionName = actionName;
        }
    }

    static Result run(List<Task> tasks, int requestedIndex, TaskContext ctx, RLBotTelemetry telemetry, RLBotLogger logger)
    {
        if (requestedIndex < 0 || requestedIndex >= tasks.size())
        {
            String badName = "InvalidAction(" + requestedIndex + ")";
            logger.warn("[IPC] Action index " + requestedIndex + " out of range [0," + tasks.size() + ")");
            telemetry.setMode("External: " + badName);
            return new Result(-1, false, badName);
        }
        int index = requestedIndex;
        Task task = tasks.get(index);
        String name = task.getClass().getSimpleName();
        telemetry.setMode("External: " + name);

        try
        {
            task.run(ctx);
        }
        catch (Exception e)
        {
            logger.error("Task error in " + name + ": " + e.getMessage());
        }
        return new Result(index, true, name);
    }

}
