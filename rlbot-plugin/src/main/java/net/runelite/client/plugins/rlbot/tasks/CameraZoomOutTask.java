package net.runelite.client.plugins.rlbot.tasks;

public class CameraZoomOutTask implements Task
{
    @Override
    public void run(TaskContext context)
    {
        context.input.zoomOutSmall();
        context.input.zoomOutSmall();
        context.setBusyForMs(80);
    }
}
