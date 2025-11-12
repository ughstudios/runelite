package net.runelite.client.plugins.rlbot.tasks;

public class CameraZoomInTask implements Task
{
    @Override
    public void run(TaskContext context)
    {
        context.input.zoomInSmall();
        context.input.zoomInSmall();
        context.setBusyForMs(80);
    }
}
