package net.runelite.client.plugins.rlbot.tasks;

/**
 * Rotates the camera using RLBotInputHandler.
 */
abstract class CameraRotateTask implements Task
{
    private final int deltaX;
    private final int deltaY;

    CameraRotateTask(int deltaX, int deltaY)
    {
        this.deltaX = deltaX;
        this.deltaY = deltaY;
    }

    @Override
    public void run(TaskContext context)
    {
        context.input.rotateCameraSafe(deltaX, deltaY);
        context.setBusyForMs(100);
    }
}
