package net.runelite.client.plugins.rlbot.tasks;

public class CameraRotateStepTask implements Task {
	public enum Direction { LEFT, RIGHT, UP, DOWN }
	private final Direction direction;
	private final int yawDelta;
	private final int pitchDelta;

	public CameraRotateStepTask(Direction direction) {
		this.direction = direction;
		// Fixed per-step amounts for consistency
		switch (direction) {
			case LEFT:
				yawDelta = -96; pitchDelta = 0; break;
			case RIGHT:
				yawDelta = 96; pitchDelta = 0; break;
			case UP:
				yawDelta = 0; pitchDelta = -16; break;
			case DOWN:
				yawDelta = 0; pitchDelta = 16; break;
			default:
				yawDelta = 0; pitchDelta = 0; break;
		}
	}

	@Override
	public boolean shouldRun(TaskContext ctx) { return true; }

	@Override
	public void run(TaskContext ctx) {
		UiHelper.closeObstructions(ctx);
		ctx.clientThread.invoke(() -> {
			try {
				ctx.client.setCameraPitchRelaxerEnabled(true);
				int yaw = ctx.client.getCameraYawTarget();
				int pitch = ctx.client.getCameraPitchTarget();
				int newYaw = (yaw + yawDelta) & 0x7FF;
				int newPitch = Math.max(128, Math.min(512, pitch + pitchDelta));
				ctx.client.setCameraYawTarget(newYaw);
				ctx.client.setCameraPitchTarget(newPitch);
			} catch (Throwable t) {
				int dx = (direction == Direction.LEFT ? -48 : direction == Direction.RIGHT ? 48 : 0);
				int dy = (direction == Direction.UP ? -12 : direction == Direction.DOWN ? 12 : 0);
				ctx.input.rotateCameraDrag(dx, dy);
			}
		});
		ctx.setBusyForMs(100);
	}
}
