package net.runelite.client.plugins.rlbot.tasks;

public class CameraRotateTask implements Task {
	private static long lastRotationMs = 0L;
	
	@Override
	public boolean shouldRun(TaskContext ctx) {
		// Only rotate camera when player is stuck (not moving for a while)
		if (ctx.isBusy() && !ctx.timedOutSince(200)) return false;
		
		// Check if player hasn't moved recently (indicating stuck state)
		boolean playerStuck = !ctx.isPlayerMovingRecent(5000); // No movement for 5 seconds
		
		// Also check if we have high navigation failure count (indicating pathfinding issues)
		boolean navStuck = ctx.getNavNoProgressCount() >= 5; // Higher threshold
		
		// Only run camera rotation when actually stuck
		boolean shouldRotate = playerStuck || navStuck;
		
		// Add cooldown to prevent constant rotation (max once every 3 seconds)
		long now = System.currentTimeMillis();
		if (shouldRotate && (now - lastRotationMs < 3000)) {
			return false; // Too soon since last rotation
		}
		
		return shouldRotate;
	}

	@Override
	public void run(TaskContext ctx) {
		UiHelper.closeObstructions(ctx);
		if (ctx.isBusy() && !ctx.timedOutSince(200)) return;
		
		ctx.logger.info("[CameraRotate] Rotating camera to help with stuck navigation");
		
		// Do a single, controlled rotation instead of random amounts
		try {
			ctx.clientThread.invoke(() -> {
				try {
					// Enable relaxed pitch so we can tilt freely
					ctx.client.setCameraPitchRelaxerEnabled(true);

					int yawTarget = ctx.client.getCameraYawTarget();
					int pitchTarget = ctx.client.getCameraPitchTarget();

					// Single rotation: 90 degrees clockwise (512 units in RuneLite's coordinate system)
					int newYaw = (yawTarget + 512) & 0x7FF; // wrap 0..2047
					
					// Keep pitch the same or make a small adjustment
					int newPitch = Math.max(128, Math.min(512, pitchTarget + 64)); // Small upward tilt

					ctx.client.setCameraYawTarget(newYaw);
					ctx.client.setCameraPitchTarget(newPitch);
					
					ctx.logger.info("[CameraRotate] Rotated camera: yaw " + yawTarget + " -> " + newYaw + ", pitch " + pitchTarget + " -> " + newPitch);
				} catch (Throwable t) {
					// Fallback: single camera drag if direct setters fail
					ctx.input.rotateCameraSafe(180, 32); // Single drag right and up
					ctx.logger.info("[CameraRotate] Used fallback camera drag");
				}
			});
		} catch (Exception e) {
			// Final fallback: single camera drag
			ctx.input.rotateCameraSafe(180, 32);
			ctx.logger.info("[CameraRotate] Used final fallback camera drag");
		}
		
		// Record that we rotated
		lastRotationMs = System.currentTimeMillis();
		
		ctx.setBusyForMs(500); // Longer busy time
	}
}
