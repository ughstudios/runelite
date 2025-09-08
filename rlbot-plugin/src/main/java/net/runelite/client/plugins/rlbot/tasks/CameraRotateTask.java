package net.runelite.client.plugins.rlbot.tasks;

/**
 * Enhanced camera task that performs various camera adjustments when the agent is stuck.
 * Provides rotation, tilting, and zooming to help overcome navigation obstacles.
 */
public class CameraRotateTask implements Task {
	private static long lastCameraActionMs = 0L;
	private static int actionSequence = 0; // Track which action to perform next
	
	// Camera action types
	private enum CameraAction {
		ROTATE_LEFT,
		ROTATE_RIGHT, 
		ROTATE_180,
		TILT_UP,
		TILT_DOWN,
		ZOOM_OUT,
		ZOOM_IN,
		RESET_PITCH,
		SWEEP_HORIZONTAL
	}
	
	@Override
	public boolean shouldRun(TaskContext ctx) {
		// Only perform camera actions when specific conditions are met
		if (ctx.isBusy() && !ctx.timedOutSince(300)) return false;
		
		// Add cooldown to prevent constant camera movement (max once every 3 seconds)
		long now = System.currentTimeMillis();
		if (now - lastCameraActionMs < 3000) return false;
		
		// Check activity states
		boolean isChopping = ctx.isWoodcuttingAnim();
		boolean isBanking = isBankOpen(ctx);
		boolean isMoving = ctx.isPlayerMovingRecent(4000); // 4 seconds for this task
		boolean isWalking = ctx.isPlayerWalking();
		
		// Navigation stuck condition (higher priority)
		boolean navStuck = ctx.getNavNoProgressCount() >= 5; // Higher threshold than CameraAdjustmentTask
		
		// Trigger conditions based on user requirements
		boolean shouldRotate = false;
		String reason = "";
		
		// Case 1: Navigation is very stuck (priority condition)
		if (navStuck) {
			shouldRotate = true;
			reason = "navigation very stuck (count: " + ctx.getNavNoProgressCount() + ")";
		}
		// Case 2: Not chopping AND not moving for extended period (5+ seconds)
		else if (!isChopping && !ctx.isPlayerMovingRecent(5000)) {
			shouldRotate = true;
			reason = "not chopping and hasn't moved for 5+ seconds";
		}
		// Case 3: Not banking AND not moving for extended period (5+ seconds)
		else if (!isBanking && !isChopping && !ctx.isPlayerMovingRecent(5000)) {
			shouldRotate = true;
			reason = "not banking and hasn't moved for 5+ seconds";
		}
		// Case 4: Completely idle for very long time (8+ seconds)
		else if (!isChopping && !isBanking && !isWalking && !ctx.isPlayerMovingRecent(8000)) {
			shouldRotate = true;
			reason = "completely idle for 8+ seconds";
		}
		
		if (shouldRotate) {
			ctx.logger.info("[CameraRotate] Triggering camera rotation - " + reason + 
				" (isChopping=" + isChopping + ", isBanking=" + isBanking + 
				", isMoving=" + isMoving + ", navStuck=" + navStuck + ")");
		}
		
		return shouldRotate;
	}
	
	/**
	 * Check if bank interface is open
	 */
	private boolean isBankOpen(TaskContext ctx) {
		try {
			net.runelite.api.widgets.Widget bank = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.BANK_CONTAINER);
			return bank != null && !bank.isHidden();
		} catch (Exception e) {
			return false;
		}
	}

	@Override
	public void run(TaskContext ctx) {
		UiHelper.closeObstructions(ctx);
		if (ctx.isBusy() && !ctx.timedOutSince(200)) return;
		
		// Select which camera action to perform based on sequence
		CameraAction action = selectCameraAction(ctx);
		ctx.logger.info("[CameraAdjust] Performing camera action: " + action);
		
		performCameraAction(ctx, action);
		
		// Record that we performed a camera action
		lastCameraActionMs = System.currentTimeMillis();
		actionSequence = (actionSequence + 1) % CameraAction.values().length;
		
		ctx.setBusyForMs(400); // Brief busy time to allow camera to settle
	}
	
	/**
	 * Select which camera action to perform based on current situation
	 */
	private CameraAction selectCameraAction(TaskContext ctx) {
		int navStuckCount = ctx.getNavNoProgressCount();
		boolean isVeryStuck = navStuckCount >= 5;
		boolean hasntMovedLong = !ctx.isPlayerMovingRecent(8000);
		
		// If very stuck, try more dramatic camera changes
		if (isVeryStuck || hasntMovedLong) {
			switch (actionSequence % 4) {
				case 0: return CameraAction.ZOOM_OUT;
				case 1: return CameraAction.ROTATE_180;
				case 2: return CameraAction.SWEEP_HORIZONTAL;
				default: return CameraAction.RESET_PITCH;
			}
		}
		
		// For moderate stuck situations, cycle through all actions
		CameraAction[] actions = CameraAction.values();
		return actions[actionSequence % actions.length];
	}
	
	/**
	 * Perform the specified camera action
	 */
	private void performCameraAction(TaskContext ctx, CameraAction action) {
		try {
			ctx.clientThread.invoke(() -> {
				try {
					// Enable relaxed pitch for full camera control
					ctx.client.setCameraPitchRelaxerEnabled(true);
					
					int currentYaw = ctx.client.getCameraYawTarget();
					int currentPitch = ctx.client.getCameraPitchTarget();
					
					switch (action) {
						case ROTATE_LEFT:
							rotateCameraYaw(ctx, currentYaw, -256); // 45 degrees left
							ctx.logger.info("[CameraAdjust] Rotated camera left");
							break;
							
						case ROTATE_RIGHT:
							rotateCameraYaw(ctx, currentYaw, 256); // 45 degrees right
							ctx.logger.info("[CameraAdjust] Rotated camera right");
							break;
							
						case ROTATE_180:
							rotateCameraYaw(ctx, currentYaw, 1024); // 180 degrees
							ctx.logger.info("[CameraAdjust] Rotated camera 180 degrees");
							break;
							
						case TILT_UP:
							adjustCameraPitch(ctx, currentPitch, -80); // Tilt up
							ctx.logger.info("[CameraAdjust] Tilted camera up");
							break;
							
						case TILT_DOWN:
							adjustCameraPitch(ctx, currentPitch, 80); // Tilt down
							ctx.logger.info("[CameraAdjust] Tilted camera down");
							break;
							
						case ZOOM_OUT:
							ctx.input.zoomOutSmall();
							ctx.input.zoomOutSmall(); // Double zoom out for better visibility
							ctx.logger.info("[CameraAdjust] Zoomed out");
							break;
							
						case ZOOM_IN:
							ctx.input.zoomInSmall();
							ctx.logger.info("[CameraAdjust] Zoomed in");
							break;
							
						case RESET_PITCH:
							// Reset to middle pitch for better overview
							ctx.client.setCameraPitchTarget(256); // Middle pitch
							ctx.logger.info("[CameraAdjust] Reset pitch to middle position");
							break;
							
						case SWEEP_HORIZONTAL:
							// Perform a horizontal sweep to scan area
							performHorizontalSweep(ctx, currentYaw);
							ctx.logger.info("[CameraAdjust] Performed horizontal sweep");
							break;
					}
					
				} catch (Throwable t) {
					// Fallback to input handler methods
					useFallbackCameraAction(ctx, action);
					ctx.logger.info("[CameraAdjust] Used fallback method for: " + action);
				}
			});
		} catch (Exception e) {
			// Final fallback
			useFallbackCameraAction(ctx, action);
			ctx.logger.info("[CameraAdjust] Used final fallback for: " + action);
		}
	}
	
	/**
	 * Rotate camera yaw by specified amount
	 */
	private void rotateCameraYaw(TaskContext ctx, int currentYaw, int deltaYaw) {
		int newYaw = (currentYaw + deltaYaw) & 0x7FF; // Wrap 0..2047
		ctx.client.setCameraYawTarget(newYaw);
	}
	
	/**
	 * Adjust camera pitch by specified amount
	 */
	private void adjustCameraPitch(TaskContext ctx, int currentPitch, int deltaPitch) {
		int newPitch = Math.max(128, Math.min(512, currentPitch + deltaPitch));
		ctx.client.setCameraPitchTarget(newPitch);
	}
	
	/**
	 * Perform a horizontal sweep to scan the surrounding area
	 */
	private void performHorizontalSweep(TaskContext ctx, int startYaw) {
		// Sweep 90 degrees left from current position
		int leftYaw = (startYaw - 256) & 0x7FF;
		
		// Set to left position (the sweep will be visible as camera movement)
		ctx.client.setCameraYawTarget(leftYaw);
		
		// The actual sweep motion will happen naturally as the camera moves to target
		// We could enhance this with timed intermediate positions if needed
	}
	
	/**
	 * Fallback camera actions using input handler when direct client calls fail
	 */
	private void useFallbackCameraAction(TaskContext ctx, CameraAction action) {
		switch (action) {
			case ROTATE_LEFT:
				ctx.input.rotateCameraSafe(-120, 0);
				break;
			case ROTATE_RIGHT:
				ctx.input.rotateCameraSafe(120, 0);
				break;
			case ROTATE_180:
				ctx.input.rotateCameraSafe(240, 0);
				break;
			case TILT_UP:
				ctx.input.rotateCameraSafe(0, -60);
				break;
			case TILT_DOWN:
				ctx.input.rotateCameraSafe(0, 60);
				break;
			case ZOOM_OUT:
				ctx.input.zoomOutSmall();
				ctx.input.zoomOutSmall();
				break;
			case ZOOM_IN:
				ctx.input.zoomInSmall();
				break;
			case RESET_PITCH:
				ctx.input.rotateCameraSafe(0, 0); // Neutral drag
				break;
			case SWEEP_HORIZONTAL:
				ctx.input.rotateCameraSafe(-120, 0);
				try { Thread.sleep(200); } catch (InterruptedException ignored) {}
				ctx.input.rotateCameraSafe(240, 0);
				break;
		}
	}
}

