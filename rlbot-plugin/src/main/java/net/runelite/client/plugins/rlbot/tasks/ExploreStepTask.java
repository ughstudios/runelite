package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Player;
import net.runelite.api.coords.WorldPoint;

public class ExploreStepTask implements Task {
	public enum Cardinal { NORTH, SOUTH, EAST, WEST }
	private final Cardinal dir;
	private final int tiles;

	public ExploreStepTask(Cardinal dir, int tiles) {
		this.dir = dir;
		this.tiles = Math.max(1, Math.min(tiles, 30));
	}

	@Override
	public boolean shouldRun(TaskContext ctx) { return true; }

	@Override
	public void run(TaskContext ctx) {
		UiHelper.closeObstructions(ctx);
		if (ctx.isBusy() && !ctx.timedOutSince(200)) return;
		Player me = ctx.client.getLocalPlayer();
		if (me == null) return;
		WorldPoint wp = me.getWorldLocation();
		int dx = 0, dy = 0;
		switch (dir) {
			case NORTH: dy = tiles; break;
			case SOUTH: dy = -tiles; break;
			case EAST: dx = tiles; break;
			case WEST: dx = -tiles; break;
		}
		WorldPoint target = new WorldPoint(wp.getX() + dx, wp.getY() + dy, wp.getPlane());
		boolean clicked = WorldPathing.clickStepToward(ctx, target, Math.min(6, tiles));
		if (!clicked) {
			MinimapPathing.stepTowards(ctx, target, 0);
		}
		ctx.setBusyForMs(400);
	}
}
