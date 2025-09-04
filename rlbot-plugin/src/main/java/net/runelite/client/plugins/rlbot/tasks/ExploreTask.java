package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.api.Player;
import net.runelite.api.coords.WorldPoint;

public class ExploreTask implements Task {
	@Override
	public boolean shouldRun(TaskContext ctx) {
		// Only explore when we can't find trees and have been stuck for a while
		if (ctx.isBusy() && !ctx.timedOutSince(1000)) {
			ctx.logger.info("[Explore] shouldRun=false: busy and not timed out");
			return false;
		}
		
		// Don't explore if we can see trees to chop (unless inventory is full)
		if (!ctx.isInventoryFull()) {
			net.runelite.api.GameObject tree = ObjectFinder.findNearestByNames(ctx, 
				new String[]{"tree", "oak", "willow", "yew", "maple"}, "Chop down");
			if (tree != null && ObjectFinder.projectToCanvas(ctx, tree) != null) {
				ctx.logger.info("[Explore] shouldRun=false: trees visible and inventory not full");
				return false; // Trees are visible and we can chop, no need to explore
			}
		}
		
		// Don't explore if we can see a bank and inventory is full
		if (ctx.isInventoryFull()) {
			net.runelite.api.GameObject bank = ObjectFinder.findNearestByNames(ctx, 
				new String[]{"bank booth", "bank chest"}, "Bank");
			if (bank != null && ObjectFinder.projectToCanvas(ctx, bank) != null) {
				ctx.logger.info("[Explore] shouldRun=false: bank visible and inventory full");
				return false; // Bank is visible, no need to explore
			}
		}
		
		// Explore if player hasn't moved recently (stuck) OR if inventory is full and no bank visible
		boolean playerMovingRecent = ctx.isPlayerMovingRecent(2000);
		boolean inventoryFull = ctx.isInventoryFull();
		boolean playerMovingRecentShort = ctx.isPlayerMovingRecent(1000);
		
		boolean shouldExplore = !playerMovingRecent || (inventoryFull && !playerMovingRecentShort);
		ctx.logger.info("[Explore] shouldRun=" + shouldExplore + ": playerMovingRecent=" + playerMovingRecent + ", inventoryFull=" + inventoryFull + ", playerMovingRecentShort=" + playerMovingRecentShort);
		return shouldExplore;
	}

	@Override
	public void run(TaskContext ctx) {
		UiHelper.closeObstructions(ctx);
		if (ctx.isBusy() && !ctx.timedOutSince(1000)) return;
		Client client = ctx.client;
		Player me = client.getLocalPlayer();
		if (me == null) return;
		WorldPoint myWp = me.getWorldLocation();
		int exploreRadius = 12;
		int offsetX = -exploreRadius; // deterministic westward step
		int offsetY = 0;
		WorldPoint rawTarget = new WorldPoint(myWp.getX() + offsetX, myWp.getY() + offsetY, myWp.getPlane());
		WorldPoint exploreTarget = NavigateToHotspotTask.ensurePathable(ctx, rawTarget);
		if (exploreTarget == null) {
			ctx.logger.warn("[Explore] Raw explore target not pathable; skipping step");
			ctx.setBusyForMs(300);
			return;
		}
		ctx.logger.info("[Explore] Stepping toward (" + exploreTarget.getX() + "," + exploreTarget.getY() + ")");
		RunHelper.ensureRunOn(ctx);
		boolean worldClicked = WorldPathing.clickStepToward(ctx, exploreTarget, 6);
		if (!worldClicked) {
			double jitter = 0.0; // deterministic
			MinimapPathing.stepTowards(ctx, exploreTarget, jitter);
		}
		ctx.setBusyForMs(500);
	}
}
