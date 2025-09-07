package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.api.Player;
import net.runelite.api.coords.WorldPoint;
import net.runelite.client.plugins.rlbot.rewards.LogQualityRewards;

public class ExploreTask implements Task {
	@Override
	public boolean shouldRun(TaskContext ctx) {
		// Only explore when we can't find trees and have been stuck for a while
		/*if (ctx.isBusy() && !ctx.timedOutSince(1000)) {
			ctx.logger.info("[Explore] shouldRun=false: busy and not timed out");
			return false;
		}
		// Do not explore if inventory is full; banking must take priority
		if (ctx.isInventoryFull()) {
			ctx.logger.info("[Explore] shouldRun=false: inventory full");
			return false;
		}
		
		// AGGRESSIVE EXPLORATION: Check if we're stuck on low-tier trees when higher-tier are available
		int wc = 1; 
		try { 
			wc = ctx.client.getRealSkillLevel(net.runelite.api.Skill.WOODCUTTING); 
		} catch (Exception ignored) {}
		
		// Check what tier trees we currently have available
		java.util.List<WorldPoint> availableTrees = TreeDiscovery.getAvailableTrees();
		int bestAvailableTier = 0;
		for (WorldPoint tree : availableTrees) {
			// Get tree name from config
			java.util.List<net.runelite.client.plugins.rlbot.config.RLBotConfigManager.TreeLocation> trees = net.runelite.client.plugins.rlbot.config.RLBotConfigManager.getTrees();
			for (net.runelite.client.plugins.rlbot.config.RLBotConfigManager.TreeLocation t : trees) {
				if (t.toWorldPoint().equals(tree)) {
					int tier = LogQualityRewards.getLogQualityTier(t.name);
					if (tier > bestAvailableTier) bestAvailableTier = tier;
					break;
				}
			}
		}
		
		// Check what tier we could potentially find based on our level
		int maxPossibleTier = 1; // default
		if (wc >= 75) maxPossibleTier = 8; // magic
		else if (wc >= 60) maxPossibleTier = 7; // yew
		else if (wc >= 45) maxPossibleTier = 5; // maple
		else if (wc >= 30) maxPossibleTier = 3; // willow
		else if (wc >= 15) maxPossibleTier = 2; // oak
		
		// AGGRESSIVE: Explore if we have low-tier trees available but could find higher-tier ones
		boolean hasLowTierOnly = bestAvailableTier > 0 && bestAvailableTier < maxPossibleTier;
		boolean noHighTierDiscovered = bestAvailableTier < maxPossibleTier;
		
		if (hasLowTierOnly || noHighTierDiscovered) {
			ctx.logger.info("[Explore] AGGRESSIVE MODE: bestAvailableTier=" + bestAvailableTier + 
				", maxPossibleTier=" + maxPossibleTier + ", wcLevel=" + wc + " - seeking higher-tier trees!");
			return true;
		}
		
		// Check if we can see any trees at all (using level-appropriate trees)
		String[] allowedTrees = TreeDiscovery.allowedTreeNamesForLevel(wc);
		net.runelite.api.GameObject tree = ObjectFinder.findNearestByNames(ctx, 
			allowedTrees, "Chop down");
		if (tree != null && ObjectFinder.projectToCanvas(ctx, tree) != null) {
			ctx.logger.info("[Explore] shouldRun=false: trees visible and inventory not full");
			return false; // Trees are visible and we can chop, no need to explore
		}
		
		// Explore if player hasn't moved recently (stuck) OR if inventory is full and no bank visible
		boolean playerMovingRecent = ctx.isPlayerMovingRecent(2000);
		boolean inventoryFull = ctx.isInventoryFull();
		boolean playerMovingRecentShort = ctx.isPlayerMovingRecent(1000);
		
		boolean shouldExplore = !playerMovingRecent || (inventoryFull && !playerMovingRecentShort);
		ctx.logger.info("[Explore] shouldRun=" + shouldExplore + ": playerMovingRecent=" + playerMovingRecent + ", inventoryFull=" + inventoryFull + ", playerMovingRecentShort=" + playerMovingRecentShort);
		return shouldExplore;*/
		return true;
	}

	@Override
	public void run(TaskContext ctx) {
		UiHelper.closeObstructions(ctx);
		if (ctx.isBusy() && !ctx.timedOutSince(1000)) return;
		Client client = ctx.client;
		Player me = client.getLocalPlayer();
		if (me == null) return;
		WorldPoint myWp = me.getWorldLocation();
		
		// AGGRESSIVE EXPLORATION: Much larger radius and varied directions
		int exploreRadius = 25; // Increased from 12 to 25
		
		// Use random directions instead of deterministic westward movement
		java.util.Random rng = new java.util.Random();
		double angle = rng.nextDouble() * 2 * Math.PI; // Random angle 0-2π
		int offsetX = (int) (Math.cos(angle) * exploreRadius);
		int offsetY = (int) (Math.sin(angle) * exploreRadius);
		
		WorldPoint rawTarget = new WorldPoint(myWp.getX() + offsetX, myWp.getY() + offsetY, myWp.getPlane());
		WorldPoint exploreTarget = NavigateToHotspotTask.ensurePathable(ctx, rawTarget);
		if (exploreTarget == null) {
			ctx.logger.warn("[Explore] Raw explore target not pathable; trying smaller radius");
			// Try smaller radius if the large one fails
			exploreRadius = 15;
			angle = rng.nextDouble() * 2 * Math.PI;
			offsetX = (int) (Math.cos(angle) * exploreRadius);
			offsetY = (int) (Math.sin(angle) * exploreRadius);
			rawTarget = new WorldPoint(myWp.getX() + offsetX, myWp.getY() + offsetY, myWp.getPlane());
			exploreTarget = NavigateToHotspotTask.ensurePathable(ctx, rawTarget);
		}
		
		if (exploreTarget == null) {
			ctx.logger.warn("[Explore] All explore targets not pathable; skipping step");
			ctx.setBusyForMs(300);
			return;
		}
		
		ctx.logger.info("[Explore] AGGRESSIVE exploration toward (" + exploreTarget.getX() + "," + exploreTarget.getY() + ") - radius=" + exploreRadius);
		RunHelper.ensureRunOn(ctx);
		
		// Try world click first for shorter distances
		boolean worldClicked = WorldPathing.clickStepToward(ctx, exploreTarget, 8);
		if (!worldClicked) {
			// Use minimap for longer distances with some jitter
			double jitter = Math.toRadians(rng.nextDouble() * 30 - 15); // ±15 degrees jitter
			MinimapPathing.stepTowards(ctx, exploreTarget, jitter);
		}
		ctx.setBusyForMs(300); // Reduced busy time for faster exploration
	}
}
