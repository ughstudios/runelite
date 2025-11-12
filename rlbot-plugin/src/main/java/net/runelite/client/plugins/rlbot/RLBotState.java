package net.runelite.client.plugins.rlbot;

import java.util.List;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.Player;
import net.runelite.api.Skill;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.widgets.WidgetInfo;
import net.runelite.client.plugins.rlbot.tasks.BankDiscovery;
import net.runelite.client.plugins.rlbot.tasks.ObjectFinder;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;
import net.runelite.client.plugins.rlbot.tasks.TreeDiscovery;

/**
 * Builds observations for the RL agent and provides a snapshot of the world state.
 */
final class RLBotState
{
    private RLBotState() {}

    static final List<String> STATE_FEATURE_NAMES = List.of(
        "inventory_free_slots_bucket",
        "bank_open",
        "is_woodcutting",
        "tree_distance_norm",
        "bank_distance_norm"
    );

    static final class Snapshot
    {
        final int freeSlots;
        final boolean bankOpen;
        final boolean inventoryFull;
        final boolean inventoryNearFull;
        final boolean woodcutting;
        final boolean movingRecent;
        final float runEnergy;
        final Integer treeDistance;
        final Integer bankDistance;
        final boolean treeVisible;
        final boolean bankVisible;
        final int woodcutXp;

        Snapshot(
            int freeSlots,
            boolean bankOpen,
            boolean inventoryFull,
            boolean inventoryNearFull,
            boolean woodcutting,
            boolean movingRecent,
            float runEnergy,
            Integer treeDistance,
            Integer bankDistance,
            boolean treeVisible,
            boolean bankVisible,
            int woodcutXp
        )
        {
            this.freeSlots = freeSlots;
            this.bankOpen = bankOpen;
            this.inventoryFull = inventoryFull;
            this.inventoryNearFull = inventoryNearFull;
            this.woodcutting = woodcutting;
            this.movingRecent = movingRecent;
            this.runEnergy = runEnergy;
            this.treeDistance = treeDistance;
            this.bankDistance = bankDistance;
            this.treeVisible = treeVisible;
            this.bankVisible = bankVisible;
            this.woodcutXp = woodcutXp;
        }

        boolean needsBank()
        {
            return inventoryFull || inventoryNearFull;
        }
    }

    static Snapshot capture(TaskContext ctx, Client client)
    {
        int freeSlots = ctx.getInventoryFreeSlots();
        boolean bankOpen = isBankOpen(ctx);
        boolean inventoryFull = ctx.isInventoryFull();
        boolean inventoryNearFull = ctx.isInventoryNearFull();
        boolean woodcutting = ctx.isWoodcuttingAnim();
        boolean movingRecent = ctx.isPlayerMovingRecent(600);
        float runEnergy = ctx.getRunEnergy01();
        Integer treeDistance = distanceToNearestTree(ctx, client);
        Integer bankDistance = distanceToNearestBank(ctx, client);
        boolean treeVisible = isAnyTreeVisible(ctx, client);
        boolean bankVisible = isAnyBankVisible(ctx, client);
        int woodcutXp = getWoodcutXp(client);
        return new Snapshot(
            freeSlots,
            bankOpen,
            inventoryFull,
            inventoryNearFull,
            woodcutting,
            movingRecent,
            runEnergy,
            treeDistance,
            bankDistance,
            treeVisible,
            bankVisible,
            woodcutXp
        );
    }

    static float[] buildStateVector(Snapshot snapshot)
    {
        float treeDistanceNorm = snapshot.treeDistance == null ? 1.0f : Math.min(1.0f, snapshot.treeDistance / 30.0f);
        float bankDistanceNorm = snapshot.bankDistance == null ? 1.0f : Math.min(1.0f, snapshot.bankDistance / 30.0f);
        return new float[]{
            snapshot.freeSlots / 4.0f,
            snapshot.bankOpen ? 1f : 0f,
            snapshot.woodcutting ? 1f : 0f,
            treeDistanceNorm,
            bankDistanceNorm
        };
    }

    private static boolean isBankOpen(TaskContext ctx)
    {
        try
        {
            return ctx.client.getWidget(WidgetInfo.BANK_CONTAINER) != null;
        }
        catch (Exception e)
        {
            return false;
        }
    }

    private static boolean isAnyTreeVisible(TaskContext ctx, Client client)
    {
        try
        {
            int wcLevel = client.getRealSkillLevel(Skill.WOODCUTTING);
            String[] allowed = TreeDiscovery.allowedTreeNamesForLevel(wcLevel);
            GameObject tree = ObjectFinder.findNearestByNames(ctx, allowed, "Chop down");
            if (tree == null)
            {
                return false;
            }
            return ObjectFinder.projectToCanvas(ctx, tree) != null;
        }
        catch (Exception e)
        {
            return false;
        }
    }

    private static boolean isAnyBankVisible(TaskContext ctx, Client client)
    {
        try
        {
            GameObject bank = ObjectFinder.findNearestBankInteractable(ctx);
            if (bank == null)
            {
                return false;
            }
            return ObjectFinder.projectToCanvas(ctx, bank) != null;
        }
        catch (Exception e)
        {
            return false;
        }
    }

    private static Integer distanceToNearestTree(TaskContext ctx, Client client)
    {
        try
        {
            Player me = client.getLocalPlayer();
            if (me == null)
            {
                return null;
            }
            WorldPoint myWp = me.getWorldLocation();
            if (myWp == null)
            {
                return null;
            }
            TreeDiscovery.scanAndDiscoverTrees(ctx);
            WorldPoint tree = TreeDiscovery.getNearestDiscoveredTree(myWp);
            if (tree == null)
            {
                return null;
            }
            int distance = myWp.distanceTo(tree);
            return distance >= 0 ? distance : null;
        }
        catch (Exception e)
        {
            return null;
        }
    }

    private static Integer distanceToNearestBank(TaskContext ctx, Client client)
    {
        try
        {
            Player me = client.getLocalPlayer();
            if (me == null)
            {
                return null;
            }
            WorldPoint myWp = me.getWorldLocation();
            if (myWp == null)
            {
                return null;
            }
            BankDiscovery.scanAndDiscoverBanks(ctx);
            WorldPoint bank = BankDiscovery.getNearestDiscoveredBank(myWp);
            if (bank == null)
            {
                return null;
            }
            int distance = myWp.distanceTo(bank);
            return distance >= 0 ? distance : null;
        }
        catch (Exception e)
        {
            return null;
        }
    }

    private static int getWoodcutXp(Client client)
    {
        try
        {
            return client.getSkillExperience(Skill.WOODCUTTING);
        }
        catch (Exception e)
        {
            return 0;
        }
    }
}

