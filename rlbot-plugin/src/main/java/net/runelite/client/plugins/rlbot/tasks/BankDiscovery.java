package net.runelite.client.plugins.rlbot.tasks;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import net.runelite.api.ObjectComposition;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.api.TileObject;
import net.runelite.api.coords.WorldPoint;

/**
 * Minimal runtime caching of bank locations discovered in the loaded scene.
 */
public final class BankDiscovery
{
    private static final String[] BANK_NAMES = {
        "bank booth", "bank chest", "bank", "deposit box", "bank deposit box", "bank counter"
    };

    private static final Set<WorldPoint> discoveredBanks = ConcurrentHashMap.newKeySet();
    private static final Set<WorldPoint> blacklistedBanks = ConcurrentHashMap.newKeySet();

    private static volatile WorldPoint lastTargetedBank;

    private BankDiscovery()
    {
    }

    public static void scanAndDiscoverBanks(TaskContext ctx)
    {
        try
        {
            Scene scene = ctx.client.getScene();
            if (scene == null)
            {
                return;
            }
            Tile[][][] tiles = scene.getTiles();
            for (Tile[][] plane : tiles)
            {
                for (Tile[] column : plane)
                {
                    for (Tile tile : column)
                    {
                        if (tile == null)
                        {
                            continue;
                        }
                        for (TileObject obj : tile.getGameObjects())
                        {
                            if (obj == null)
                            {
                                continue;
                            }
                            ObjectComposition comp = ctx.client.getObjectDefinition(obj.getId());
                            if (comp == null)
                            {
                                continue;
                            }
                            String name = comp.getName();
                            if (name == null)
                            {
                                continue;
                            }
                            if (!looksLikeBank(name.toLowerCase()) || isBankTable(comp))
                            {
                                continue;
                            }
                            WorldPoint wp = obj.getWorldLocation();
                            if (wp != null)
                            {
                                discoveredBanks.add(wp);
                            }
                        }
                    }
                }
            }
        }
        catch (Exception ignored)
        {
            // discovery is best effort only
        }
    }

    private static boolean looksLikeBank(String lower)
    {
        for (String candidate : BANK_NAMES)
        {
            if (lower.contains(candidate))
            {
                return true;
            }
        }
        return false;
    }

    private static boolean isBankTable(ObjectComposition comp)
    {
        if (comp.getName() == null)
        {
            return false;
        }
        String lower = comp.getName().toLowerCase();
        return lower.contains("table") && lower.contains("bank");
    }

    public static List<WorldPoint> getDiscoveredBanks()
    {
        List<WorldPoint> result = new ArrayList<>();
        for (WorldPoint wp : discoveredBanks)
        {
            if (!blacklistedBanks.contains(wp))
            {
                result.add(wp);
            }
        }
        return result;
    }

    public static WorldPoint getNearestDiscoveredBank(WorldPoint from)
    {
        if (from == null)
        {
            return null;
        }
        List<WorldPoint> banks = getDiscoveredBanks();
        WorldPoint best = null;
        int bestDist = Integer.MAX_VALUE;
        for (WorldPoint wp : banks)
        {
            int dist = from.distanceTo(wp);
            if (dist >= 0 && dist < bestDist)
            {
                bestDist = dist;
                best = wp;
            }
        }
        return best;
    }

    public static void setLastTargetedBank(WorldPoint location)
    {
        lastTargetedBank = location;
    }

    public static void blacklistLastTargetedBank()
    {
        if (lastTargetedBank != null)
        {
            blacklistedBanks.add(lastTargetedBank);
        }
    }

    public static boolean isBlacklisted(WorldPoint location)
    {
        return location != null && blacklistedBanks.contains(location);
    }
}
