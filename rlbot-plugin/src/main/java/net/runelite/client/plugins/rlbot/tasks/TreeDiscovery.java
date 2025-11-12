package net.runelite.client.plugins.rlbot.tasks;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import net.runelite.api.ObjectComposition;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.api.TileObject;
import net.runelite.api.coords.WorldPoint;

/**
 * Runtime-only helper for discovering choppable trees and tracking temporary
 * depletion. All data lives in-memory to keep the plugin lightweight.
 */
public final class TreeDiscovery
{
    private static final long DEPLETION_COOLDOWN_MS = 5_000L;

    private static final Map<WorldPoint, String> discoveredTrees = new ConcurrentHashMap<>();
    private static final Map<WorldPoint, Long> depletedUntil = new ConcurrentHashMap<>();
    private static final Map<WorldPoint, Integer> cameraAdjustmentAttempts = new ConcurrentHashMap<>();

    private static volatile WorldPoint lastTargetedTree;

    private TreeDiscovery()
    {
    }

    public static void scanAndDiscoverTrees(TaskContext ctx)
    {
        TreeScanner.scanAndDiscoverTrees(ctx, discoveredTrees);
        cleanupExpiredDepletedTrees();
    }

    static boolean isChoppableName(String lower)
    {
        return lower.contains("tree")
            || lower.contains("oak")
            || lower.contains("willow")
            || lower.contains("maple")
            || lower.contains("yew")
            || lower.contains("magic")
            || lower.contains("teak")
            || lower.contains("mahogany");
    }

    // chop action detection moved to TreeScanner

    public static List<WorldPoint> getDiscoveredTrees()
    {
        return new ArrayList<>(discoveredTrees.keySet());
    }

    public static List<WorldPoint> getAvailableTrees()
    {
        cleanupExpiredDepletedTrees();
        long now = System.currentTimeMillis();
        List<WorldPoint> result = new ArrayList<>();
        for (WorldPoint wp : discoveredTrees.keySet())
        {
            long until = depletedUntil.getOrDefault(wp, 0L);
            if (until <= now)
            {
                result.add(wp);
            }
        }
        return result;
    }

    public static List<WorldPoint> getBestAvailableTreesForLevel(int woodcuttingLevel)
    {
        List<WorldPoint> available = getAvailableTrees();
        if (available.isEmpty())
        {
            return available;
        }
        String[] allowed = allowedTreeNamesForLevel(woodcuttingLevel);
        List<WorldPoint> filtered = new ArrayList<>();
        for (WorldPoint wp : available)
        {
            String name = discoveredTrees.getOrDefault(wp, "").toLowerCase();
            for (String allow : allowed)
            {
                if (name.contains(allow))
                {
                    filtered.add(wp);
                    break;
                }
            }
        }
        return filtered.isEmpty() ? available : filtered;
    }

    public static WorldPoint getNearestDiscoveredTree(WorldPoint from)
    {
        if (from == null)
        {
            return null;
        }
        List<WorldPoint> available = getAvailableTrees();
        WorldPoint best = null;
        int bestDist = Integer.MAX_VALUE;
        for (WorldPoint wp : available)
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

    public static void markDepleted(WorldPoint location)
    {
        if (location != null)
        {
            depletedUntil.put(location, System.currentTimeMillis() + DEPLETION_COOLDOWN_MS);
        }
    }

    public static boolean isDepleted(WorldPoint location)
    {
        if (location == null)
        {
            return false;
        }
        cleanupExpiredDepletedTrees();
        return depletedUntil.getOrDefault(location, 0L) > System.currentTimeMillis();
    }

    public static String[] allowedTreeNamesForLevel(int woodcuttingLevel)
    {
        List<String> names = new ArrayList<>();
        names.add("tree");
        if (woodcuttingLevel >= 15)
        {
            names.add("oak");
        }
        if (woodcuttingLevel >= 30)
        {
            names.add("willow");
        }
        if (woodcuttingLevel >= 35)
        {
            names.add("teak");
        }
        if (woodcuttingLevel >= 45)
        {
            names.add("maple");
        }
        if (woodcuttingLevel >= 50)
        {
            names.add("mahogany");
        }
        if (woodcuttingLevel >= 60)
        {
            names.add("yew");
        }
        if (woodcuttingLevel >= 75)
        {
            names.add("magic");
        }
        return names.toArray(new String[0]);
    }

    public static boolean isTreeAllowedForLevel(String treeName, int woodcuttingLevel)
    {
        if (treeName == null)
        {
            return false;
        }
        String lower = treeName.toLowerCase();
        for (String allow : allowedTreeNamesForLevel(woodcuttingLevel))
        {
            if (lower.contains(allow))
            {
                return true;
            }
        }
        return false;
    }

    public static void setLastTargetedTree(WorldPoint location)
    {
        lastTargetedTree = location;
    }

    public static void blacklistLastTargetedTree()
    {
        if (lastTargetedTree != null)
        {
            depletedUntil.put(lastTargetedTree, System.currentTimeMillis() + 120_000L);
        }
    }

    public static int getCameraAdjustmentAttempts(WorldPoint location)
    {
        if (location == null)
        {
            return 0;
        }
        return cameraAdjustmentAttempts.getOrDefault(location, 0);
    }

    public static void incrementCameraAdjustmentAttempts(WorldPoint location)
    {
        if (location != null)
        {
            cameraAdjustmentAttempts.merge(location, 1, Integer::sum);
        }
    }

    public static void resetCameraAdjustmentAttempts(WorldPoint location)
    {
        if (location != null)
        {
            cameraAdjustmentAttempts.remove(location);
        }
    }

    private static void cleanupExpiredDepletedTrees()
    {
        long now = System.currentTimeMillis();
        depletedUntil.entrySet().removeIf(e -> e.getValue() <= now);
    }
}
