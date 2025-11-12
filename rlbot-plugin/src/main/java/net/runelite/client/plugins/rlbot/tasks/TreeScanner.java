package net.runelite.client.plugins.rlbot.tasks;

import java.util.Map;
import net.runelite.api.ObjectComposition;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.api.TileObject;
import net.runelite.api.coords.WorldPoint;

final class TreeScanner {
    private TreeScanner() {}

    static void scanAndDiscoverTrees(TaskContext ctx, Map<WorldPoint, String> discoveredTrees) {
        try {
            Scene scene = ctx.client.getScene();
            if (scene == null) return;
            Tile[][][] tiles = scene.getTiles();
            for (Tile[][] plane : tiles) {
                for (Tile[] column : plane) {
                    for (Tile tile : column) {
                        if (tile == null) continue;
                        for (TileObject obj : tile.getGameObjects()) {
                            if (obj == null) continue;
                            ObjectComposition comp = ctx.client.getObjectDefinition(obj.getId());
                            if (comp == null) continue;
                            String name = comp.getName();
                            if (name == null) continue;
                            if (!TreeDiscovery.isChoppableName(name.toLowerCase()) || !hasChopAction(comp)) continue;
                            WorldPoint wp = obj.getWorldLocation();
                            if (wp != null) discoveredTrees.putIfAbsent(wp, name);
                        }
                    }
                }
            }
        } catch (Exception ignored) { }
    }

    private static boolean hasChopAction(ObjectComposition comp) {
        String[] actions = comp.getActions();
        if (actions == null) return false;
        for (String action : actions) {
            if (action != null) {
                String lower = action.toLowerCase();
                if (lower.contains("chop") || lower.contains("cut")) return true;
            }
        }
        return false;
    }
}

