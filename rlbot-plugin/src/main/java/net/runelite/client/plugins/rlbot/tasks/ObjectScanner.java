package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.ObjectComposition;
import net.runelite.api.Player;
import net.runelite.api.Scene;
import net.runelite.api.Tile;
import net.runelite.api.coords.WorldPoint;

final class ObjectScanner {
    private ObjectScanner() {}

    static GameObject nearestByAction(TaskContext ctx, String requiredAction) {
        Player me = ctx.client.getLocalPlayer();
        if (me == null) return null;
        WorldPoint myWp = me.getWorldLocation();
        if (myWp == null) return null;
        GameObject best = null; int bestDist = Integer.MAX_VALUE;
        Scene scene = ctx.client.getScene();
        if (scene == null) return null;
        Tile[][][] tiles = scene.getTiles();
        for (Tile[][] plane : tiles) for (Tile[] col : plane) for (Tile t : col) {
            if (t == null) continue;
            for (GameObject go : t.getGameObjects()) {
                if (go == null) continue;
                ObjectComposition comp = ctx.client.getObjectDefinition(go.getId());
                if (comp == null || comp.getActions() == null) continue;
                if (ObjectFinder.isBankTable(comp)) continue;
                boolean has = false;
                for (String a : comp.getActions()) { if (a != null && a.equals(requiredAction)) { has = true; break; } }
                if (!has) continue;
                if (ObjectFinder.projectToCanvas(ctx, go) == null) continue;
                int d = t.getWorldLocation().distanceTo(myWp);
                if (d >= 0 && d < bestDist) { bestDist = d; best = go; }
            }
        }
        return best;
    }

    static GameObject nearestByNames(TaskContext ctx, String[] nameSubstrings, String requiredAction) {
        Client client = ctx.client; Player me = client.getLocalPlayer(); if (me == null) return null;
        WorldPoint myWp = me.getWorldLocation(); Scene scene = client.getScene(); Tile[][][] tiles = scene.getTiles();
        int bestDist = Integer.MAX_VALUE; GameObject best = null;
        for (Tile[][] plane : tiles) for (Tile[] col : plane) for (Tile tile : col) {
            if (tile == null) continue;
            for (GameObject go : tile.getGameObjects()) {
                if (go == null) continue;
                ObjectComposition comp = client.getObjectDefinition(go.getId()); if (comp == null) continue;
                String name = comp.getName(); if (name == null) continue; String lower = name.toLowerCase();
                boolean nameMatch = false; for (String sub : nameSubstrings) { if (sub != null && lower.contains(sub.toLowerCase())) { nameMatch = true; break; } }
                if (!nameMatch) continue;
                if (requiredAction != null) {
                    boolean ok = false; String[] acts = comp.getActions(); if (acts != null) for (String a : acts) { if (a != null && (a.equalsIgnoreCase(requiredAction) || a.toLowerCase().contains(requiredAction.toLowerCase()))) { ok = true; break; } }
                    if (!ok) continue;
                }
                int d = tile.getWorldLocation().distanceTo(myWp); if (d >= 0 && d < bestDist) { bestDist = d; best = go; }
            }
        }
        return best;
    }

    static GameObject nearestBankInteractable(TaskContext ctx) {
        Client client = ctx.client; Player me = client.getLocalPlayer(); if (me == null) return null;
        WorldPoint myWp = me.getWorldLocation(); Scene scene = client.getScene(); Tile[][][] tiles = scene.getTiles();
        int bestDist = Integer.MAX_VALUE; GameObject best = null;
        for (Tile[][] plane : tiles) for (Tile[] col : plane) for (Tile tile : col) {
            if (tile == null) continue;
            for (GameObject go : tile.getGameObjects()) {
                if (go == null) continue;
                ObjectComposition comp = client.getObjectDefinition(go.getId()); if (comp == null) continue;
                String name = comp.getName(); if (name == null) continue; String lower = name.toLowerCase();
                boolean looksBank = lower.contains("bank booth") || lower.contains("bank chest") || lower.contains("bank deposit") || lower.equals("bank") || lower.contains("deposit box");
                if (!looksBank) continue; if (ObjectFinder.isBankTable(comp)) continue;
                boolean actionOk = false; String[] actions = comp.getActions(); if (actions != null) for (String a : actions) { if (a == null) continue; String al = a.toLowerCase(); if (al.contains("bank") || al.contains("use") || al.contains("open") || al.contains("deposit")) { actionOk = true; break; } }
                if (!actionOk) continue;
                int d = tile.getWorldLocation().distanceTo(myWp); if (d >= 0 && d < bestDist) { bestDist = d; best = go; }
            }
        }
        return best;
    }
}

