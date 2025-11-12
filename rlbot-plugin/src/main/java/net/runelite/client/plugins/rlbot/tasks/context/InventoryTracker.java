package net.runelite.client.plugins.rlbot.tasks.context;

import net.runelite.api.InventoryID;
import net.runelite.api.Item;
import net.runelite.api.ItemContainer;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;

public final class InventoryTracker {
    private Integer lastLoggedFreeSlotsValue = null;
    private long lastInventoryLogMs = 0L;
    private Boolean lastNearFullState = null;
    private Boolean lastFullState = null;

    public int getInventoryFreeSlots(TaskContext ctx) {
        try {
            ItemContainer inv = ctx.client.getItemContainer(InventoryID.INVENTORY);
            if (inv == null) {
                long now = System.currentTimeMillis();
                if (now - lastInventoryLogMs > 30000L) {
                    ctx.logger.debug("[Context] Inventory container is null; assuming empty (28 free slots)");
                    lastInventoryLogMs = now;
                }
                return 28;
            }
            Item[] items = inv.getItems();
            if (items == null) {
                long now = System.currentTimeMillis();
                if (now - lastInventoryLogMs > 30000L) {
                    ctx.logger.debug("[Context] Inventory items array is null; assuming empty (28 free slots)");
                    lastInventoryLogMs = now;
                }
                return 28;
            }

            int occupied = 0;
            for (Item it : items) {
                if (it != null) {
                    int id = it.getId();
                    if (id > 0 && id != -1) {
                        occupied++;
                    }
                }
            }

            int free = 28 - occupied;
            long now = System.currentTimeMillis();
            boolean shouldLog = (lastLoggedFreeSlotsValue == null)
                || (free != lastLoggedFreeSlotsValue)
                || (now - lastInventoryLogMs > 60000L);
            if (shouldLog) {
                ctx.logger.debug("[Context] Inventory: occupied=" + occupied + ", free=" + free);
                lastLoggedFreeSlotsValue = free;
                lastInventoryLogMs = now;
            }
            return free;
        } catch (Exception e) {
            ctx.logger.error("[Context] Error reading inventory: " + e.getMessage());
            return 28;
        }
    }

    public boolean isInventoryNearFull(TaskContext ctx) {
        int freeSlots = getInventoryFreeSlots(ctx);
        boolean nearFull = freeSlots <= 5;
        if (lastNearFullState == null || lastNearFullState != nearFull) {
            ctx.logger.debug("[Context] isInventoryNearFull changed: freeSlots=" + freeSlots + ", nearFull=" + nearFull);
            lastNearFullState = nearFull;
        }
        return nearFull;
    }

    public boolean isInventoryFull(TaskContext ctx) {
        int freeSlots = getInventoryFreeSlots(ctx);
        boolean full = freeSlots <= 0;
        if (lastFullState == null || lastFullState != full) {
            ctx.logger.debug("[Context] isInventoryFull changed: freeSlots=" + freeSlots + ", full=" + full);
            lastFullState = full;
        }
        return full;
    }
}

