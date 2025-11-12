package net.runelite.client.plugins.rlbot.tasks.context;

import net.runelite.api.widgets.Widget;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;

public final class WidgetLoggerUtil {
    public void logAllOpenWidgets(TaskContext ctx) {
        ctx.logger.info("=== WIDGET SCAN START ===");
        Widget[] widgetRoots = ctx.client.getWidgetRoots();
        if (widgetRoots == null) {
            ctx.logger.warn("No widget roots found");
            return;
        }
        int totalWidgets = 0;
        int visibleWidgets = 0;
        for (int rootId = 0; rootId < widgetRoots.length; rootId++) {
            Widget rootWidget = widgetRoots[rootId];
            if (rootWidget == null) continue;
            totalWidgets++;
            if (!rootWidget.isHidden()) {
                visibleWidgets++;
                logWidgetDetails(ctx, rootWidget, rootId, 0);
            }
        }
        ctx.logger.info("=== WIDGET SCAN SUMMARY ===");
        ctx.logger.info("Total root widgets: " + totalWidgets);
        ctx.logger.info("Visible root widgets: " + visibleWidgets);
        ctx.logger.info("=== WIDGET SCAN END ===");
    }

    public void logWidgetsBFS(TaskContext ctx) {
        ctx.logger.info("=== WIDGET BFS SCAN START ===");
        Widget[] widgetRoots = ctx.client.getWidgetRoots();
        if (widgetRoots == null) {
            ctx.logger.warn("No widget roots found");
            return;
        }
        int totalVisible = 0;
        for (int gid = 0; gid < widgetRoots.length; gid++) {
            Widget root = widgetRoots[gid];
            if (root == null) continue;
            java.util.ArrayDeque<Widget> dq = new java.util.ArrayDeque<>();
            dq.add(root);
            while (!dq.isEmpty()) {
                Widget w = dq.removeFirst();
                if (w == null) continue;
                if (!w.isHidden()) {
                    totalVisible++;
                    logWidgetDetails(ctx, w, gid, w.getId());
                }
                Widget[] children = w.getChildren();
                if (children != null) {
                    for (Widget ch : children) {
                        if (ch != null) dq.addLast(ch);
                    }
                }
            }
        }
        ctx.logger.info("Visible widgets total: " + totalVisible);
        ctx.logger.info("=== WIDGET BFS SCAN END ===");
    }

    public void logSpecificWidgetGroups(TaskContext ctx, int... groupIds) {
        ctx.logger.info("=== SPECIFIC WIDGET GROUPS SCAN START ===");
        Widget[] widgetRoots = ctx.client.getWidgetRoots();
        if (widgetRoots == null) {
            ctx.logger.warn("No widget roots found");
            return;
        }
        for (int groupId : groupIds) {
            if (groupId < 0 || groupId >= widgetRoots.length) {
                ctx.logger.warn("Invalid group ID: " + groupId);
                continue;
            }
            Widget rootWidget = widgetRoots[groupId];
            if (rootWidget == null) {
                ctx.logger.info("Group " + groupId + ": null");
                continue;
            }
            ctx.logger.info("=== Group " + groupId + " ===");
            int visibleCount = 0;
            if (!rootWidget.isHidden()) {
                visibleCount++;
                logWidgetDetails(ctx, rootWidget, groupId, 0);
            }
            ctx.logger.info("Group " + groupId + " visible widgets: " + visibleCount);
        }
        ctx.logger.info("=== SPECIFIC WIDGET GROUPS SCAN END ===");
    }

    private void logWidgetDetails(TaskContext ctx, Widget widget, int groupId, int widgetId) {
        StringBuilder sb = new StringBuilder();
        sb.append("  Widget[").append(groupId).append(",").append(widgetId).append("]: ");
        sb.append("hidden=").append(widget.isHidden());
        sb.append(", visible=").append(!widget.isHidden());
        sb.append(", x=").append(widget.getRelativeX());
        sb.append(", y=").append(widget.getRelativeY());
        sb.append(", width=").append(widget.getWidth());
        sb.append(", height=").append(widget.getHeight());
        String text = widget.getText();
        if (text != null && !text.isEmpty()) {
            sb.append(", text=\"").append(text).append("\"");
        }
        String name = widget.getName();
        if (name != null && !name.isEmpty()) {
            sb.append(", name=\"").append(name).append("\"");
        }
        String[] actions = widget.getActions();
        if (actions != null && actions.length > 0) {
            sb.append(", actions=[");
            for (int i = 0; i < actions.length; i++) {
                if (actions[i] != null && !actions[i].isEmpty()) {
                    if (i > 0) sb.append(", ");
                    sb.append("\"").append(actions[i]).append("\"");
                }
            }
            sb.append("]");
        }
        int itemId = widget.getItemId();
        if (itemId != -1) {
            sb.append(", itemId=").append(itemId);
        }
        int itemQuantity = widget.getItemQuantity();
        if (itemQuantity > 0) {
            sb.append(", quantity=").append(itemQuantity);
        }
        int modelId = widget.getModelId();
        if (modelId != -1) {
            sb.append(", modelId=").append(modelId);
        }
        int spriteId = widget.getSpriteId();
        if (spriteId != -1) {
            sb.append(", spriteId=").append(spriteId);
        }
        ctx.logger.info(sb.toString());
    }
}

