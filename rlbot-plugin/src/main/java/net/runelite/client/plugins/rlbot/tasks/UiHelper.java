package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.widgets.Widget;

/**
 * Minimal UI helper utilities required for woodcutting/banking tasks.
 */
final class UiHelper
{
    private UiHelper()
    {
    }

    static void closeObstructions(TaskContext ctx)
    {
        try
        {
            ctx.input.pressKey(java.awt.event.KeyEvent.VK_ESCAPE);
            ctx.setBusyForMs(150);
        }
        catch (Exception ignored)
        {
        }
    }

    static boolean clickCloseIfVisible(TaskContext ctx)
    {
        try
        {
            Widget closeButton = ctx.client.getWidget(162, 33);
            if (closeButton != null && !closeButton.isHidden() && closeButton.getBounds() != null)
            {
                java.awt.Rectangle b = closeButton.getBounds();
                java.awt.Point p = new java.awt.Point(b.x + b.width / 2, b.y + b.height / 2);
                ctx.input.smoothMouseMove(p);
                ctx.input.clickAt(p);
                ctx.setBusyForMs(200);
                return true;
            }
        }
        catch (Exception ignored)
        {
        }
        return false;
    }

    static boolean closeBankMainById(TaskContext ctx, int rootWidgetId)
    {
        try
        {
            Widget bankRoot = ctx.client.getWidget(rootWidgetId);
            if (bankRoot == null)
            {
                return false;
            }
            Widget closeButton = bankRoot.getChild(11); // heuristic close button index
            if (closeButton != null && !closeButton.isHidden())
            {
                java.awt.Rectangle b = closeButton.getBounds();
                if (b != null && b.width > 0)
                {
                    java.awt.Point p = new java.awt.Point(b.x + b.width / 2, b.y + b.height / 2);
                    ctx.input.smoothMouseMove(p);
                    ctx.input.clickAt(p);
                    ctx.setBusyForMs(200);
                    return true;
                }
            }
        }
        catch (Exception ignored)
        {
        }
        return false;
    }
}
