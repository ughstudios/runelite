package net.runelite.client.plugins.rlbot.tasks;

import java.awt.Point;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import net.runelite.api.GameObject;

/**
 * Handles interacting with trees using a hover-first, asynchronous click.
 */
public class TreeClicker
{
    public enum Result
    {
        CLICKED,
        FAILED
    }

    private static final String[] ACTION_PREFERENCES = {"Chop down", "Chop", "Cut down", "Cut"};

    public static Result clickTree(TaskContext context, GameObject tree)
    {
        if (tree == null)
        {
            context.logger.info("[TreeClicker] Tree is null, cannot click");
            return Result.FAILED;
        }

        Point clickPoint = ObjectFinder.projectToClickablePoint(context, tree);
        if (clickPoint == null || clickPoint.x < 0 || clickPoint.y < 0)
        {
            clickPoint = ObjectFinder.projectToCanvas(context, tree);
        }
        if (clickPoint == null)
        {
            context.logger.warn("[TreeClicker] Cannot project tree to canvas");
            return Result.FAILED;
        }

        String action = resolveTreeAction(context, tree);
        context.logger.info("[TreeClicker] Hovering toward " + clickPoint + " for action=\"" + action + "\"");
        context.input.smoothMouseMove(clickPoint);
        context.setBusyForMs(120);

        if (!scheduleClick(context, tree, new Point(clickPoint), action))
        {
            context.logger.error("[TreeClicker] Unable to schedule click thread");
            return Result.FAILED;
        }

        return Result.CLICKED;
    }

    private static boolean scheduleClick(TaskContext context, GameObject tree, Point canvasPoint, String actionLabel)
    {
        Thread clicker = new Thread(() -> {
            try
            {
                Thread.sleep(120);
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }

            if (tryMoveAndClick(context, canvasPoint, actionLabel))
            {
                return;
            }

            if (tryMenuInteract(context, tree, actionLabel))
            {
                return;
            }

            context.logger.warn("[TreeClicker] Falling back to blind click at " + canvasPoint);
            context.input.clickAt(canvasPoint);
        }, "rlbot-tree-clicker");
        clicker.setDaemon(true);
        try
        {
            clicker.start();
        }
        catch (Exception e)
        {
            context.logger.error("[TreeClicker] Failed to start click thread: " + e.getMessage());
            return false;
        }
        return true;
    }

    private static boolean tryMoveAndClick(TaskContext context, Point canvasPoint, String actionLabel)
    {
        AtomicBoolean clicked = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);
        context.clientThread.invoke(() -> {
            try
            {
                clicked.set(context.input.moveAndClickWithValidation(canvasPoint, actionLabel));
            }
            finally
            {
                latch.countDown();
            }
        });
        try
        {
            latch.await(400, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }
        if (clicked.get())
        {
            context.setBusyForMs(220);
            context.logger.info("[TreeClicker] Validated click succeeded");
        }
        return clicked.get();
    }

    private static boolean tryMenuInteract(TaskContext context, GameObject tree, String actionLabel)
    {
        AtomicBoolean interacted = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);
        context.clientThread.invoke(() -> {
            try
            {
                interacted.set(context.input.interactWithGameObject(tree, actionLabel));
            }
            finally
            {
                latch.countDown();
            }
        });
        try
        {
            latch.await(300, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }
        if (interacted.get())
        {
            context.setBusyForMs(220);
            context.logger.info("[TreeClicker] Menu interaction succeeded for action '" + actionLabel + "'");
        }
        return interacted.get();
    }

    private static String resolveTreeAction(TaskContext context, GameObject tree)
    {
        try
        {
            net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(tree.getId());
            if (comp == null)
            {
                return "Chop down";
            }
            String[] actions = comp.getActions();
            if (actions == null)
            {
                return "Chop down";
            }

            for (String pref : ACTION_PREFERENCES)
            {
                for (String action : actions)
                {
                    if (action != null && action.equalsIgnoreCase(pref))
                    {
                        return action;
                    }
                }
            }
        }
        catch (Exception ignored)
        {
        }
        return "Chop down";
    }

    public static boolean adjustCameraForTree(TaskContext context, GameObject tree)
    {
        return false;
    }
}
