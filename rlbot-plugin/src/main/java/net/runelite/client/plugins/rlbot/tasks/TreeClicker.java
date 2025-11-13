package net.runelite.client.plugins.rlbot.tasks;

import java.awt.Point;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import net.runelite.api.GameObject;
import net.runelite.api.coords.WorldPoint;

/**
 * Handles interacting with trees using a hover-first, multi-tick flow.
 */
public class TreeClicker
{
    private static final long HOVER_SETTLE_MS = 80L;
    private static final long HOVER_TIMEOUT_MS = 2000L;

    private static PendingHover pendingHover;

    public enum Result
    {
        NONE,
        STAGED,
        CLICKED,
        FAILED
    }

    public static Result clickTree(TaskContext context, GameObject tree)
    {
        if (tree == null)
        {
            clearPending();
            return Result.NONE;
        }

        if (pendingHover != null && !pendingHover.matches(tree))
        {
            clearPending();
        }

        if (pendingHover != null)
        {
            return attemptClick(context, tree);
        }

        return stageHover(context, tree);
    }

    private static Result stageHover(TaskContext context, GameObject tree)
    {
        Point clickPoint = ObjectFinder.projectToClickablePoint(context, tree);
        if (clickPoint == null || clickPoint.x < 0 || clickPoint.y < 0)
        {
            clickPoint = ObjectFinder.projectToCanvas(context, tree);
        }
        if (clickPoint == null)
        {
            context.logger.warn("[TreeClicker] Cannot project tree to canvas");
            return Result.NONE;
        }

        String action = resolveTreeAction(context, tree);
        Point targetCanvasPoint = new Point(clickPoint);
        pendingHover = new PendingHover(tree.getWorldLocation(), tree.getId(), targetCanvasPoint, action, System.currentTimeMillis());

        context.logger.info("[TreeClicker] Hover staged at " + pendingHover.worldPoint + " targeting " + targetCanvasPoint + " action=\"" + action + "\"");
        context.input.smoothMouseMove(targetCanvasPoint);
        context.setBusyForMs(120);
        return Result.STAGED;
    }

    private static Result attemptClick(TaskContext context, GameObject tree)
    {
        if (pendingHover == null)
        {
            return Result.NONE;
        }

        long now = System.currentTimeMillis();
        if (now - pendingHover.startedMs < HOVER_SETTLE_MS)
        {
            context.logger.debug("[TreeClicker] Waiting for hover settle (" + (now - pendingHover.startedMs) + "ms)");
            return Result.STAGED;
        }

        if (now - pendingHover.startedMs > HOVER_TIMEOUT_MS)
        {
            context.logger.debug("[TreeClicker] Hover timed out for " + pendingHover.worldPoint);
            clearPending();
            return Result.FAILED;
        }

        boolean clicked = context.input.moveAndClickWithValidation(pendingHover.canvasPoint, pendingHover.actionLabel);
        if (clicked)
        {
            clearPending();
            context.setBusyForMs(220);
            return Result.CLICKED;
        }

        context.logger.warn("[TreeClicker] Validated click failed; falling back to interaction for '" + pendingHover.actionLabel + "'");
        AtomicBoolean interactSuccess = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);
        context.clientThread.invoke(() -> {
            try
            {
                interactSuccess.set(context.input.interactWithGameObject(tree, pendingHover.actionLabel));
            }
            finally
            {
                latch.countDown();
            }
        });
        try
        {
            latch.await(250, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException interrupted)
        {
            Thread.currentThread().interrupt();
        }

        if (interactSuccess.get())
        {
            clearPending();
            context.setBusyForMs(220);
            return Result.CLICKED;
        }

        context.logger.warn("[TreeClicker] Interaction fallback failed; executing blind click");
        boolean blind = context.input.clickAt(pendingHover.canvasPoint);
        clearPending();
        if (blind)
        {
            context.setBusyForMs(200);
            return Result.CLICKED;
        }

        context.logger.error("[TreeClicker] All click attempts failed");
        return Result.FAILED;
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

            String[] prefs = new String[] {"Chop down", "Chop", "Cut down", "Cut"};
            for (String pref : prefs)
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

    private static void clearPending()
    {
        pendingHover = null;
    }

    private static final class PendingHover
    {
        private final WorldPoint worldPoint;
        private final int treeId;
        private final Point canvasPoint;
        private final String actionLabel;
        private final long startedMs;

        private PendingHover(WorldPoint worldPoint, int treeId, Point canvasPoint, String actionLabel, long startedMs)
        {
            this.worldPoint = worldPoint;
            this.treeId = treeId;
            this.canvasPoint = canvasPoint;
            this.actionLabel = actionLabel;
            this.startedMs = startedMs;
        }

        private boolean matches(GameObject tree)
        {
            WorldPoint current = tree.getWorldLocation();
            return current != null && current.equals(worldPoint) && tree.getId() == treeId;
        }
    }
}
