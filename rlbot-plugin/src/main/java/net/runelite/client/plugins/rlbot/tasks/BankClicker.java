package net.runelite.client.plugins.rlbot.tasks;

import java.awt.Point;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import net.runelite.api.GameObject;
import net.runelite.api.coords.WorldPoint;

/**
 * Hover-first, staged bank clicker mirroring TreeClicker.
 */
public class BankClicker
{
    private static final String[] BANK_ACTIONS = {"Bank", "Open", "Use", "Deposit"};
    private static final long HOVER_SETTLE_MS = 80L;
    private static final long HOVER_TIMEOUT_MS = 2000L;

    public enum Result { NONE, STAGED, CLICKED, FAILED }

    private static Pending pending;

    public static Result clickBank(TaskContext context, GameObject bank)
    {
        if (bank == null)
        {
            clear();
            return Result.NONE;
        }
        if (pending != null && !pending.matches(bank))
        {
            clear();
        }
        if (pending != null)
        {
            return attemptClick(context, bank);
        }
        return stageHover(context, bank);
    }

    /**
     * Returns true if a hover has been staged and a follow-up click should be executed on the next tick.
     */
    public static boolean isPending()
    {
        return pending != null;
    }

    private static Result stageHover(TaskContext context, GameObject bank)
    {
        String action = resolveAction(context, bank);
        if (action == null)
        {
            return Result.NONE;
        }
        Point p = ObjectFinder.projectToClickablePoint(context, bank);
        if (p == null || p.x < 0 || p.y < 0)
        {
            p = ObjectFinder.projectToCanvas(context, bank);
        }
        if (p == null)
        {
            // If projection fails (behind camera / off-viewport), try direct menu interaction first.
            boolean interacted = context.input.interactWithGameObject(bank, action);
            if (interacted)
            {
                context.logger.info("[BankClicker] Direct interact succeeded for action '" + action + "' (no projection)");
                context.setBusyForMs(260);
                return Result.CLICKED;
            }
            // Nudge camera to reveal the object, then retry on next tick
            try
            {
                context.logger.debug("[BankClicker] Projection null; rotating camera slightly to reveal bank");
                // small left tilt sequence to bring booth into view
                context.input.rotateCameraLeftSmall();
                context.input.tiltCameraUpSmall();
                context.setBusyForMs(120);
            }
            catch (Exception ignored) {}
            return Result.NONE;
        }
        pending = new Pending(bank.getWorldLocation(), bank.getId(), new Point(p), action, System.currentTimeMillis());
        context.logger.info("[BankClicker] Hover staged at " + pending.world + " targeting " + p + " action=\"" + action + "\"");
        context.input.smoothMouseMove(p);
        context.setBusyForMs(120);
        return Result.STAGED;
    }

    private static Result attemptClick(TaskContext context, GameObject bank)
    {
        long now = System.currentTimeMillis();
        if (now - pending.startedMs < HOVER_SETTLE_MS)
        {
            return Result.STAGED;
        }
        if (now - pending.startedMs > HOVER_TIMEOUT_MS)
        {
            context.logger.debug("[BankClicker] Hover timed out for " + pending.world);
            clear();
            return Result.FAILED;
        }

        // Try validated click
        AtomicBoolean ok = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);
        final Point at = new Point(pending.canvas);
        final String label = pending.action;
        context.clientThread.invoke(() -> {
            try { ok.set(context.input.moveAndClickWithValidation(at, label)); }
            finally { latch.countDown(); }
        });
        try { latch.await(400, TimeUnit.MILLISECONDS); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
        if (ok.get())
        {
            clear();
            context.setBusyForMs(260);
            return Result.CLICKED;
        }

        // Fallback: menu interact
        AtomicBoolean interacted = new AtomicBoolean(false);
        CountDownLatch latch2 = new CountDownLatch(1);
        context.clientThread.invoke(() -> {
            try { interacted.set(context.input.interactWithGameObject(bank, label)); }
            finally { latch2.countDown(); }
        });
        try { latch2.await(300, TimeUnit.MILLISECONDS); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
        if (interacted.get())
        {
            clear();
            context.setBusyForMs(260);
            return Result.CLICKED;
        }

        // Last resort: blind click
        boolean blind = context.input.clickAt(at);
        clear();
        if (blind)
        {
            context.setBusyForMs(220);
            return Result.CLICKED;
        }
        return Result.FAILED;
    }

    private static String resolveAction(TaskContext context, GameObject bank)
    {
        try
        {
            net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(bank.getId());
            if (comp == null) return null;
            String[] actions = comp.getActions();
            if (actions == null) return null;
            for (String pref : BANK_ACTIONS)
            {
                for (String a : actions)
                {
                    if (a != null && (a.equalsIgnoreCase(pref) || a.toLowerCase().contains(pref.toLowerCase())))
                    {
                        return a;
                    }
                }
            }
        }
        catch (Exception ignored) {}
        return null;
    }

    private static void clear() { pending = null; }

    private static final class Pending
    {
        final WorldPoint world;
        final int id;
        final Point canvas;
        final String action;
        final long startedMs;
        Pending(WorldPoint w, int id, Point c, String a, long t) { this.world = w; this.id = id; this.canvas = c; this.action = a; this.startedMs = t; }
        boolean matches(GameObject bank)
        {
            WorldPoint cur = bank.getWorldLocation();
            return cur != null && cur.equals(world) && bank.getId() == id;
        }
    }
}
