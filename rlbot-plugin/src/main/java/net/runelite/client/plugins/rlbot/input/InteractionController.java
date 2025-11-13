package net.runelite.client.plugins.rlbot.input;

import java.awt.Point;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.MenuAction;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.plugins.rlbot.RLBotAgent;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * Orchestrates high level interactions: validated clicks and direct menu interactions.
 */
final class InteractionController
{
    private final RLBotLogger logger;
    private final Client client;
    private final ClientThread clientThread;
    private final MouseController mouse;
    private final TargetInspector inspector;
    private final OcclusionResolver occlusion;
    private RLBotAgent agent; // optional, for penalties

    InteractionController(RLBotLogger logger, Client client, ClientThread clientThread,
                          MouseController mouse, TargetInspector inspector, OcclusionResolver occlusion)
    {
        this.logger = logger;
        this.client = client;
        this.clientThread = clientThread;
        this.mouse = mouse;
        this.inspector = inspector;
        this.occlusion = occlusion;
    }

    void setAgent(RLBotAgent agent)
    {
        this.agent = agent;
    }

    boolean moveAndClickWithValidation(Point canvasPoint, String expectedAction)
    {
        mouse.smoothMouseMove(canvasPoint);
        // quick hover validation loop for a short window
        for (int i = 0; i < 4; i++)
        {
            if (inspector.validateTargetAtPoint(canvasPoint, expectedAction)) { break; }
        }

        GameObject targetObj = inspector.findObjectHullMatching(canvasPoint, expectedAction);
        if (occlusion.isOccludedByUI(canvasPoint))
        {
            occlusion.revealPointByCameraUI(canvasPoint);
        }
        if (targetObj != null && occlusion.isOccludedByGeometry(canvasPoint, targetObj))
        {
            occlusion.revealPointByCameraGeometry(canvasPoint, targetObj, 6);
        }

        if (!inspector.validateTargetAtPoint(canvasPoint, expectedAction))
        {
            if (agent != null) { agent.addExternalPenalty(0.5f); }
            return false;
        }
        mouse.click();
        return true;
    }

    boolean clickAtWithValidation(Point canvasPoint, String expectedAction)
    {
        GameObject targetObj = inspector.findObjectHullMatching(canvasPoint, expectedAction);
        if (occlusion.isOccludedByUI(canvasPoint))
        {
            occlusion.revealPointByCameraUI(canvasPoint);
        }
        if (targetObj != null && occlusion.isOccludedByGeometry(canvasPoint, targetObj))
        {
            occlusion.revealPointByCameraGeometry(canvasPoint, targetObj, 6);
        }
        if (!inspector.validateTargetAtPoint(canvasPoint, expectedAction))
        {
            if (agent != null) { agent.addExternalPenalty(0.5f); }
            return false;
        }
        mouse.clickAt(canvasPoint);
        return true;
    }

    boolean clickWithValidation(String expectedAction)
    {
        Point p = mouse.getLastCanvasMovePoint();
        if (p == null) { return false; }
        GameObject targetObj = inspector.findObjectHullMatching(p, expectedAction);
        if (occlusion.isOccludedByUI(p))
        {
            occlusion.revealPointByCameraUI(p);
        }
        if (targetObj != null && occlusion.isOccludedByGeometry(p, targetObj))
        {
            occlusion.revealPointByCameraGeometry(p, targetObj, 6);
        }
        if (!inspector.validateTargetAtPoint(p, expectedAction))
        {
            if (agent != null) { agent.addExternalPenalty(0.5f); }
            return false;
        }
        mouse.click();
        return true;
    }

    boolean interactWithGameObject(GameObject gameObject, String action)
    {
        if (gameObject == null)
        {
            logger.warn("[InteractionController] Cannot interact with null GameObject");
            return false;
        }

        java.util.concurrent.atomic.AtomicBoolean success = new java.util.concurrent.atomic.AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);

        clientThread.invoke(() -> {
            try
            {
                net.runelite.api.ObjectComposition composition = client.getObjectDefinition(gameObject.getId());
                if (composition == null) { logger.warn("[InteractionController] No composition for object id=" + gameObject.getId()); return; }
                String[] actions = composition.getActions();
                if (actions == null) { logger.warn("[InteractionController] No actions for object id=" + gameObject.getId()); return; }
                int actionIndex = -1;
                String desired = action == null ? "" : action.trim();
                String[] synonyms;
                if (desired.equalsIgnoreCase("Bank"))
                {
                    synonyms = new String[] {"Bank", "Use", "Open", "Open bank", "Use-quickly"};
                }
                else if (desired.equalsIgnoreCase("Chop down") || desired.equalsIgnoreCase("Chop"))
                {
                    synonyms = new String[] {"Chop down", "Chop"};
                }
                else
                {
                    synonyms = new String[] { desired };
                }
                for (String cand : synonyms)
                {
                    String cl = cand == null ? "" : cand.toLowerCase();
                    for (int i = 0; i < actions.length; i++)
                    {
                        String ai = actions[i];
                        if (ai == null) { continue; }
                        if (ai.toLowerCase().contains(cl)) { actionIndex = i; desired = ai; break; }
                    }
                    if (actionIndex != -1) { break; }
                }
                if (actionIndex == -1) { logger.warn("[InteractionController] Action not found for '" + action + "' on id=" + gameObject.getId()); return; }
                net.runelite.api.MenuAction menuAction;
                switch (actionIndex)
                {
                    case 0: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FIRST_OPTION; break;
                    case 1: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_SECOND_OPTION; break;
                    case 2: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_THIRD_OPTION; break;
                    case 3: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FOURTH_OPTION; break;
                    case 4: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FIFTH_OPTION; break;
                    default: return;
                }
                net.runelite.api.coords.WorldPoint wp = gameObject.getWorldLocation();
                net.runelite.api.coords.LocalPoint lp = net.runelite.api.coords.LocalPoint.fromWorld(client, wp);
                if (lp == null) { logger.warn("[InteractionController] LocalPoint null for object at " + wp); return; }
                int sceneX = lp.getSceneX();
                int sceneY = lp.getSceneY();
                client.menuAction(sceneX, sceneY, menuAction, gameObject.getId(), 0, desired, "");
                success.set(true);
            }
            catch (Exception e)
            {
                logger.warn("[InteractionController] interactWithGameObject failed: " + e.getMessage());
            }
            finally
            {
                latch.countDown();
            }
        });

        try
        {
            latch.await(150, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException ie)
        {
            Thread.currentThread().interrupt();
        }

        return success.get();
    }

    /**
     * Directly invoke a game object action by index (0..4), bypassing name matching.
     * Returns true if the menu action was queued on the client thread.
     */
    boolean interactWithGameObject(GameObject gameObject, int actionIndex, String label)
    {
        if (gameObject == null)
        {
            logger.warn("[InteractionController] Cannot interact (index) with null GameObject");
            return false;
        }
        if (actionIndex < 0 || actionIndex > 4)
        {
            logger.warn("[InteractionController] Invalid action index: " + actionIndex);
            return false;
        }

        java.util.concurrent.atomic.AtomicBoolean success = new java.util.concurrent.atomic.AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);

        clientThread.invoke(() -> {
            try
            {
                net.runelite.api.MenuAction menuAction;
                switch (actionIndex)
                {
                    case 0: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FIRST_OPTION; break;
                    case 1: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_SECOND_OPTION; break;
                    case 2: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_THIRD_OPTION; break;
                    case 3: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FOURTH_OPTION; break;
                    case 4: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FIFTH_OPTION; break;
                    default: return; // should be unreachable due to guard above
                }
                net.runelite.api.coords.WorldPoint wp = gameObject.getWorldLocation();
                net.runelite.api.coords.LocalPoint lp = net.runelite.api.coords.LocalPoint.fromWorld(client, wp);
                if (lp == null)
                {
                    logger.warn("[InteractionController] LocalPoint null for object at " + wp);
                    return;
                }
                int sceneX = lp.getSceneX();
                int sceneY = lp.getSceneY();
                String option = (label == null) ? "" : label;
                client.menuAction(sceneX, sceneY, menuAction, gameObject.getId(), 0, option, "");
                success.set(true);
            }
            catch (Exception e)
            {
                logger.warn("[InteractionController] interactWithGameObject(index) failed: " + e.getMessage());
            }
            finally
            {
                latch.countDown();
            }
        });

        try
        {
            latch.await(150, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException ie)
        {
            Thread.currentThread().interrupt();
        }
        return success.get();
    }
}
