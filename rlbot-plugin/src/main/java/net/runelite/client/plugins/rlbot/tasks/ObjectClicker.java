package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.GameObject;
import net.runelite.api.ObjectComposition;
import java.awt.Point;

/**
 * Generalized object clicking system that provides consistent behavior
 * for clicking on various types of objects (trees, banks, etc.).
 * 
 * This class extracts the successful clicking logic from ChopNearestTreeTask
 * and makes it reusable for all object types.
 */
public final class ObjectClicker {
    private ObjectClicker() {}
    
    /**
     * Configuration for different object types and their expected actions.
     */
    public static class ObjectType {
        public final String[] namePatterns;
        public final String[] preferredActions;
        public final String[] fallbackActions;
        public final boolean useConvexHull;
        public final int clickOffsetY;
        
        public ObjectType(String[] namePatterns, String[] preferredActions, String[] fallbackActions, 
                         boolean useConvexHull, int clickOffsetY) {
            this.namePatterns = namePatterns;
            this.preferredActions = preferredActions;
            this.fallbackActions = fallbackActions;
            this.useConvexHull = useConvexHull;
            this.clickOffsetY = clickOffsetY;
        }
    }
    
    // Predefined object types
    public static final ObjectType TREE = new ObjectType(
        new String[]{"tree", "oak", "willow", "yew", "maple"},
        new String[]{"Chop down", "Chop", "Cut", "Cut down"},
        new String[]{"Chop", "Cut"},
        true, // Use convex hull
        0     // No offset needed for trees
    );
    
    public static final ObjectType BANK = new ObjectType(
        new String[]{"bank booth", "bank chest", "bank", "deposit box", "bank deposit box"},
        new String[]{"Bank", "Deposit", "Use", "Open", "Examine"}, // Include Examine as fallback for closed booths
        new String[]{"Bank", "Deposit", "Use", "Open", "Examine"}, // Same as preferred
        true, // Use convex hull
        -14   // Click higher to avoid NPC heads/walls
    );
    
    /**
     * Click on a GameObject using the specified object type configuration.
     * 
     * @param context Task context
     * @param gameObject The GameObject to click
     * @param objectType The object type configuration
     * @return true if click was successful, false otherwise
     */
    public static boolean clickObject(TaskContext context, GameObject gameObject, ObjectType objectType) {
        if (context == null || gameObject == null || objectType == null) {
            return false;
        }
        
        try {
            // Get object composition to determine the correct action
            ObjectComposition comp = context.client.getObjectDefinition(gameObject.getId());
            if (comp == null) {
                context.logger.warn("[ObjectClicker] No object composition for ID: " + gameObject.getId());
                return false;
            }
            
            // Find the best action for this object
            String bestAction = findBestAction(comp, objectType);
            
            // Always log available actions for debugging
            StringBuilder actionLog = new StringBuilder();
            String[] actions = comp.getActions();
            if (actions != null) {
                for (int i = 0; i < actions.length; i++) {
                    if (actions[i] != null) {
                        if (i > 0) actionLog.append(", ");
                        actionLog.append(actions[i]);
                    }
                }
            }
            context.logger.info("[ObjectClicker] Object: " + comp.getName() + " (ID: " + gameObject.getId() + ") has actions: [" + actionLog.toString() + "]");
            context.logger.info("[ObjectClicker] Looking for actions: " + java.util.Arrays.toString(objectType.preferredActions));
            context.logger.info("[ObjectClicker] Best action found: " + (bestAction != null ? "'" + bestAction + "'" : "none"));
            
            if (bestAction == null) {
                context.logger.warn("[ObjectClicker] No suitable action found for object: " + comp.getName() + 
                                  " (ID: " + gameObject.getId() + "). Available actions: [" + actionLog.toString() + "]");
                return false;
            }
            
            context.logger.info("[ObjectClicker] Using action '" + bestAction + "' for object: " + comp.getName() + " (ID: " + gameObject.getId() + ")");

            // Attempt a direct menu action first to guarantee the intended action (avoid Walk-here yellow click)
            try {
                int actionIdx = -1;
                String[] acts = comp.getActions();
                if (acts != null) {
                    for (int i = 0; i < acts.length; i++) {
                        String a = acts[i];
                        if (a == null) continue;
                        String al = a.toLowerCase();
                        String bl = bestAction.toLowerCase();
                        // Prefer exact match; otherwise allow contains in either direction to handle variations like "Chop down" vs "Chop"
                        if (a.equals(bestAction) || a.equalsIgnoreCase(bestAction) || al.contains(bl) || bl.contains(al)) {
                            actionIdx = i; break;
                        }
                    }
                }
                if (actionIdx >= 0 && actionIdx <= 4) {
                    net.runelite.api.coords.LocalPoint lp = gameObject.getLocalLocation();
                    int sceneX = lp != null ? lp.getSceneX() : gameObject.getSceneMinLocation().getX();
                    int sceneY = lp != null ? lp.getSceneY() : gameObject.getSceneMinLocation().getY();
                    net.runelite.api.MenuAction opcode =
                        actionIdx == 0 ? net.runelite.api.MenuAction.GAME_OBJECT_FIRST_OPTION :
                        actionIdx == 1 ? net.runelite.api.MenuAction.GAME_OBJECT_SECOND_OPTION :
                        actionIdx == 2 ? net.runelite.api.MenuAction.GAME_OBJECT_THIRD_OPTION :
                        actionIdx == 3 ? net.runelite.api.MenuAction.GAME_OBJECT_FOURTH_OPTION :
                                         net.runelite.api.MenuAction.GAME_OBJECT_FIFTH_OPTION;

                    // Normalize the option we send to validation: use the action string from comp
                    final String option = acts[actionIdx] != null ? acts[actionIdx] : bestAction;
                    final String target = comp.getName();
                    final int objId = gameObject.getId();
                    final int param0 = sceneX; // scene X
                    final int param1 = sceneY; // scene Y
                    context.logger.info("[ObjectClicker] Invoking menuAction op=" + opcode + " at (" + param0 + "," + param1 + ") id=" + objId);
                    context.clientThread.invoke(() -> {
                        try {
                            // Signature: menuAction(int param0, int param1, MenuAction action, int id, int itemId, String option, String target)
                            context.client.menuAction(param0, param1, opcode, objId, -1, option, target);
                        } catch (Exception e) {
                            context.logger.warn("[ObjectClicker] menuAction failed: " + e.getMessage());
                        }
                    });
                    // Prefer reliable direct interaction, but also proceed to a validated canvas click
                    // to ensure hover/menu state is synchronized.
                    context.setBusyForMs(getBusyTimeForObjectType(objectType));
                    context.logger.info("[ObjectClicker] Menu action invoked; proceeding to validated canvas click to settle hover");
                } else {
                    // If we couldn't resolve the action index here, fall back to input handler's interaction
                    // which has its own synonym and index resolution. This avoids unreliable canvas clicks.
                    try {
                        context.logger.info("[ObjectClicker] Falling back to input.interactWithGameObject for action '" + bestAction + "'");
                        context.input.interactWithGameObject(gameObject, bestAction);
                        context.setBusyForMs(getBusyTimeForObjectType(objectType));
                        context.logger.info("[ObjectClicker] Fallback interact invoked; proceeding to validated canvas click to settle hover");
                    } catch (Exception ex) {
                        context.logger.warn("[ObjectClicker] Fallback interact failed: " + ex.getMessage());
                    }
                }
            } catch (Exception e) {
                context.logger.warn("[ObjectClicker] Direct menu action attempt failed: " + e.getMessage());
            }
            
            // Project object to canvas
            Point clickPoint = projectObjectToClickPoint(context, gameObject, objectType);
            if (clickPoint == null) {
                context.logger.warn("[ObjectClicker] Could not project object to canvas - running diagnostics");
                ObjectFinder.projectToCanvasWithDiagnostics(context, gameObject, true);
                return false;
            }
            
            // Log projection details for debugging
            context.logger.info("[ObjectClicker] Projected " + comp.getName() + " to canvas point: (" + clickPoint.x + "," + clickPoint.y + ")");
            context.logger.info("[ObjectClicker] Object world location: " + gameObject.getWorldLocation());
            context.logger.info("[ObjectClicker] Using convex hull: " + objectType.useConvexHull + ", Y offset: " + objectType.clickOffsetY);
            
            // SANITY CHECK: Validate click coordinates are within the current viewport
            int vx = context.client.getViewportXOffset();
            int vy = context.client.getViewportYOffset();
            int vw = context.client.getViewportWidth();
            int vh = context.client.getViewportHeight();
            boolean inViewport = clickPoint.x >= vx && clickPoint.y >= vy && clickPoint.x < (vx + vw) && clickPoint.y < (vy + vh);
            if (!inViewport) {
                context.logger.warn("[ObjectClicker] INVALID COORDINATES: Click point (" + clickPoint.x + "," + clickPoint.y + ") is outside viewport bounds x=[" + vx + "-" + (vx+vw) + "] y=[" + vy + "-" + (vy+vh) + "]");
                context.logger.warn("[ObjectClicker] This suggests a projection/viewport mismatch - skipping click to avoid errors");
                return false;
            }
            
            // Check for occlusion and handle it
            if (context.input.isOccludedByGeometryWithDiagnostics(clickPoint, gameObject, true)) {
                context.logger.warn("[ObjectClicker] Object is occluded by geometry - applying camera adjustments");
                
                boolean revealed = false;
                for (int attempt = 0; attempt < 6; attempt++) {
                    if (context.input.revealPointByCameraGeometry(clickPoint, gameObject, attempt)) {
                        revealed = true;
                        break;
                    }
                    context.setBusyForMs(160);
                }
                
                if (!revealed) {
                    context.logger.warn("[ObjectClicker] Could not reveal occluded object after multiple attempts");
                    return false;
                }
                
                // Re-project after camera adjustment
                clickPoint = projectObjectToClickPoint(context, gameObject, objectType);
                if (clickPoint == null) {
                    context.logger.warn("[ObjectClicker] Could not re-project object after camera adjustment");
                    return false;
                }
            }
            
            // Perform the click with validation on the client thread (same behavior as manual action)
            final String validateLabel = (actions != null) ? longestAction(actions, objectType.preferredActions) : bestAction;
            final java.awt.Point finalClickPoint = clickPoint;
            final java.util.concurrent.atomic.AtomicBoolean clickResult = new java.util.concurrent.atomic.AtomicBoolean(false);
            final java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
            try {
                context.clientThread.invoke(() -> {
                    try {
                        boolean ok = context.input.moveAndClickWithValidation(finalClickPoint, validateLabel);
                        clickResult.set(ok);
                    } catch (Exception e) {
                        context.logger.warn("[ObjectClicker] moveAndClickWithValidation failed on client thread: " + e.getMessage());
                        clickResult.set(false);
                    } finally {
                        latch.countDown();
                    }
                });
                // Wait briefly for the click to execute
                try { latch.await(600, java.util.concurrent.TimeUnit.MILLISECONDS); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            } catch (Exception e) {
                context.logger.warn("[ObjectClicker] Scheduling validated click failed: " + e.getMessage());
            }
            if (!clickResult.get()) {
                context.logger.warn("[ObjectClicker] Click validation failed for action: " + bestAction);
                return false;
            }
            
            context.logger.info("[ObjectClicker] Successfully clicked on " + comp.getName() + 
                              " at canvas point: (" + clickPoint.x + "," + clickPoint.y + ") with action: " + bestAction);
            
            // Set appropriate busy time based on object type
            int busyTime = getBusyTimeForObjectType(objectType);
            context.setBusyForMs(busyTime);
            
            return true;
            
        } catch (Exception e) {
            context.logger.warn("[ObjectClicker] Error clicking object: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Find the best action for an object based on its composition and object type.
     */
    private static String findBestAction(ObjectComposition comp, ObjectType objectType) {
        String[] actions = comp.getActions();
        if (actions == null) {
            return null;
        }
        
        // Check if all actions are null (empty action array)
        boolean hasAnyAction = false;
        for (String action : actions) {
            if (action != null) {
                hasAnyAction = true;
                break;
            }
        }
        if (!hasAnyAction) {
            // Skip objects with no actions (like closed bank booths)
            return null;
        }
        
        // Log available actions for debugging
        StringBuilder actionLog = new StringBuilder();
        for (int i = 0; i < actions.length; i++) {
            if (actions[i] != null) {
                if (i > 0) actionLog.append(", ");
                actionLog.append(actions[i]);
            }
        }
        
        // Try preferred actions (exact match)
        for (String preferredAction : objectType.preferredActions) {
            for (String action : actions) {
                if (action != null && action.equals(preferredAction)) {
                    return action;
                }
            }
        }
        
        // Try fallback actions (exact match) - only if different from preferred
        for (String fallbackAction : objectType.fallbackActions) {
            // Skip if already tried in preferred actions
            boolean alreadyTried = false;
            for (String preferredAction : objectType.preferredActions) {
                if (preferredAction.equals(fallbackAction)) {
                    alreadyTried = true;
                    break;
                }
            }
            if (!alreadyTried) {
                for (String action : actions) {
                    if (action != null && action.equals(fallbackAction)) {
                        return action;
                    }
                }
            }
        }
        
        // Try case-insensitive exact matches
        for (String preferredAction : objectType.preferredActions) {
            for (String action : actions) {
                if (action != null && action.equalsIgnoreCase(preferredAction)) {
                    return action;
                }
            }
        }
        
        // Finally try partial matches for preferred actions
        for (String preferredAction : objectType.preferredActions) {
            for (String action : actions) {
                if (action != null && action.toLowerCase().contains(preferredAction.toLowerCase())) {
                    return action;
                }
            }
        }
        
        // For banks, DO NOT click arbitrary actions. Require explicit preferred/fallbacks only.
        
        return null;
    }
    
    /**
     * Project a GameObject to a clickable point on the canvas.
     */
    private static Point projectObjectToClickPoint(TaskContext context, GameObject gameObject, ObjectType objectType) {
        if (objectType.useConvexHull) {
            // Use convex hull for more precise clicking
            Point clickPoint = ObjectFinder.projectToClickablePoint(context, gameObject);
            if (clickPoint != null && objectType.clickOffsetY != 0) {
                // Apply offset to avoid NPCs or ground
                clickPoint.y = Math.max(0, clickPoint.y + objectType.clickOffsetY);
            }
            return clickPoint;
        } else {
            // Use simple projection
            return ObjectFinder.projectToCanvas(context, gameObject);
        }
    }
    
    /**
     * Get the appropriate busy time for different object types.
     */
    private static int getBusyTimeForObjectType(ObjectType objectType) {
        if (objectType == TREE) {
            return 250; // Slightly longer to allow action to register and animation to start
        } else if (objectType == BANK) {
            return 150; // Reduced: Banks need time to open interface
        } else {
            return 100; // Reduced: Default busy time
        }
    }

    private static String longestAction(String[] actions, String[] preferredOrder) {
        String best = null;
        // Prefer preferred actions if present; pick the longest text for validation consistency
        for (String p : preferredOrder) {
            if (p == null) continue;
            for (String a : actions) {
                if (a == null) continue;
                if (a.equals(p) || a.equalsIgnoreCase(p) || a.toLowerCase().contains(p.toLowerCase()) || p.toLowerCase().contains(a.toLowerCase())) {
                    if (best == null || a.length() > best.length()) best = a;
                }
            }
        }
        if (best != null) return best;
        // Fallback: any longest non-null action
        for (String a : actions) {
            if (a == null) continue;
            if (best == null || a.length() > best.length()) best = a;
        }
        return best != null ? best : "Chop down";
    }
    
    /**
     * Click on the nearest object of a specific type.
     * 
     * @param context Task context
     * @param objectType The object type to find and click
     * @return true if an object was found and clicked successfully, false otherwise
     */
    public static boolean clickNearestObject(TaskContext context, ObjectType objectType) {
        // Find the nearest object of this type
        GameObject nearest = findNearestObjectOfType(context, objectType);
        if (nearest == null) {
            context.logger.info("[ObjectClicker] No " + objectType.namePatterns[0] + " objects found");
            return false;
        }
        
        return clickObject(context, nearest, objectType);
    }
    
    /**
     * Find the nearest object of a specific type.
     */
    private static GameObject findNearestObjectOfType(TaskContext context, ObjectType objectType) {
        // Try preferred actions first
        for (String action : objectType.preferredActions) {
            GameObject found = ObjectFinder.findNearestByAction(context, action);
            if (found != null) {
                return found;
            }
        }
        
        // Then try by name patterns
        return ObjectFinder.findNearestByNames(context, objectType.namePatterns, null);
    }
}
