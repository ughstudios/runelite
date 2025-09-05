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
        new String[]{"Bank", "Deposit"}, // Only actual RuneScape bank actions
        new String[]{"Bank", "Deposit"}, // Same as preferred - no fake fallbacks
        true, // Use convex hull
        -8    // Click slightly above center to avoid NPCs
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
            if (bestAction == null) {
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
                context.logger.warn("[ObjectClicker] No suitable action found for object: " + comp.getName() + 
                                  " (ID: " + gameObject.getId() + "). Available actions: [" + actionLog.toString() + "]");
                return false;
            }
            
            context.logger.info("[ObjectClicker] Using action '" + bestAction + "' for object: " + comp.getName() + " (ID: " + gameObject.getId() + ")");
            
            // Project object to canvas
            Point clickPoint = projectObjectToClickPoint(context, gameObject, objectType);
            if (clickPoint == null) {
                context.logger.warn("[ObjectClicker] Could not project object to canvas - running diagnostics");
                ObjectFinder.projectToCanvasWithDiagnostics(context, gameObject, true);
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
            
            // Perform the click with validation
            boolean clickSuccess = context.input.moveAndClickWithValidation(clickPoint, bestAction);
            if (!clickSuccess) {
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
        
        // For banks, try any non-null action as last resort
        if (objectType == BANK) {
            for (String action : actions) {
                if (action != null && !action.trim().isEmpty()) {
                    return action;
                }
            }
        }
        
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
            return 200; // Trees need time to start woodcutting animation
        } else if (objectType == BANK) {
            return 250; // Banks need time to open interface
        } else {
            return 200; // Default
        }
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
