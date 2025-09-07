package net.runelite.client.plugins.rlbot.input;

import java.awt.Point;
import net.runelite.api.Client;
import net.runelite.api.GameObject;
import net.runelite.api.Perspective;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.coords.WorldPoint;
import net.runelite.client.plugins.rlbot.RLBotAgent;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * Handles detection and resolution of chatbox collisions when clicking on game objects.
 * This class provides intelligent camera adjustment strategies to reveal objects
 * that are occluded by the chatbox interface.
 */
public class ChatboxCollisionHandler {
    private final Client client;
    private final RLBotInputHandler inputHandler;
    private final RLBotLogger logger;
    private final RLBotAgent rlAgent;
    
    // Track chatbox collision attempts to avoid infinite loops
    private long lastCollisionTime = 0;
    private int consecutiveCollisions = 0;
    private static final int MAX_CONSECUTIVE_COLLISIONS = 3;
    private static final long COLLISION_RESET_TIME_MS = 5000; // Reset counter after 5 seconds
    
    // Different adjustment strategies to try
    private enum AdjustmentStrategy {
        ZOOM_OUT,           // Zoom out to reduce chatbox coverage
        ROTATE_LEFT,        // Rotate camera left to move object away from chatbox
        ROTATE_RIGHT,       // Rotate camera right to move object away from chatbox
        TILT_UP,           // Tilt camera up to change perspective
        COMBINED_ROTATE_ZOOM // Combine rotation and zoom for stubborn cases
    }
    
    public ChatboxCollisionHandler(Client client, RLBotInputHandler inputHandler, RLBotLogger logger, RLBotAgent rlAgent) {
        this.client = client;
        this.inputHandler = inputHandler;
        this.logger = logger;
        this.rlAgent = rlAgent;
    }
    
    /**
     * Check if a canvas point is within the chatbox area and handle the collision
     * if necessary by adjusting the camera.
     * 
     * @param canvasPoint The point being clicked
     * @param targetObject The game object being targeted (if known)
     * @param expectedAction The expected action (for logging)
     * @return true if collision was detected and handled, false if no collision
     */
    public boolean handleChatboxCollision(Point canvasPoint, GameObject targetObject, String expectedAction) {
        if (!isInChatArea(canvasPoint)) {
            // Reset collision counter on successful non-chatbox clicks
            resetCollisionCounter();
            return false;
        }
        
        logger.warn("[CHATBOX_COLLISION] Detected click attempt in chatbox area at " + canvasPoint + 
                   " for action: " + expectedAction);
        
        // Track collision for learning and loop prevention
        updateCollisionCounter();
        
        // Apply penalty to the RL agent
        if (rlAgent != null) {
            rlAgent.addExternalPenalty(0.3f); // Moderate penalty for chatbox collision
        }
        
        // If we've had too many consecutive collisions, give up to avoid infinite loops
        if (consecutiveCollisions >= MAX_CONSECUTIVE_COLLISIONS) {
            logger.error("[CHATBOX_COLLISION] Too many consecutive collisions (" + consecutiveCollisions + 
                        "), giving up on camera adjustments");
            return true; // Return true to indicate we "handled" it by giving up
        }
        
        // Try to resolve the collision by adjusting the camera
        boolean resolved = resolveCollisionWithCameraAdjustment(canvasPoint, targetObject, expectedAction);
        
        if (resolved) {
            logger.info("[CHATBOX_COLLISION] Successfully resolved collision with camera adjustment");
        } else {
            logger.warn("[CHATBOX_COLLISION] Failed to resolve collision after camera adjustment");
        }
        
        return true; // We handled the collision (whether successfully or not)
    }
    
    /**
     * Check if a canvas point is within the chat area.
     * Uses live bounds from CHAT_CONTAINER (group 164, child 93).
     * 
     * @param canvasPoint The point to check
     * @return true if the point is in the chat area, false otherwise
     * @throws IllegalStateException if the chat widget cannot be found (should never happen)
     */
    public boolean isInChatArea(Point canvasPoint) {
        net.runelite.api.widgets.Widget chat = client.getWidget(164, 93); // ToplevelPreEoc.CHAT_CONTAINER
        if (chat == null) {
            throw new IllegalStateException("Chat widget (164, 93) not found - this should never happen!");
        }
        
        if (chat.isHidden()) {
            return false; // Chat is hidden, so no collision possible
        }
        
        java.awt.Rectangle bounds = chat.getBounds();
        if (bounds == null || bounds.width <= 0 || bounds.height <= 0) {
            throw new IllegalStateException("Chat widget bounds are invalid: " + bounds);
        }
        
        return bounds.contains(canvasPoint);
    }
    
    /**
     * Attempt to resolve the chatbox collision by adjusting the camera.
     * Uses different strategies based on the collision count and object position.
     */
    private boolean resolveCollisionWithCameraAdjustment(Point canvasPoint, GameObject targetObject, String expectedAction) {
        AdjustmentStrategy strategy = selectAdjustmentStrategy(canvasPoint, targetObject);
        
        logger.info("[CHATBOX_COLLISION] Attempting resolution with strategy: " + strategy);
        
        try {
            switch (strategy) {
                case ZOOM_OUT:
                    // Zoom out to make the chatbox cover less of the screen
                    for (int i = 0; i < 3; i++) {
                        inputHandler.zoomOutSmall();
                    }
                    break;
                    
                case ROTATE_LEFT:
                    // Rotate camera left to move the object away from chatbox area
                    inputHandler.rotateCameraSafe(-120, 0);
                    break;
                    
                case ROTATE_RIGHT:
                    // Rotate camera right to move the object away from chatbox area
                    inputHandler.rotateCameraSafe(120, 0);
                    break;
                    
                case TILT_UP:
                    // Tilt camera up to change the perspective
                    inputHandler.rotateCameraSafe(0, -60);
                    break;
                    
                case COMBINED_ROTATE_ZOOM:
                    // Combine rotation and zoom for stubborn cases
                    inputHandler.rotateCameraSafe(90, -30);
                    inputHandler.zoomOutSmall();
                    inputHandler.zoomOutSmall();
                    break;
            }
            
            // Give the camera adjustment time to take effect
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            // Check if the adjustment resolved the collision
            Point newProjection = null;
            if (targetObject != null) {
                LocalPoint localPoint = LocalPoint.fromWorld(client.getTopLevelWorldView(), targetObject.getWorldLocation());
                if (localPoint != null) {
                    net.runelite.api.Point apiPoint = Perspective.localToCanvas(client, localPoint, targetObject.getPlane());
                    if (apiPoint != null) {
                        newProjection = new Point(apiPoint.getX(), apiPoint.getY());
                    }
                }
            }
            
            if (newProjection != null && !isInChatArea(newProjection)) {
                logger.info("[CHATBOX_COLLISION] Camera adjustment successful - object now at " + newProjection);
                return true;
            } else {
                logger.warn("[CHATBOX_COLLISION] Camera adjustment did not resolve collision");
                return false;
            }
            
        } catch (Exception e) {
            logger.error("[CHATBOX_COLLISION] Error during camera adjustment: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Select the best adjustment strategy based on the collision context.
     */
    private AdjustmentStrategy selectAdjustmentStrategy(Point canvasPoint, GameObject targetObject) {
        // Strategy selection based on collision count and object position
        switch (consecutiveCollisions) {
            case 1:
                // First collision: try simple zoom out
                return AdjustmentStrategy.ZOOM_OUT;
                
            case 2:
                // Second collision: try rotation based on object position
                if (targetObject != null) {
                    return selectRotationStrategy(targetObject);
                } else {
                    return AdjustmentStrategy.ROTATE_LEFT;
                }
                
            case 3:
                // Third collision: try tilt adjustment
                return AdjustmentStrategy.TILT_UP;
                
            default:
                // Last resort: combined approach
                return AdjustmentStrategy.COMBINED_ROTATE_ZOOM;
        }
    }
    
    /**
     * Select rotation strategy based on object's world position relative to player.
     */
    private AdjustmentStrategy selectRotationStrategy(GameObject targetObject) {
        try {
            WorldPoint playerPos = client.getLocalPlayer().getWorldLocation();
            WorldPoint objectPos = targetObject.getWorldLocation();
            
            if (playerPos != null && objectPos != null) {
                int deltaX = objectPos.getX() - playerPos.getX();
                // int deltaY = objectPos.getY() - playerPos.getY(); // Not used currently
                
                // If object is more to the east, rotate right to move it left (away from chatbox)
                // If object is more to the west, rotate left to move it right (away from chatbox)
                if (deltaX > 0) {
                    return AdjustmentStrategy.ROTATE_RIGHT;
                } else {
                    return AdjustmentStrategy.ROTATE_LEFT;
                }
            }
        } catch (Exception e) {
            logger.warn("[CHATBOX_COLLISION] Error determining rotation strategy: " + e.getMessage());
        }
        
        // Default to left rotation
        return AdjustmentStrategy.ROTATE_LEFT;
    }
    
    /**
     * Update the collision counter and timestamp.
     */
    private void updateCollisionCounter() {
        long currentTime = System.currentTimeMillis();
        
        // Reset counter if enough time has passed
        if (currentTime - lastCollisionTime > COLLISION_RESET_TIME_MS) {
            consecutiveCollisions = 0;
        }
        
        consecutiveCollisions++;
        lastCollisionTime = currentTime;
        
        logger.debug("[CHATBOX_COLLISION] Collision count: " + consecutiveCollisions);
    }
    
    /**
     * Reset the collision counter (called on successful non-chatbox clicks).
     */
    private void resetCollisionCounter() {
        if (consecutiveCollisions > 0) {
            logger.debug("[CHATBOX_COLLISION] Resetting collision counter");
            consecutiveCollisions = 0;
            lastCollisionTime = 0;
        }
    }
    
    /**
     * Get the current collision count (for debugging/monitoring).
     */
    public int getConsecutiveCollisions() {
        return consecutiveCollisions;
    }
}
