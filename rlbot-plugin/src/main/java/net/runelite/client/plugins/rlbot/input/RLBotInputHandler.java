package net.runelite.client.plugins.rlbot.input;

import java.awt.Canvas;
import java.awt.Component;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.security.SecureRandom;
import java.util.Random;
import javax.inject.Inject;
import javax.swing.SwingUtilities;
import net.runelite.api.Client;
import net.runelite.api.MenuEntry;
import net.runelite.api.ObjectComposition;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.input.KeyManager;
import net.runelite.client.input.MouseManager;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * Handles mouse and keyboard input for the RLBot plugin.
 * Uses direct event dispatching to the game canvas for reliable input handling.
 */
public class RLBotInputHandler {
    
    /**
     * The logger instance.
     */
    private final RLBotLogger logger;
    
    /**
     * Random number generator for humanization.
     */
    private final Random random = new SecureRandom();
    
    /**
     * RuneLite client instance.
     */
    private final Client client;
    
    /**
     * Client thread for RuneLite operations.
     */
    private final ClientThread clientThread;
    
    /**
     * Key manager for RuneLite key events.
     */
    private final KeyManager keyManager;
    
    /**
     * Mouse manager for RuneLite mouse events.
     */
    private final MouseManager mouseManager;
    
    /**
     * Reference to the RL agent for applying penalties.
     */
    private net.runelite.client.plugins.rlbot.RLBotAgent rlAgent;

    // Track the last canvas point we moved to, to allow precise clicking
    private volatile Point lastCanvasMovePoint = null;
    
    /**
     * Creates a new RLBotInputHandler with dependency injection.
     * 
     * @param logger The logger
     * @param client The RuneLite client
     * @param clientThread The client thread
     * @param keyManager The key manager
     * @param mouseManager The mouse manager
     */
    @Inject
    public RLBotInputHandler(
        RLBotLogger logger,
        Client client,
        ClientThread clientThread,
        KeyManager keyManager,
        MouseManager mouseManager
    ) {
        this.logger = logger;
        this.client = client;
        this.clientThread = clientThread;
        this.keyManager = keyManager;
        this.mouseManager = mouseManager;
    }
    
    /**
     * Initialize the input handler.
     */
    public void initialize() {
        logger.info("Initializing RLBot input handler");
        
        // Test if we can access the canvas (non-blocking)
        Canvas canvas = getCanvas();
        if (canvas != null) {
            try {
                // Only check location if canvas is showing to avoid IllegalComponentStateException
                if (canvas.isShowing()) {
                    Point location = canvas.getLocationOnScreen();
                    logger.info("Canvas found at location: " + location.x + "," + location.y);
                } else {
                    logger.info("Canvas exists but not yet showing - will be available later");
                }
            } catch (Exception e) {
                logger.error("Error getting canvas location: " + e.getMessage() + ": " + e.toString());
            }
        } else {
            logger.warn("Canvas not yet available - will be initialized when needed");
        }
        
        logger.info("RLBot input handler initialization complete");
    }
    
    /**
     * Get the game canvas
     * 
     * @return The Canvas object or null if not available
     */
    private Canvas getCanvas() {
        if (client == null) {
            logger.error("Client is null, cannot get canvas");
            return null;
        }
        
        return client.getCanvas();
    }
    
    /**
     * Set the RL agent reference for applying penalties.
     */
    public void setRLAgent(net.runelite.client.plugins.rlbot.RLBotAgent agent) {
        this.rlAgent = agent;
    }
    
    /**
     * Check if a canvas point is within the chat area that should be avoided.
     * Based on widget inspector: chat container bounds are x=0, y=338, width=519, height=165
     *
     * @param canvasPoint The point to check
     * @return true if the point is in the chat area
     */
    private boolean isInChatArea(Point canvasPoint) {
        // Chat area bounds: x=0, y=338, width=519, height=165
        return canvasPoint.x >= 0 && canvasPoint.x <= 519 && 
               canvasPoint.y >= 338 && canvasPoint.y <= 503; // 338 + 165 = 503
    }

    /**
     * True if the point lies within the current 3D viewport area (excludes UI panels).
     */
    private boolean isPointInViewport(Point canvasPoint) {
        try {
            int vx = client.getViewportXOffset();
            int vy = client.getViewportYOffset();
            int vw = client.getViewportWidth();
            int vh = client.getViewportHeight();
            if (vw <= 0 || vh <= 0) {
                // If viewport not initialized yet, don't block interaction solely on this
                return true;
            }
            return canvasPoint.x >= vx && canvasPoint.x < vx + vw &&
                   canvasPoint.y >= vy && canvasPoint.y < vy + vh;
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * Returns true if the point is likely occluded by UI (chatbox, side panels) and not clickable in 3D.
     */
    private boolean isOccludedByUI(Point canvasPoint) {
        // Outside viewport is UI (minimap/side panels/chat frame areas)
        if (!isPointInViewport(canvasPoint)) {
            return true;
        }
        // Also treat known chat area overlay as occluding
        if (isInChatArea(canvasPoint)) {
            return true;
        }
        return false;
    }

    /**
     * Check what's under the mouse cursor at the specified canvas point.
     * This validates that the target has the expected action before clicking.
     *
     * @param canvasPoint The canvas point to check
     * @param expectedAction The expected action (e.g., "Chop", "Bank", "Use")
     * @return true if the target has the expected action, false otherwise
     */
    public boolean validateTargetAtPoint(Point canvasPoint, String expectedAction) {
        if (client == null) {
            logger.warn("[RLBOT_INPUT] Client is null, cannot validate target");
            return false;
        }

        // Early block if occluded by UI
        if (isOccludedByUI(canvasPoint)) {
            logger.debug("[RLBOT_INPUT] Point " + canvasPoint + " is occluded by UI; skipping target validation");
            return false;
        }

        try {
            // Search scene for a projected point near the requested canvasPoint
            final int plane = client.getPlane();
            net.runelite.api.Scene scene = client.getScene();
            if (scene == null) return false;
            net.runelite.api.Tile[][] tiles = scene.getTiles()[plane];
            if (tiles == null) return false;

            // Check game objects by projecting to canvas and comparing proximity
            for (int x = 0; x < tiles.length; x++) {
                net.runelite.api.Tile[] col = tiles[x];
                if (col == null) continue;
                for (int y = 0; y < col.length; y++) {
                    net.runelite.api.Tile tile = col[y];
                    if (tile == null) continue;
                    for (net.runelite.api.GameObject gameObject : tile.getGameObjects()) {
                        if (gameObject == null) continue;
                        net.runelite.api.coords.LocalPoint lp = gameObject.getLocalLocation();
                        if (lp == null) continue;
                        net.runelite.api.Point proj = net.runelite.api.Perspective.localToCanvas(client, lp, plane);
                        if (proj == null) continue;
                        if (Math.abs(proj.getX() - canvasPoint.x) <= 10 && Math.abs(proj.getY() - canvasPoint.y) <= 10) {
                            net.runelite.api.ObjectComposition composition = client.getObjectDefinition(gameObject.getId());
                            if (composition == null) continue;
                            String[] actions = composition.getActions();
                            if (actions == null) continue;
                            for (String action : actions) {
                                if (action != null && action.toLowerCase().contains(expectedAction.toLowerCase())) {
                                    logger.debug("[RLBOT_INPUT] Found valid target at " + canvasPoint + ": " + composition.getName() + " with action '" + action + "'");
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            // Check NPCs on the tile
            for (net.runelite.api.NPC npc : client.getNpcs()) {
                if (npc == null) continue;
                
                net.runelite.api.Point npcCanvasPoint = net.runelite.api.Perspective.localToCanvas(client, npc.getLocalLocation(), client.getPlane(), npc.getLogicalHeight());
                if (npcCanvasPoint == null) continue;
                
                // Check if NPC is close to our target point (within 10 pixels)
                if (Math.abs(npcCanvasPoint.getX() - canvasPoint.x) <= 10 && 
                    Math.abs(npcCanvasPoint.getY() - canvasPoint.y) <= 10) {
                    
                    net.runelite.api.NPCComposition npcComposition = npc.getTransformedComposition();
                    if (npcComposition == null) continue;
                    
                    String[] actions = npcComposition.getActions();
                    if (actions == null) continue;
                    
                    for (String action : actions) {
                        if (action != null && action.toLowerCase().contains(expectedAction.toLowerCase())) {
                            logger.debug("[RLBOT_INPUT] Found valid NPC at " + canvasPoint + ": " + npcComposition.getName() + " with action '" + action + "'");
                            return true;
                        }
                    }
                }
            }

            // Ground item validation omitted for compatibility

            logger.debug("[RLBOT_INPUT] No valid target with action '" + expectedAction + "' found at " + canvasPoint);
            return false;
            
        } catch (Exception e) {
            logger.warn("[RLBOT_INPUT] Error validating target at " + canvasPoint + ": " + e.getMessage());
            return false;
        }
    }

    /**
     * Get detailed information about what's under the mouse cursor.
     * This is useful for debugging and understanding what the bot is targeting.
     *
     * @param canvasPoint The canvas point to check
     * @return A string describing what's under the cursor, or null if nothing found
     */
    public String getTargetInfoAtPoint(Point canvasPoint) {
        if (client == null) {
            return "Client is null";
        }

        try {
            final int plane = client.getPlane();
            net.runelite.api.Scene scene = client.getScene();
            if (scene == null) return "No scene";
            net.runelite.api.Tile[][] tiles = scene.getTiles()[plane];
            if (tiles == null) return "No tiles for plane";

            StringBuilder info = new StringBuilder();
            
            // Check game objects near the canvas point
            for (int x = 0; x < tiles.length; x++) {
                net.runelite.api.Tile[] col = tiles[x];
                if (col == null) continue;
                for (int y = 0; y < col.length; y++) {
                    net.runelite.api.Tile tile = col[y];
                    if (tile == null) continue;
                    for (net.runelite.api.GameObject gameObject : tile.getGameObjects()) {
                        if (gameObject == null) continue;
                        net.runelite.api.coords.LocalPoint lp = gameObject.getLocalLocation();
                        if (lp == null) continue;
                        net.runelite.api.Point proj = net.runelite.api.Perspective.localToCanvas(client, lp, plane);
                        if (proj == null) continue;
                        if (Math.abs(proj.getX() - canvasPoint.x) <= 10 && Math.abs(proj.getY() - canvasPoint.y) <= 10) {
                            net.runelite.api.ObjectComposition composition = client.getObjectDefinition(gameObject.getId());
                            if (composition == null) continue;
                            info.append("GameObject: ").append(composition.getName()).append(" (ID: ").append(gameObject.getId()).append(")");
                            String[] actions = composition.getActions();
                            if (actions != null) {
                                info.append(" Actions: [");
                                for (int i = 0; i < actions.length; i++) {
                                    if (actions[i] != null) {
                                        if (i > 0) info.append(", ");
                                        info.append(actions[i]);
                                    }
                                }
                                info.append("]");
                            }
                            info.append("; ");
                        }
                    }
                }
            }

            // Check NPCs
            for (net.runelite.api.NPC npc : client.getNpcs()) {
                if (npc == null) continue;
                
                net.runelite.api.Point npcCanvasPoint = net.runelite.api.Perspective.localToCanvas(client, npc.getLocalLocation(), client.getPlane(), npc.getLogicalHeight());
                if (npcCanvasPoint == null) continue;
                
                if (Math.abs(npcCanvasPoint.getX() - canvasPoint.x) <= 10 && 
                    Math.abs(npcCanvasPoint.getY() - canvasPoint.y) <= 10) {
                    
                    net.runelite.api.NPCComposition npcComposition = npc.getTransformedComposition();
                    if (npcComposition == null) continue;
                    
                    info.append("NPC: ").append(npcComposition.getName()).append(" (ID: ").append(npc.getId()).append(")");
                    String[] actions = npcComposition.getActions();
                    if (actions != null) {
                        info.append(" Actions: [");
                        for (int i = 0; i < actions.length; i++) {
                            if (actions[i] != null) {
                                if (i > 0) info.append(", ");
                                info.append(actions[i]);
                            }
                        }
                        info.append("]");
                    }
                    info.append("; ");
                }
            }

            // Ground item info omitted for compatibility

            return info.length() > 0 ? info.toString() : "Nothing found at this location";
            
        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }

    /**
     * Move the mouse to the specified point on the game canvas and dispatch appropriate events.
     * This version waits for the movement to complete before returning.
     *
     * @param canvasPoint The point on the game canvas to move to
     */
    public void smoothMouseMove(Point canvasPoint) {
        // Check if the target point is in the chat area and apply penalty for RL learning
        if (isInChatArea(canvasPoint)) {
            logger.warn("[RLBOT_INPUT] Attempting to move mouse to chat area (" + canvasPoint.x + "," + canvasPoint.y + ") - applying penalty");
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.3f); // Penalty for trying to move mouse to chat area
            }
        }
        
        logger.debug("[RLBOT_INPUT] BEGIN smoothMouseMove to canvas point: " + canvasPoint.x + "," + canvasPoint.y);
        
        // Run interpolation on a background daemon thread and post events to EDT
        Thread mover = new Thread(() -> {
            Canvas canvas = getCanvas();
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot move mouse");
                return;
            }
            Point start;
            try {
                Point mp = canvas.getMousePosition();
                start = mp != null ? mp : new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            } catch (Exception ex) {
                start = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }
            int px = 20;
            try {
                px = ((net.runelite.client.plugins.rlbot.RLBotConfig)net.runelite.client.RuneLite.getInjector().getInstance(net.runelite.client.plugins.rlbot.RLBotConfig.class)).mouseMoveInterpolationPx();
            } catch (Throwable ignore) {}
            double dx = canvasPoint.x - start.x;
            double dy = canvasPoint.y - start.y;
            double dist = Math.hypot(dx, dy);
            int steps = Math.max(1, (int)Math.ceil(dist / Math.max(5, px)));
            for (int i = 1; i <= steps; i++) {
                int x = start.x + (int)Math.round(dx * i / steps);
                int y = start.y + (int)Math.round(dy * i / steps);
                Point stepPoint = new Point(x, y);
                SwingUtilities.invokeLater(() -> dispatchMouseMoveEvent(canvas, stepPoint));
                try { Thread.sleep(8); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            }
            lastCanvasMovePoint = new Point(canvasPoint);
        }, "rlbot-mouse-move");
        mover.setDaemon(true);
        mover.start();
        
        logger.debug("[RLBOT_INPUT] END smoothMouseMove (completed)");
    }
    
    /**
     * Dispatches a mouse move event to the component.
     *
     * @param component The component to dispatch the event to
     * @param point The point on the component
     */
    private void dispatchMouseMoveEvent(Component component, Point point) {
        logger.info("[RLBOT_INPUT] BEGIN dispatchMouseMoveEvent to: " + point.x + "," + point.y);
        
        long when = System.currentTimeMillis();
        int modifiers = 0;
        int clickCount = 0;
        boolean popupTrigger = false;
        
        // Check if point is within canvas bounds
        boolean isInBounds = point.x >= 0 && point.y >= 0 && 
                           point.x < component.getWidth() && 
                           point.y < component.getHeight();
        
        // If moving out of bounds, dispatch a mouse exit event
        if (!isInBounds) {
            MouseEvent exitEvent = new MouseEvent(
                component,
                MouseEvent.MOUSE_EXITED,
                when,
                modifiers,
                point.x,
                point.y,
                clickCount,
                popupTrigger
            );
            component.dispatchEvent(exitEvent);
            return;
        }
        
        // If we were previously out of bounds, dispatch a mouse enter event
        MouseEvent enterEvent = new MouseEvent(
            component,
            MouseEvent.MOUSE_ENTERED,
            when,
            modifiers,
            point.x,
            point.y,
            clickCount,
            popupTrigger
        );
        component.dispatchEvent(enterEvent);
        
        logger.info("[RLBOT_INPUT] Creating MouseEvent with params: id=MOUSE_MOVED, when=" + when + 
                    ", modifiers=" + modifiers + ", point=(" + point.x + "," + point.y + 
                    "), clickCount=" + clickCount + ", popupTrigger=" + popupTrigger);
        
        MouseEvent event = new MouseEvent(
            component,
            MouseEvent.MOUSE_MOVED,
            when,
            modifiers,
            point.x,
            point.y,
            clickCount,
            popupTrigger
        );
        
        // First make sure component has focus
        logger.info("[RLBOT_INPUT] Requesting focus on component: " + component.getClass().getName());
        component.requestFocus();
        
        // Post to EDT; no sleeps here
        SwingUtilities.invokeLater(() -> component.dispatchEvent(event));
        
        logger.info("[RLBOT_INPUT] END dispatchMouseMoveEvent");
    }
    
        /**
     * Move the mouse to the specified point and validate the target before clicking.
     * This is the complete flow: move -> validate -> click.
     *
     * @param canvasPoint The canvas point to move to and click
     * @param expectedAction The expected action (e.g., "Chop", "Bank", "Use")
     * @return true if the target was valid and click was successful, false otherwise
     */
    public boolean moveAndClickWithValidation(Point canvasPoint, String expectedAction) {
        logger.debug("[RLBOT_INPUT] BEGIN moveAndClickWithValidation at " + canvasPoint + " for action: " + expectedAction);
        
        // First move the mouse to the target (now synchronous)
        smoothMouseMove(canvasPoint);
        
        // Now validate what's under the mouse cursor
        if (isOccludedByUI(canvasPoint)) {
            logger.warn("[RLBOT_INPUT] Target point is occluded by UI at " + canvasPoint + "; aborting click");
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.2f);
            }
            return false;
        }
        if (!validateTargetAtPoint(canvasPoint, expectedAction)) {
            logger.warn("[RLBOT_INPUT] Target validation failed for action '" + expectedAction + "' at " + canvasPoint);
            String targetInfo = getTargetInfoAtPoint(canvasPoint);
            logger.info("[RLBOT_INPUT] Target info: " + targetInfo);
            
            // Apply penalty for clicking on invalid target
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.5f);
            }
            return false;
        }
        
        // Target is valid, proceed with click (now synchronous)
        long t0 = System.nanoTime();
        logger.debug("[RLBOT_INPUT] Target validation passed, proceeding with click");
        click();
        long t1 = System.nanoTime();
        logger.perf("Input.moveAndClick validated click took " + ((t1 - t0) / 1_000_000) + " ms");
        return true;
    }

    /**
     * Click at the current mouse position with validation.
     * This method validates that the target has the expected action before clicking.
     *
     * @param expectedAction The expected action (e.g., "Chop", "Bank", "Use")
     * @return true if the click was successful and target was valid, false otherwise
     */
    public boolean clickWithValidation(String expectedAction) {
        logger.debug("[RLBOT_INPUT] BEGIN clickWithValidation for action: " + expectedAction);
        
        // Get the current mouse position
        Point clickPoint = lastCanvasMovePoint;
        if (clickPoint == null) {
            logger.warn("[RLBOT_INPUT] No known mouse position for validation");
            return false;
        }
        
        // Validate the target before clicking
        if (!validateTargetAtPoint(clickPoint, expectedAction)) {
            logger.warn("[RLBOT_INPUT] Target validation failed for action '" + expectedAction + "' at " + clickPoint);
            String targetInfo = getTargetInfoAtPoint(clickPoint);
            logger.info("[RLBOT_INPUT] Target info: " + targetInfo);
            
            // Apply penalty for clicking on invalid target
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.5f);
            }
            return false;
        }
        
        // Target is valid, proceed with click
        long t0 = System.nanoTime();
        logger.debug("[RLBOT_INPUT] Target validation passed, proceeding with click");
        click();
        long t1 = System.nanoTime();
        logger.perf("Input.clickWithValidation took " + ((t1 - t0) / 1_000_000) + " ms");
        return true;
    }

    /**
     * Click at a specific canvas coordinate with validation.
     * This method validates that the target has the expected action before clicking.
     *
     * @param canvasPoint The canvas point to click at
     * @param expectedAction The expected action (e.g., "Chop", "Bank", "Use")
     * @return true if the click was successful and target was valid, false otherwise
     */
    public boolean clickAtWithValidation(Point canvasPoint, String expectedAction) {
        logger.debug("[RLBOT_INPUT] BEGIN clickAtWithValidation at " + canvasPoint + " for action: " + expectedAction);
        
        // Validate the target before clicking
        if (isOccludedByUI(canvasPoint)) {
            logger.warn("[RLBOT_INPUT] Click point is occluded by UI at " + canvasPoint + "; aborting click");
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.2f);
            }
            return false;
        }
        if (!validateTargetAtPoint(canvasPoint, expectedAction)) {
            logger.warn("[RLBOT_INPUT] Target validation failed for action '" + expectedAction + "' at " + canvasPoint);
            String targetInfo = getTargetInfoAtPoint(canvasPoint);
            logger.info("[RLBOT_INPUT] Target info: " + targetInfo);
            
            // Apply penalty for clicking on invalid target
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.5f);
            }
            return false;
        }
        
        // Target is valid, proceed with click
        long t0 = System.nanoTime();
        logger.debug("[RLBOT_INPUT] Target validation passed, proceeding with click");
        clickAt(canvasPoint);
        long t1 = System.nanoTime();
        logger.perf("Input.clickAtWithValidation took " + ((t1 - t0) / 1_000_000) + " ms");
        return true;
    }

        /**
     * Click at the current mouse position.
     */
    public void click() {
        logger.debug("[RLBOT_INPUT] BEGIN click at current mouse position");
        
        // Run click on background thread and post events to EDT
        Thread clicker = new Thread(() -> {
            Canvas canvas = getCanvas();
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot click");
                return;
            }
            Point clickPoint = lastCanvasMovePoint;
            if (clickPoint == null) {
                try {
                    Point mousePos = canvas.getMousePosition();
                    if (mousePos != null) clickPoint = mousePos;
                } catch (Exception ignored) {}
            }
            if (clickPoint == null) {
                clickPoint = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }
            dispatchMouseClickEvent(canvas, clickPoint);
            lastCanvasMovePoint = null;
        }, "rlbot-click");
        clicker.setDaemon(true);
        clicker.start();
        
        logger.debug("[RLBOT_INPUT] END click (completed)");
    }

    /**
     * Click at a specific canvas coordinate.
     */
    public void clickAt(Point canvasPoint) {
        // Check if the target point is in the chat area and apply penalty for RL learning
        if (isInChatArea(canvasPoint)) {
            logger.warn("[RLBOT_INPUT] Attempting to click in chat area (" + canvasPoint.x + "," + canvasPoint.y + ") - applying penalty");
            if (rlAgent != null) {
                rlAgent.addExternalPenalty(0.5f); // Higher penalty for clicking in chat area
            }
        }
        
        logger.debug("[RLBOT_INPUT] BEGIN clickAt point: " + canvasPoint.x + "," + canvasPoint.y);
        
        Thread clicker = new Thread(() -> {
            Canvas canvas = getCanvas();
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot clickAt");
                return;
            }
            dispatchMouseClickEvent(canvas, canvasPoint);
            lastCanvasMovePoint = null;
        }, "rlbot-click");
        clicker.setDaemon(true);
        clicker.start();
        
        logger.debug("[RLBOT_INPUT] END clickAt (completed)");
    }
    
    /**
     * Dispatches mouse press, release, and click events to simulate a click.
     *
     * @param component The component to dispatch the events to
     * @param point The point on the component
     */
    private void dispatchMouseClickEvent(Component component, Point point) {
        logger.info("[RLBOT_INPUT] BEGIN dispatchMouseClickEvent at: " + point.x + "," + point.y);
        
        long when = System.currentTimeMillis();
        int modifiers = InputEvent.BUTTON1_DOWN_MASK;
        int clickCount = 1;
        boolean popupTrigger = false;
        
        // First make sure component has focus
        logger.info("[RLBOT_INPUT] Requesting focus on component: " + component.getClass().getName());
        component.requestFocus();
        
        // Run press/release scheduling on a daemon thread and post events to EDT
        Thread t = new Thread(() -> {
            try {
                MouseEvent pressEvent = new MouseEvent(
                    component,
                    MouseEvent.MOUSE_PRESSED,
                    when,
                    modifiers,
                    point.x,
                    point.y,
                    clickCount,
                    popupTrigger,
                    MouseEvent.BUTTON1
                );
                MouseEvent releaseEvent = new MouseEvent(
                    component,
                    MouseEvent.MOUSE_RELEASED,
                    when + 50,
                    modifiers,
                    point.x,
                    point.y,
                    clickCount,
                    popupTrigger,
                    MouseEvent.BUTTON1
                );
                MouseEvent clickEvent = new MouseEvent(
                    component,
                    MouseEvent.MOUSE_CLICKED,
                    when + 51,
                    modifiers,
                    point.x,
                    point.y,
                    clickCount,
                    popupTrigger,
                    MouseEvent.BUTTON1
                );
                SwingUtilities.invokeLater(() -> component.dispatchEvent(pressEvent));
                try { Thread.sleep(30); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
                SwingUtilities.invokeLater(() -> {
                    component.dispatchEvent(releaseEvent);
                    component.dispatchEvent(clickEvent);
                });
                try { Thread.sleep(50); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] Error scheduling mouse click events: " + e.getMessage() + ": " + e.toString());
            }
        }, "rlbot-click-seq");
        t.setDaemon(true);
        t.start();
        
        logger.info("[RLBOT_INPUT] END dispatchMouseClickEvent");
    }
    
    /**
     * Interact with a game object using the proper menu action system.
     * This is the correct way to interact with game objects in RuneLite.
     *
     * @param gameObject The game object to interact with
     * @param action The menu action (e.g., "Use", "Bank", etc.)
     */
    public void interactWithGameObject(net.runelite.api.GameObject gameObject, String action) {
        logger.info("[RLBOT_INPUT] BEGIN interactWithGameObject: " + action + " on object " + gameObject.getId());
        
        clientThread.invoke(() -> {
            try {
                // Get object composition to find the correct action index
                net.runelite.api.ObjectComposition composition = client.getObjectDefinition(gameObject.getId());
                if (composition == null) {
                    logger.error("[RLBOT_INPUT] Could not get object composition for object " + gameObject.getId());
                    return;
                }
                
                String[] actions = composition.getActions();
                if (actions == null) {
                    logger.error("[RLBOT_INPUT] Object has no actions: " + gameObject.getId());
                    return;
                }
                
                // Find the action index (with synonyms fallback for common labels) using case-insensitive contains
                int actionIndex = -1;
                String desired = action == null ? "" : action.trim();
                String[] synonyms;
                if (desired.equalsIgnoreCase("Bank")) {
                    synonyms = new String[] {"Bank"}; // enforce exact Bank only
                } else if (desired.equalsIgnoreCase("Chop down") || desired.equalsIgnoreCase("Chop")) {
                    synonyms = new String[] {"Chop down", "Chop"};
                } else {
                    synonyms = new String[] {desired};
                }
                for (String candidate : synonyms) {
                    String candLc = candidate == null ? "" : candidate.toLowerCase();
                    for (int i = 0; i < actions.length; i++) {
                        String ai = actions[i];
                        if (ai == null) continue;
                        String aiLc = ai.toLowerCase();
                        if (aiLc.contains(candLc)) {
                            actionIndex = i;
                            desired = ai; // use the concrete label from the object for invocation/logging
                            break;
                        }
                    }
                    if (actionIndex != -1) break;
                }
                
                if (actionIndex == -1) {
                    logger.error("[RLBOT_INPUT] Action '" + action + "' not found for object " + gameObject.getId() + 
                                 ". Available actions: " + java.util.Arrays.toString(actions));
                    return;
                }
                
                // Convert action index to MenuAction
                net.runelite.api.MenuAction menuAction;
                switch (actionIndex) {
                    case 0: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FIRST_OPTION; break;
                    case 1: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_SECOND_OPTION; break;
                    case 2: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_THIRD_OPTION; break;
                    case 3: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FOURTH_OPTION; break;
                    case 4: menuAction = net.runelite.api.MenuAction.GAME_OBJECT_FIFTH_OPTION; break;
                    default:
                        logger.error("[RLBOT_INPUT] Unsupported action index: " + actionIndex);
                        return;
                }
                
                // Use scene coordinates for object interactions
                net.runelite.api.coords.LocalPoint lp = net.runelite.api.coords.LocalPoint.fromWorld(client, gameObject.getWorldLocation());
                if (lp == null) {
                    logger.error("[RLBOT_INPUT] LocalPoint null for object " + gameObject.getId());
                    return;
                }
                int sceneX = lp.getSceneX();
                int sceneY = lp.getSceneY();

                logger.info("[RLBOT_INPUT] Invoking menu action: " + menuAction + " on object " + gameObject.getId() +
                           " at scene(" + sceneX + "," + sceneY + ") with action '" + desired + "'");

                // Optionally move mouse near object for human-like behavior (no extra clicks)
                net.runelite.api.Point canvasPoint = net.runelite.api.Perspective.localToCanvas(client, lp, 0);
                if (canvasPoint != null) {
                    java.awt.Point hoverPoint = findBestClickPosition(canvasPoint, gameObject);
                    smoothMouseMove(hoverPoint);
                }

                // Invoke the menu action with scene coords (this is the primary interaction method)
                logger.info("[RLBOT_INPUT] Invoking menu action: sceneX=" + sceneX + ", sceneY=" + sceneY + ", menuAction=" + menuAction + ", objectId=" + gameObject.getId() + ", action='" + desired + "'");
                client.menuAction(
                    sceneX,
                    sceneY,
                    menuAction,
                    gameObject.getId(),
                    -1,
                    desired,
                    ""
                );
                logger.info("[RLBOT_INPUT] Menu action invoked successfully");
                
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] Error interacting with game object: " + e.getMessage() + ": " + e.toString());
            }
        });
        
        logger.info("[RLBOT_INPUT] END interactWithGameObject");
    }
    
    /**
     * Right-click at the current mouse position.
     */
    public void rightClick() {
        logger.info("Right-clicking at current mouse position");
        
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            if (canvas == null) {
                logger.error("Canvas is null, cannot right-click");
                return;
            }
            
            // Get the current mouse position relative to the canvas
            Point canvasPosition;
            try {
                Point mousePos = canvas.getMousePosition();
                if (mousePos == null) {
                    logger.warn("Mouse position is null, using canvas center");
                    canvasPosition = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
                } else {
                    canvasPosition = mousePos;
                }
            } catch (Exception e) {
                logger.error("Error getting mouse position: " + e.getMessage() + ": " + e.toString());
                canvasPosition = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }
            
            // Dispatch mouse events
            dispatchMouseRightClickEvent(canvas, canvasPosition);
            logger.info("Right-click event dispatched at: " + canvasPosition.x + "," + canvasPosition.y);
        });
    }
    
    /**
     * Dispatches mouse events to simulate a right-click.
     *
     * @param component The component to dispatch the events to
     * @param point The point on the component
     */
    private void dispatchMouseRightClickEvent(Component component, Point point) {
        long when = System.currentTimeMillis();
        int modifiers = InputEvent.BUTTON3_DOWN_MASK;
        int clickCount = 1;
        boolean popupTrigger = true;
        
        // First make sure component has focus
        component.requestFocus();
        
        try {
            // Create mouse events
            MouseEvent pressEvent = new MouseEvent(
                component,
                MouseEvent.MOUSE_PRESSED,
                when,
                modifiers,
                point.x,
                point.y,
                clickCount,
                popupTrigger,
                MouseEvent.BUTTON3
            );
            
            MouseEvent releaseEvent = new MouseEvent(
                component,
                MouseEvent.MOUSE_RELEASED,
                when + 50,
                modifiers,
                point.x,
                point.y,
                clickCount,
                popupTrigger,
                MouseEvent.BUTTON3
            );
            
            MouseEvent clickEvent = new MouseEvent(
                component,
                MouseEvent.MOUSE_CLICKED,
                when + 51,
                modifiers,
                point.x,
                point.y,
                clickCount,
                popupTrigger,
                MouseEvent.BUTTON3
            );
            
            // Dispatch events with proper timing
            SwingUtilities.invokeAndWait(() -> {
                component.dispatchEvent(pressEvent);
            });
            
            SwingUtilities.invokeAndWait(() -> {
                component.dispatchEvent(releaseEvent);
                component.dispatchEvent(clickEvent);
            });
            
        } catch (Exception e) {
            logger.error("Error dispatching mouse right-click events: " + e.getMessage() + ": " + e.toString());
        }
    }
    
    /**
     * Press and release a key.
     *
     * @param keyCode The key code to press
     */
    public void pressKey(int keyCode) {
        logger.info("[RLBOT_INPUT] BEGIN pressKey: " + keyCode + " (key name: " + KeyEvent.getKeyText(keyCode) + ")");
        
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            logger.info("[RLBOT_INPUT] Canvas retrieved: " + (canvas != null ? "success" : "null"));
            
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot press key");
                return;
            }
            
            try {
                logger.info("[RLBOT_INPUT] About to dispatch key event for key code: " + keyCode);
                dispatchKeyEvent(canvas, keyCode);
                logger.info("[RLBOT_INPUT] Key event dispatched successfully for key code: " + keyCode);
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] Exception during key press: " + e.getMessage() + ": " + e.toString());
            }
        });
        
        logger.info("[RLBOT_INPUT] END pressKey (clientThread invoked)");
    }
    
    /**
     * Dispatches key press and release events to the component.
     *
     * @param component The component to dispatch the events to
     * @param keyCode The key code to press
     */
    private void dispatchKeyEvent(Component component, int keyCode) {
        logger.info("[RLBOT_INPUT] BEGIN dispatchKeyEvent for key: " + keyCode + " (key name: " + KeyEvent.getKeyText(keyCode) + ")");
        
        long when = System.currentTimeMillis();
        int modifiers = 0;
        
        // First make sure component has focus
        logger.info("[RLBOT_INPUT] Requesting focus on component: " + component.getClass().getName());
        component.requestFocus();
        
        try {
            // Create key events
            logger.info("[RLBOT_INPUT] Creating press event with params: id=KEY_PRESSED, when=" + when + 
                        ", modifiers=" + modifiers + ", keyCode=" + keyCode);
            
            KeyEvent pressEvent = new KeyEvent(
                component,
                KeyEvent.KEY_PRESSED,
                when,
                modifiers,
                keyCode,
                KeyEvent.CHAR_UNDEFINED
            );
            
            // Determine if we need to send a KEY_TYPED event
            char keyChar = KeyEvent.CHAR_UNDEFINED;
            boolean sendTyped = false;
            
            if (keyCode >= KeyEvent.VK_0 && keyCode <= KeyEvent.VK_9) {
                keyChar = (char)('0' + (keyCode - KeyEvent.VK_0));
                sendTyped = true;
                logger.info("[RLBOT_INPUT] Will send KEY_TYPED event for number character: " + keyChar);
            } else if (keyCode >= KeyEvent.VK_A && keyCode <= KeyEvent.VK_Z) {
                keyChar = (char)('a' + (keyCode - KeyEvent.VK_A));
                sendTyped = true;
                logger.info("[RLBOT_INPUT] Will send KEY_TYPED event for letter character: " + keyChar);
            } else if (keyCode == KeyEvent.VK_SPACE) {
                keyChar = ' ';
                sendTyped = true;
                logger.info("[RLBOT_INPUT] Will send KEY_TYPED event for space character");
            } else {
                logger.info("[RLBOT_INPUT] No KEY_TYPED event needed for key: " + KeyEvent.getKeyText(keyCode));
            }
            
            KeyEvent typedEvent = null;
            if (sendTyped) {
                logger.info("[RLBOT_INPUT] Creating typed event with params: id=KEY_TYPED, when=" + (when + 10) + 
                            ", keyChar=" + keyChar);
                
                typedEvent = new KeyEvent(
                    component,
                    KeyEvent.KEY_TYPED,
                    when + 10,
                    modifiers,
                    KeyEvent.VK_UNDEFINED,
                    keyChar
                );
            }
            
            logger.info("[RLBOT_INPUT] Creating release event with params: id=KEY_RELEASED, when=" + (when + 50));
            KeyEvent releaseEvent = new KeyEvent(
                component,
                KeyEvent.KEY_RELEASED,
                when + 50,
                modifiers,
                keyCode,
                KeyEvent.CHAR_UNDEFINED
            );
            
            // Dispatch events with proper timing
            final KeyEvent finalTypedEvent = typedEvent;
            logger.info("[RLBOT_INPUT] About to dispatch press and potentially typed events via SwingUtilities.invokeAndWait");
            SwingUtilities.invokeAndWait(() -> {
                logger.info("[RLBOT_INPUT] Inside invokeAndWait, about to dispatch press event");
                component.dispatchEvent(pressEvent);
                logger.info("[RLBOT_INPUT] Press event dispatched");
                
                if (finalTypedEvent != null) {
                    logger.info("[RLBOT_INPUT] About to dispatch typed event");
                    component.dispatchEvent(finalTypedEvent);
                    logger.info("[RLBOT_INPUT] Typed event dispatched");
                }
            });
            // Removed blocking sleeps between key events
            // Thread.sleep(50);
            logger.info("[RLBOT_INPUT] About to dispatch release event via SwingUtilities.invokeAndWait");
            SwingUtilities.invokeAndWait(() -> {
                logger.info("[RLBOT_INPUT] Inside invokeAndWait, about to dispatch release event");
                component.dispatchEvent(releaseEvent);
                logger.info("[RLBOT_INPUT] Release event dispatched");
            });
            // Removed final sleep
            // Thread.sleep(50);
        } catch (Exception e) {
            logger.error("[RLBOT_INPUT] Error dispatching key events: " + e.getMessage() + ": " + e.toString());
        }
        
        logger.info("[RLBOT_INPUT] END dispatchKeyEvent");
    }
    
    /**
     * Type a string of characters.
     *
     * @param text The text to type
     */
    public void typeText(String text) {
        logger.info("Typing text: " + text);
        
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            if (canvas == null) {
                logger.error("Canvas is null, cannot type text");
                return;
            }
            
            for (char c : text.toCharArray()) {
                int keyCode = KeyEvent.getExtendedKeyCodeForChar(c);
                boolean isUpperCase = Character.isUpperCase(c);
                
                if (isUpperCase) {
                    // Press shift first
                    dispatchModifierKeyEvent(canvas, KeyEvent.VK_SHIFT, true);
                    // Removed sleep
                    // Thread.sleep(20);
                }
                
                // Type the character
                dispatchCharKeyEvent(canvas, keyCode, c);
                
                if (isUpperCase) {
                    // Release shift
                    dispatchModifierKeyEvent(canvas, KeyEvent.VK_SHIFT, false);
                }
                
                // Removed per-character sleep
                // Thread.sleep(50);
            }
            
            logger.info("Text typing completed: " + text);
        });
    }

    /**
     * Rotate/tilt camera by simulating a middle-mouse drag from viewport center.
     * Positive dx drags to the right (rotate right), positive dy drags downward (tilt down).
     */
    public void rotateCameraDrag(int dx, int dy) {
        logger.info("[RLBOT_INPUT] BEGIN rotateCameraDrag dx=" + dx + " dy=" + dy);
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot rotate camera drag");
                return;
            }
            int vx = client.getViewportXOffset();
            int vy = client.getViewportYOffset();
            int vw = client.getViewportWidth();
            int vh = client.getViewportHeight();
            int startX = vx + Math.max(10, vw / 2);
            int startY = vy + Math.max(10, vh / 2);
            Point start = new Point(startX, startY);
            Point end = new Point(startX + dx, startY + dy);
            try {
                dispatchMiddleDrag(canvas, start, end);
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] rotateCameraDrag error: " + e.getMessage() + ": " + e.toString());
            }
        });
        logger.info("[RLBOT_INPUT] END rotateCameraDrag (clientThread invoked)");
    }

    private void dispatchMiddleDrag(Component component, Point start, Point end) {
        long when = System.currentTimeMillis();
        int button = MouseEvent.BUTTON2;
        int modifiers = InputEvent.getMaskForButton(button);
        component.requestFocus();

        // Move to start
        dispatchMouseMoveEvent(component, start);

        // Press middle mouse
        MouseEvent press = new MouseEvent(
            component,
            MouseEvent.MOUSE_PRESSED,
            when,
            modifiers,
            start.x,
            start.y,
            1,
            false,
            button
        );
        SwingUtilities.invokeLater(() -> component.dispatchEvent(press));
        // sleepQuiet(20);

        // Drag in small steps
        int steps = 6;
        for (int i = 1; i <= steps; i++) {
            int x = start.x + (end.x - start.x) * i / steps;
            int y = start.y + (end.y - start.y) * i / steps;
            MouseEvent drag = new MouseEvent(
                component,
                MouseEvent.MOUSE_DRAGGED,
                when + 10L * i,
                modifiers,
                x,
                y,
                1,
                false,
                button
            );
            SwingUtilities.invokeLater(() -> component.dispatchEvent(drag));
            // sleepQuiet(10);
        }

        // Release
        MouseEvent release = new MouseEvent(
            component,
            MouseEvent.MOUSE_RELEASED,
            when + 10L * (steps + 2),
            modifiers,
            end.x,
            end.y,
            1,
            false,
            button
        );
        SwingUtilities.invokeLater(() -> component.dispatchEvent(release));
        // sleepQuiet(20);
    }

    private void sleepQuiet(long ms) {
        // Removed global sleeps to avoid lag; kept method for compatibility
        // try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    // Convenience camera controls
    public void rotateCameraLeftSmall() { pressKey(KeyEvent.VK_LEFT); }
    public void rotateCameraRightSmall() { pressKey(KeyEvent.VK_RIGHT); }
    public void tiltCameraUpSmall() { pressKey(KeyEvent.VK_PAGE_UP); }
    public void tiltCameraDownSmall() { pressKey(KeyEvent.VK_PAGE_DOWN); }
    
    /**
     * Dispatches a key event for a character.
     *
     * @param component The component to dispatch the events to
     * @param keyCode The key code
     * @param keyChar The character to type
     */
    private void dispatchCharKeyEvent(Component component, int keyCode, char keyChar) {
        long when = System.currentTimeMillis();
        int modifiers = Character.isUpperCase(keyChar) ? KeyEvent.SHIFT_DOWN_MASK : 0;
        
        try {
            // Create key events
            KeyEvent pressEvent = new KeyEvent(
                component,
                KeyEvent.KEY_PRESSED,
                when,
                modifiers,
                keyCode,
                KeyEvent.CHAR_UNDEFINED
            );
            
            KeyEvent typedEvent = new KeyEvent(
                component,
                KeyEvent.KEY_TYPED,
                when + 10,
                modifiers,
                KeyEvent.VK_UNDEFINED,
                keyChar
            );
            
            KeyEvent releaseEvent = new KeyEvent(
                component,
                KeyEvent.KEY_RELEASED,
                when + 50,
                modifiers,
                keyCode,
                KeyEvent.CHAR_UNDEFINED
            );
            
            // Dispatch events with proper timing
            SwingUtilities.invokeAndWait(() -> {
                component.dispatchEvent(pressEvent);
                component.dispatchEvent(typedEvent);
            });
            // Removed small delay between press and release
            // Thread.sleep(20);
            SwingUtilities.invokeAndWait(() -> {
                component.dispatchEvent(releaseEvent);
            });
        } catch (Exception e) {
            logger.error("Error dispatching character key events: " + e.getMessage() + ": " + e.toString());
        }
    }
    
    /**
     * Dispatches a modifier key event (press or release).
     *
     * @param component The component to dispatch the event to
     * @param keyCode The modifier key code
     * @param press True for press, false for release
     */
    private void dispatchModifierKeyEvent(Component component, int keyCode, boolean press) {
        long when = System.currentTimeMillis();
        int modifiers = 0;
        
        if (keyCode == KeyEvent.VK_SHIFT) {
            modifiers = KeyEvent.SHIFT_DOWN_MASK;
        } else if (keyCode == KeyEvent.VK_CONTROL) {
            modifiers = KeyEvent.CTRL_DOWN_MASK;
        } else if (keyCode == KeyEvent.VK_ALT) {
            modifiers = KeyEvent.ALT_DOWN_MASK;
        }
        
        try {
            // Create key event
            KeyEvent event = new KeyEvent(
                component,
                press ? KeyEvent.KEY_PRESSED : KeyEvent.KEY_RELEASED,
                when,
                press ? modifiers : 0, // Only include modifiers for press event
                keyCode,
                KeyEvent.CHAR_UNDEFINED
            );
            
            // Dispatch event
            SwingUtilities.invokeAndWait(() -> {
                component.dispatchEvent(event);
            });
        } catch (Exception e) {
            logger.error("Error dispatching modifier key event: " + e.getMessage() + ": " + e.toString());
        }
    }
    
    /**
     * Find the best click position for an object using RuneLite's convex hull.
     * This is much more reliable than guessing with offsets.
     *
     * @param centerPoint The center point of the object (fallback)
     * @param gameObject The game object to click on
     * @return The best click position found
     */
    private java.awt.Point findBestClickPosition(net.runelite.api.Point centerPoint, net.runelite.api.GameObject gameObject) {
        try {
            // Get the convex hull (actual clickable area) of the object
            Shape convexHull = gameObject.getConvexHull();
            if (convexHull != null) {
                // Get the center of the convex hull
                Rectangle bounds = convexHull.getBounds();
                if (bounds != null && bounds.width > 0 && bounds.height > 0) {
                    // Sample a few points inside the convex hull to avoid edges: center and midpoints
                    java.awt.Point[] candidates = new java.awt.Point[] {
                        new java.awt.Point(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2),
                        new java.awt.Point(bounds.x + bounds.width / 2, bounds.y + bounds.height / 3),
                        new java.awt.Point(bounds.x + bounds.width / 3, bounds.y + bounds.height / 2),
                        new java.awt.Point(bounds.x + (2*bounds.width) / 3, bounds.y + bounds.height / 2)
                    };
                    for (java.awt.Point c : candidates) {
                        if (convexHull.contains(c)) {
                            return c;
                        }
                    }
                    // Fallback to geometric center if samples fail
                    return new java.awt.Point(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
                }
            }
            
            // Fallback to the provided center point if convex hull is not available
            logger.info("[RLBOT_INPUT] Convex hull not available, using fallback center: (" + centerPoint.getX() + "," + centerPoint.getY() + ")");
            return new java.awt.Point(centerPoint.getX(), centerPoint.getY());
            
        } catch (Exception e) {
            logger.warn("[RLBOT_INPUT] Error getting convex hull: " + e.getMessage());
            // Fallback to the provided center point
            return new java.awt.Point(centerPoint.getX(), centerPoint.getY());
        }
    }
} 