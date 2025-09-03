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
     * Move the mouse to the specified point on the game canvas and dispatch appropriate events.
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
        
        // Get the canvas and dispatch events on the client thread
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            logger.debug("[RLBOT_INPUT] Canvas retrieved: " + (canvas != null ? "success" : "null"));
            
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot move mouse");
                return;
            }
            
            try {
                logger.debug("[RLBOT_INPUT] Dispatching mouse move event to: " + canvasPoint.x + "," + canvasPoint.y);
                dispatchMouseMoveEvent(canvas, canvasPoint);
                lastCanvasMovePoint = new Point(canvasPoint);
                logger.debug("[RLBOT_INPUT] Mouse move event dispatched successfully");
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] Exception during mouse move: " + e.getMessage() + ": " + e.toString());
            }
        });
        
        logger.debug("[RLBOT_INPUT] END smoothMouseMove (clientThread invoked)");
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
        
        // Use SwingUtilities.invokeLater to avoid blocking the main thread
        try {
            logger.debug("[RLBOT_INPUT] Dispatching mouse move event via SwingUtilities.invokeLater");
            SwingUtilities.invokeLater(() -> {
                component.dispatchEvent(event);
                logger.debug("[RLBOT_INPUT] Mouse move event dispatched");
            });
            // Add a small delay after mouse movement for human-like behavior
            try {
                Thread.sleep(25);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        } catch (Exception e) {
            logger.error("[RLBOT_INPUT] Error dispatching mouse move event: " + e.getMessage() + ": " + e.toString());
        }
        
        logger.info("[RLBOT_INPUT] END dispatchMouseMoveEvent");
    }
    
    /**
     * Click at the current mouse position.
     */
    public void click() {
        logger.debug("[RLBOT_INPUT] BEGIN click at current mouse position");
        
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            logger.debug("[RLBOT_INPUT] Canvas retrieved: " + (canvas != null ? "success" : "null"));
            
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot click");
                return;
            }
            
            // Prefer the last moved point; fall back to canvas.getMousePosition; then center
            Point clickPoint = lastCanvasMovePoint;
            if (clickPoint == null) {
                try {
                    Point mousePos = canvas.getMousePosition();
                    logger.debug("[RLBOT_INPUT] Canvas.getMousePosition returned: " + (mousePos != null ? mousePos.x + "," + mousePos.y : "null"));
                    if (mousePos != null) clickPoint = mousePos;
                } catch (Exception e) {
                    logger.error("[RLBOT_INPUT] Error getting mouse position: " + e.getMessage() + ": " + e.toString());
                }
            }
            if (clickPoint == null) {
                logger.warn("[RLBOT_INPUT] No known mouse point; using canvas center");
                clickPoint = new Point(canvas.getWidth() / 2, canvas.getHeight() / 2);
            }

            try {
                logger.debug("[RLBOT_INPUT] Dispatching click event at: " + clickPoint.x + "," + clickPoint.y);
                dispatchMouseClickEvent(canvas, clickPoint);
                // Clear lastCanvasMovePoint so the next action will recompute a fresh target
                lastCanvasMovePoint = null;
                logger.debug("[RLBOT_INPUT] Click event dispatched successfully");
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] Exception during click: " + e.getMessage() + ": " + e.toString());
            }
        });
        
        logger.debug("[RLBOT_INPUT] END click (clientThread invoked)");
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
        
        logger.info("[RLBOT_INPUT] BEGIN clickAt point: " + canvasPoint.x + "," + canvasPoint.y);
        clientThread.invoke(() -> {
            Canvas canvas = getCanvas();
            logger.info("[RLBOT_INPUT] Canvas retrieved: " + (canvas != null ? "success" : "null"));
            if (canvas == null) {
                logger.error("[RLBOT_INPUT] Canvas is null, cannot clickAt");
                return;
            }
            try {
                logger.info("[RLBOT_INPUT] About to dispatch clickAt event at: " + canvasPoint.x + "," + canvasPoint.y);
                dispatchMouseClickEvent(canvas, canvasPoint);
                lastCanvasMovePoint = null;
                logger.info("[RLBOT_INPUT] clickAt dispatched successfully at: " + canvasPoint.x + "," + canvasPoint.y);
            } catch (Exception e) {
                logger.error("[RLBOT_INPUT] Exception during clickAt: " + e.getMessage() + ": " + e.toString());
            }
        });
        logger.info("[RLBOT_INPUT] END clickAt (clientThread invoked)");
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
        
        try {
            // Create mouse events
            logger.info("[RLBOT_INPUT] Creating press event with params: id=MOUSE_PRESSED, when=" + when + 
                        ", modifiers=" + modifiers + ", point=(" + point.x + "," + point.y + 
                        "), clickCount=" + clickCount + ", popupTrigger=" + popupTrigger + 
                        ", button=BUTTON1");
            
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
            
            logger.info("[RLBOT_INPUT] Creating release event with params: id=MOUSE_RELEASED, when=" + (when + 50));
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
            
            logger.info("[RLBOT_INPUT] Creating click event with params: id=MOUSE_CLICKED, when=" + (when + 51));
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
            
            // Dispatch events with proper timing using non-blocking calls
            logger.debug("[RLBOT_INPUT] Dispatching press event via SwingUtilities.invokeLater");
            SwingUtilities.invokeLater(() -> {
                component.dispatchEvent(pressEvent);
                logger.debug("[RLBOT_INPUT] Press event dispatched");
            });
            
            // Small delay between press and release for human-like behavior
            try {
                Thread.sleep(30);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            logger.debug("[RLBOT_INPUT] Dispatching release and click events via SwingUtilities.invokeLater");
            SwingUtilities.invokeLater(() -> {
                component.dispatchEvent(releaseEvent);
                logger.debug("[RLBOT_INPUT] Release event dispatched");
                component.dispatchEvent(clickEvent);
                logger.debug("[RLBOT_INPUT] Click event dispatched");
            });
            
            // Add delay after click for human-like behavior
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        } catch (Exception e) {
            logger.error("[RLBOT_INPUT] Error dispatching mouse click events: " + e.getMessage() + ": " + e.toString());
        }
        
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

                // Move mouse to the object first to ensure proper targeting
                net.runelite.api.Point canvasPoint = net.runelite.api.Perspective.localToCanvas(client, lp, 0);
                if (canvasPoint != null) {
                    logger.info("[RLBOT_INPUT] Moving mouse to object at canvas point: (" + canvasPoint.getX() + "," + canvasPoint.getY() + ")");
                    
                    // Try to find a better click position by checking multiple points around the object
                    java.awt.Point bestClickPoint = findBestClickPosition(canvasPoint, gameObject);
                    logger.info("[RLBOT_INPUT] Using optimized click position: (" + bestClickPoint.x + "," + bestClickPoint.y + ")");
                    
                    smoothMouseMove(bestClickPoint);
                    
                    // Actually click on the object to open the context menu
                    logger.info("[RLBOT_INPUT] Clicking on object to open context menu");
                    clickAt(bestClickPoint);
                }

                // Invoke the menu action with scene coords (this is the primary interaction method)
                logger.info("[RLBOT_INPUT] About to invoke menu action: sceneX=" + sceneX + ", sceneY=" + sceneY + ", menuAction=" + menuAction + ", objectId=" + gameObject.getId() + ", action='" + desired + "'");
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
                    java.awt.Point hullCenter = new java.awt.Point(
                        bounds.x + bounds.width / 2,
                        bounds.y + bounds.height / 2
                    );
                    
                    logger.info("[RLBOT_INPUT] Using convex hull center: (" + hullCenter.x + "," + hullCenter.y + ") for object " + gameObject.getId());
                    return hullCenter;
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