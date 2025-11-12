package net.runelite.client.plugins.rlbot.input;

import java.awt.Canvas;
import java.awt.Point;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.inject.Inject;
import net.runelite.api.Client;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.input.KeyManager;
import net.runelite.client.input.MouseManager;
import net.runelite.client.plugins.rlbot.RLBotLogger;
import net.runelite.client.plugins.rlbot.RLBotPlugin;

/**
 * Facade for all input-related actions. Delegates to focused helpers for
 * mouse, keyboard, camera, target validation, and occlusion.
 */
public class RLBotInputHandler
{
    private final RLBotLogger logger;
    private final Client client;
    private final ClientThread clientThread;
    private final AtomicBoolean clickInProgress = new AtomicBoolean(false);

    // Helpers
    private final InputDispatcher dispatch;
    private final MouseController mouse;
    private final KeyboardController keyboard;
    private final CameraController camera;
    private final TargetInspector inspector;
    private OcclusionResolver occlusion; // depends on chatbox
    private ChatboxCollisionHandler chatboxCollisionHandler;
    private InteractionController interactions;

    @Inject
    public RLBotInputHandler(
        RLBotLogger logger,
        Client client,
        ClientThread clientThread,
        KeyManager keyManager,
        MouseManager mouseManager
    )
    {
        this.logger = logger;
        this.client = client;
        this.clientThread = clientThread;
        // key/mouse managers are wired into the dispatcher directly

        // Construct helpers with null plugin; set later in setPlugin
        this.dispatch = new InputDispatcher(logger, client, keyManager, mouseManager, null);
        this.mouse = new MouseController(logger, client, clientThread, dispatch);
        this.keyboard = new KeyboardController(logger, clientThread, dispatch);
        this.camera = new CameraController(logger, client, clientThread, dispatch, keyboard);
        this.inspector = new TargetInspector(logger, client);
        // occlusion + interactions created in setRLAgent once chatbox exists
    }

    public void initialize()
    {
        Canvas c = dispatch.getCanvas();
        if (c == null)
        {
            logger.warn("Canvas not yet available - will be initialized when needed");
        }
        else
        {
            try
            {
                if (c.isShowing())
                {
                    Point loc = c.getLocationOnScreen();
                    logger.info("Canvas found at location: " + loc.x + "," + loc.y);
                }
            }
            catch (Exception e)
            {
                logger.error("Error getting canvas location: " + e.getMessage());
            }
        }
    }

    public void setPlugin(RLBotPlugin plugin)
    {
        this.dispatch.setPlugin(plugin);
    }

    public void setRLAgent(net.runelite.client.plugins.rlbot.RLBotAgent agent)
    {
        this.chatboxCollisionHandler = new ChatboxCollisionHandler(client, this, logger, agent);
        this.occlusion = new OcclusionResolver(logger, client, chatboxCollisionHandler, camera);
        this.interactions = new InteractionController(logger, client, clientThread, mouse, inspector, occlusion);
        this.interactions.setAgent(agent);
    }

    // Validation and inspection
    public boolean validateTargetAtPoint(Point canvasPoint, String expectedAction)
    {
        if (occlusion != null && occlusion.isOccludedByUI(canvasPoint)) { return false; }
        return inspector.validateTargetAtPoint(canvasPoint, expectedAction);
    }

    public String getTargetInfoAtPoint(Point canvasPoint)
    {
        return inspector.describeTargetAt(canvasPoint);
    }

    // Mouse actions
    public void smoothMouseMove(Point canvasPoint)
    {
        mouse.smoothMouseMove(canvasPoint);
    }

    public boolean moveAndClickWithValidation(Point canvasPoint, String expectedAction)
    {
        clickInProgress.set(true);
        try { return interactions != null && interactions.moveAndClickWithValidation(canvasPoint, expectedAction); }
        finally { clickInProgress.set(false); }
    }

    public boolean clickWithValidation(String expectedAction)
    {
        clickInProgress.set(true);
        try { return interactions != null && interactions.clickWithValidation(expectedAction); }
        finally { clickInProgress.set(false); }
    }

    public boolean clickAtWithValidation(Point canvasPoint, String expectedAction)
    {
        clickInProgress.set(true);
        try { return interactions != null && interactions.clickAtWithValidation(canvasPoint, expectedAction); }
        finally { clickInProgress.set(false); }
    }

    public void click()
    {
        mouse.click();
    }

    public void clickAt(Point canvasPoint)
    {
        // best-effort chat collision handling
        if (chatboxCollisionHandler != null)
        {
            try { chatboxCollisionHandler.handleChatboxCollision(canvasPoint, null, "click"); }
            catch (Exception ignored) {}
        }
        mouse.clickAt(canvasPoint);
    }

    public void interactWithGameObject(net.runelite.api.GameObject gameObject, String action)
    {
        if (interactions != null) { interactions.interactWithGameObject(gameObject, action); }
    }

    public void rightClick()
    {
        mouse.rightClickCurrent();
    }

    // Keyboard
    public void pressKey(int keyCode) { keyboard.pressKey(keyCode); }
    public void typeText(String text) { keyboard.typeText(text); }

    // Camera
    public void rotateCameraDrag(int dx, int dy) { camera.rotateCameraDrag(dx, dy); }
    public void rotateCameraSafe(int dx, int dy) { camera.rotateCameraSafe(dx, dy); }
    public void rotateCameraLeftSmall() { camera.rotateCameraLeftSmall(); }
    public void rotateCameraRightSmall() { camera.rotateCameraRightSmall(); }
    public void tiltCameraUpSmall() { camera.tiltCameraUpSmall(); }
    public void tiltCameraDownSmall() { camera.tiltCameraDownSmall(); }
    public void zoomInSmall() { camera.zoomInSmall(); }
    public void zoomOutSmall() { camera.zoomOutSmall(); }

    public boolean isClickInProgress() { return clickInProgress.get(); }

    // Occlusion helpers exposed for diagnostics
    public boolean isOccludedByGeometry(Point canvasPoint, net.runelite.api.GameObject target)
    {
        return occlusion != null && occlusion.isOccludedByGeometry(canvasPoint, target);
    }

    public boolean isOccludedByGeometryWithDiagnostics(Point canvasPoint, net.runelite.api.GameObject target, boolean enableDiagnostics)
    {
        return occlusion != null && occlusion.isOccludedByGeometryWithDiagnostics(canvasPoint, target, enableDiagnostics);
    }

    public boolean revealPointByCameraGeometry(Point targetPoint, net.runelite.api.GameObject target, int attempts)
    {
        return occlusion != null && occlusion.revealPointByCameraGeometry(targetPoint, target, attempts);
    }
}
