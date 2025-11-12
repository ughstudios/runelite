package net.runelite.client.plugins.rlbot;

import com.google.inject.Provides;
import java.awt.Point;
import javax.inject.Inject;
import net.runelite.api.Client;
import net.runelite.api.events.ClientTick;
import net.runelite.api.events.GameTick;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.config.ConfigManager;
import net.runelite.client.eventbus.Subscribe;
import net.runelite.client.input.KeyManager;
import net.runelite.client.input.MouseManager;
import net.runelite.client.plugins.Plugin;
import net.runelite.client.plugins.PluginDescriptor;
import net.runelite.client.plugins.rlbot.input.RLBotInputHandler;
import net.runelite.client.plugins.rlbot.ui.RLBotOverlay;
import net.runelite.client.ui.overlay.OverlayManager;

@PluginDescriptor(
    name = "RLBot (Dev)",
    description = "RuneLite bot helper for woodcutting/banking",
    tags = {"bot", "ai"},
    enabledByDefault = true
)
public class RLBotPlugin extends Plugin
{
    @Inject
    private Client client;

    @Inject
    private ClientThread clientThread;

    @Inject
    private KeyManager keyManager;

    @Inject
    private MouseManager mouseManager;

    @Inject
    private RLBotConfig config;

    @Inject
    private RLBotAgent agent;

    @Inject
    private RLBotInputHandler inputHandler;

    @Inject
    private OverlayManager overlayManager;

    @Inject
    private RLBotOverlay overlay;

    @Inject
    private RLBotCursorOverlay cursorOverlay;

    private volatile Point lastMouseLocation;
    private volatile Point lastSystemMouseLocation;
    private volatile long lastSystemMouseLocationMillis;
    private volatile Point lastSyntheticMouseLocation;
    private volatile long lastSyntheticMouseLocationMillis;

    @Provides
    RLBotConfig provideConfig(ConfigManager configManager)
    {
        return configManager.getConfig(RLBotConfig.class);
    }

    @Override
    protected void startUp()
    {
        inputHandler.setPlugin(this);
        inputHandler.initialize();
        inputHandler.setRLAgent(agent);
        overlayManager.add(overlay);
        overlayManager.add(cursorOverlay);
    }

    @Override
    protected void shutDown()
    {
        overlayManager.remove(overlay);
        overlayManager.remove(cursorOverlay);
    }

    @Subscribe
    public void onGameTick(GameTick tick)
    {
        agent.onTick();
    }

    @Subscribe
    public void onClientTick(ClientTick tick)
    {
        try
        {
            java.awt.PointerInfo info = java.awt.MouseInfo.getPointerInfo();
            if (info != null)
            {
                updateLastSystemMouseLocationIfChanged(info.getLocation());
            }
        }
        catch (Exception ignored)
        {
        }
    }

    public void updateLastSystemMouseLocationIfChanged(Point point)
    {
        if (point == null)
        {
            return;
        }
        if (lastSystemMouseLocation != null && lastSystemMouseLocation.equals(point))
        {
            return;
        }
        lastMouseLocation = new Point(point);
        lastSystemMouseLocation = new Point(point);
        lastSystemMouseLocationMillis = System.currentTimeMillis();
    }

    public void updateLastSyntheticMouseLocation(Point point)
    {
        if (point == null)
        {
            return;
        }
        lastSyntheticMouseLocation = new Point(point);
        lastSyntheticMouseLocationMillis = System.currentTimeMillis();
    }

    Point getLastMouseLocation()
    {
        Point synthetic = lastSyntheticMouseLocation;
        if (synthetic != null)
        {
            return synthetic;
        }
        return lastMouseLocation;
    }
}
