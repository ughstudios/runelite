package net.runelite.client.plugins.rlbot.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import javax.inject.Inject;
import net.runelite.client.plugins.rlbot.RLBotAgent;
import net.runelite.client.plugins.rlbot.RLBotConfig;
import net.runelite.client.plugins.rlbot.RLBotTelemetry;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;
import net.runelite.client.ui.overlay.OverlayPanel;
import net.runelite.client.ui.overlay.OverlayPosition;
import net.runelite.client.ui.overlay.OverlayPriority;
import net.runelite.client.ui.overlay.components.LineComponent;
import net.runelite.client.ui.overlay.components.TitleComponent;

public class RLBotOverlay extends OverlayPanel
{
    private final RLBotConfig config;
    private final RLBotTelemetry telemetry;
    private final RLBotAgent agent;

    @Inject
    public RLBotOverlay(RLBotConfig config, RLBotTelemetry telemetry, RLBotAgent agent)
    {
        this.config = config;
        this.telemetry = telemetry;
        this.agent = agent;
        setPosition(OverlayPosition.TOP_LEFT);
        setPriority(OverlayPriority.HIGH);
    }

    @Override
    public Dimension render(Graphics2D graphics)
    {
        if (!config.showOverlay())
        {
            return null;
        }

        panelComponent.getChildren().clear();
        panelComponent.getChildren().add(TitleComponent.builder()
            .text("RLBot Status")
            .color(Color.CYAN)
            .build());

        panelComponent.getChildren().add(LineComponent.builder()
            .left("Mode")
            .right(telemetry.getMode())
            .build());

        panelComponent.getChildren().add(LineComponent.builder()
            .left("Last reward")
            .right(String.format("%.2f", agent.getLastReward()))
            .build());

        panelComponent.getChildren().add(LineComponent.builder()
            .left("Episode return")
            .right(String.format("%.1f", agent.getEpisodeReturn()))
            .build());

        panelComponent.getChildren().add(LineComponent.builder()
            .left("Steps")
            .right(Long.toString(agent.getSteps()))
            .build());

        TaskContext ctx = agent.getTaskContext();
        if (ctx != null)
        {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Free slots")
                .right(Integer.toString(ctx.getInventoryFreeSlots()))
                .build());
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Woodcutting")
                .right(ctx.isWoodcuttingAnim() ? "yes" : "no")
                .build());
        }

        int lastIdx = agent.getLastChosenAction();
        if (lastIdx >= 0)
        {
            panelComponent.getChildren().add(LineComponent.builder()
                .left("Last action")
                .right(agent.getActionName(lastIdx))
                .build());
        }

        return super.render(graphics);
    }
}
