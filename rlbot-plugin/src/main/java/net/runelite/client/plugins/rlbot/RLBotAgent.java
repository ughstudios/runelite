package net.runelite.client.plugins.rlbot;

import com.google.gson.Gson;
import com.google.inject.Singleton;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import javax.inject.Inject;
import net.runelite.api.Client;
import net.runelite.api.GameState;
import net.runelite.client.callback.ClientThread;
import net.runelite.client.plugins.rlbot.input.RLBotInputHandler;
import net.runelite.client.plugins.rlbot.ipc.ExternalControlBridge;
import net.runelite.client.plugins.rlbot.tasks.BankDepositTask;
import net.runelite.client.plugins.rlbot.tasks.CameraRotateLeftTask;
import net.runelite.client.plugins.rlbot.tasks.CameraRotateRightTask;
import net.runelite.client.plugins.rlbot.tasks.CameraZoomInTask;
import net.runelite.client.plugins.rlbot.tasks.CameraZoomOutTask;
import net.runelite.client.plugins.rlbot.tasks.ChopNearestTreeTask;
import net.runelite.client.plugins.rlbot.tasks.IdleTask;
import net.runelite.client.plugins.rlbot.tasks.NavigateToBankHotspotTask;
import net.runelite.client.plugins.rlbot.tasks.NavigateToTreeHotspotTask;
import net.runelite.client.plugins.rlbot.tasks.Task;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;
import net.runelite.client.plugins.rlbot.RLBotState.Snapshot;

@Singleton
public class RLBotAgent
{
    // Observation feature names live in RLBotState to keep this class lean

    private final Client client;
    private final RLBotLogger logger;
    private final RLBotConfig config;
    private final RLBotTelemetry telemetry;
    private final TaskContext taskContext;
    private final List<Task> tasks;

    private ExternalControlBridge external;
    private long lastStepMillis;
    private long steps;
    private double episodeReturn;
    private long episodeStartMs;
    private float lastReward;

    private int lastActionIndex = -1;
    private final LinkedList<Integer> recentActions = new LinkedList<>();
    private final int[] actionCounts;

    private Snapshot previousSnapshot;
    private Integer previousWoodcutXp;

    private final AtomicInteger externalPenaltyMilli = new AtomicInteger(0);

    private final Gson gson = new Gson();
    private final Path clickRequestFile;
    private long lastClickTimestamp = -1L;

    // Snapshot moved to RLBotState

    @Inject
    public RLBotAgent(
        Client client,
        ClientThread clientThread,
        RLBotLogger logger,
        RLBotConfig config,
        RLBotInputHandler inputHandler,
        RLBotTelemetry telemetry
    )
    {
        this.client = client;
        this.logger = logger;
        this.config = config;
        this.telemetry = telemetry;
        this.taskContext = new TaskContext(client, clientThread, logger, inputHandler, config, telemetry);
        this.tasks = new ArrayList<>();
        tasks.add(new BankDepositTask());
        tasks.add(new NavigateToBankHotspotTask());
        tasks.add(new ChopNearestTreeTask());
        tasks.add(new NavigateToTreeHotspotTask());
        tasks.add(new IdleTask());
        tasks.add(new net.runelite.client.plugins.rlbot.tasks.CameraRotateLeftTask());
        tasks.add(new net.runelite.client.plugins.rlbot.tasks.CameraRotateRightTask());
        tasks.add(new net.runelite.client.plugins.rlbot.tasks.CameraZoomInTask());
        tasks.add(new net.runelite.client.plugins.rlbot.tasks.CameraZoomOutTask());
        this.actionCounts = new int[tasks.size()];
        this.episodeStartMs = System.currentTimeMillis();

        Path ipcDir = Paths.get(config.gymIpcDir());
        this.clickRequestFile = ipcDir.resolve("click_request.json");
        try
        {
            Files.createDirectories(ipcDir);
        }
        catch (IOException ignored)
        {
        }
    }

    public void onTick()
    {
        if (!config.enableGymControl())
        {
            return;
        }
        if (client.getGameState() != GameState.LOGGED_IN)
        {
            return;
        }
        long now = System.currentTimeMillis();
        int interval = Math.max(100, config.gymStepIntervalMs());
        if (now - lastStepMillis < interval)
        {
            return;
        }
        lastStepMillis = now;

        handleClickRequest();

        if (external == null)
        {
            external = new ExternalControlBridge(logger, config.gymIpcDir());
        }

        Snapshot snapshot = RLBotState.capture(taskContext, client);
        float[] stateVector = RLBotState.buildStateVector(snapshot);
        external.writeActionSpaceIfNeeded(getActionNames(), stateVector.length);

        // Execute requested action first so the published observation's last_action reflects it
        Integer requested = external.tryReadAction();
        if (requested == null)
        {
            // If no external action is requested, but we have a staged hover pending
            // (e.g., bank click staged last tick), rerun the corresponding task
            int bankIdx = indexOfTask(net.runelite.client.plugins.rlbot.tasks.BankDepositTask.class);
            if (bankIdx >= 0 && net.runelite.client.plugins.rlbot.tasks.BankClicker.isPending())
            {
                requested = bankIdx;
            }
        }

        if (requested != null)
        {
            RLBotActionRunner.Result res = RLBotActionRunner.run(
                tasks,
                requested,
                taskContext,
                telemetry,
                logger
            );
            lastActionIndex = res.executedIndex;
            if (!res.eligible)
            {
                addExternalPenalty(0.2f);
            }
        }
        else
        {
            logger.debug("[IPC] No action pending");
        }

        float reward = RLBotReward.compute(previousSnapshot, snapshot, previousWoodcutXp);
        reward -= drainExternalPenalty();
        lastReward = reward;
        episodeReturn += reward;
        steps++;
        telemetry.addReward(Math.round(reward * 100));
        telemetry.incEpisodeSteps();

        String lastName = lastActionIndex >= 0 && lastActionIndex < tasks.size()
            ? tasks.get(lastActionIndex).getClass().getSimpleName()
            : null;
        boolean doneNow = RLBotEpisode.shouldTerminateEpisode(previousSnapshot, snapshot, episodeStartMs, now);
        external.publishObservation(stateVector, RLBotState.STATE_FEATURE_NAMES, reward, lastActionIndex, lastName, doneNow);

        previousSnapshot = snapshot;
        previousWoodcutXp = snapshot.woodcutXp;

        if (doneNow)
        {
            // Reset episodic accounting
            steps = 0L;
            episodeReturn = 0.0;
            episodeStartMs = now;
            telemetry.resetEpisode();
            // Clear last action to avoid leaking across episodes
            lastActionIndex = -1;
        }
    }

    private int indexOfTask(Class<? extends Task> cls)
    {
        for (int i = 0; i < tasks.size(); i++)
        {
            if (tasks.get(i).getClass() == cls)
            {
                return i;
            }
        }
        return -1;
    }

    // Action execution moved to RLBotActionRunner

    private List<String> getActionNames()
    {
        List<String> names = new ArrayList<>(tasks.size());
        for (Task task : tasks)
        {
            names.add(task.getClass().getSimpleName());
        }
        return names;
    }

    // World queries moved to RLBotState

    private float drainExternalPenalty()
    {
        int milli = externalPenaltyMilli.getAndSet(0);
        return milli / 1000f;
    }

    public void addExternalPenalty(float penalty)
    {
        int milli = Math.max(0, (int) Math.round(penalty * 1000f));
        externalPenaltyMilli.addAndGet(milli);
    }

    public TaskContext getTaskContext()
    {
        return taskContext;
    }

    public int getLastChosenAction()
    {
        return lastActionIndex;
    }

    public int getNumActions()
    {
        return tasks.size();
    }

    public String getActionName(int idx)
    {
        if (idx < 0 || idx >= tasks.size())
        {
            return "Action" + idx;
        }
        return tasks.get(idx).getClass().getSimpleName();
    }

    public LinkedList<Integer> getRecentActions()
    {
        return recentActions;
    }

    public long getSteps()
    {
        return steps;
    }

    public double getEpisodeReturn()
    {
        return episodeReturn;
    }

    public long getEpisodeStartMs()
    {
        return episodeStartMs;
    }

    public float getLastReward()
    {
        return lastReward;
    }

    private void handleClickRequest()
    {
        if (clickRequestFile == null || !Files.exists(clickRequestFile))
        {
            return;
        }

        try
        {
            String content = Files.readString(clickRequestFile, StandardCharsets.UTF_8);
            if (content == null || content.isBlank())
            {
                return;
            }
            Map<?, ?> payload = gson.fromJson(content, Map.class);
            if (payload == null)
            {
                return;
            }
            Number ts = (Number) payload.get("ts");
            Number x = (Number) payload.get("x");
            Number y = (Number) payload.get("y");
            if (ts == null || x == null || y == null)
            {
                return;
            }
            long timestamp = ts.longValue();
            if (timestamp <= lastClickTimestamp)
            {
                return;
            }
            lastClickTimestamp = timestamp;
            java.awt.Point target = new java.awt.Point(x.intValue(), y.intValue());
            boolean success = taskContext.input.clickAt(target);
            if (success)
            {
                logger.info("[IPC] Executed manual click at " + target);
            }
            else
            {
                logger.warn("[IPC] Failed to dispatch manual click at " + target);
            }
        }
        catch (Exception e)
        {
            logger.warn("[IPC] Failed executing manual click request: " + e.getMessage());
        }
        finally
        {
            try
            {
                Files.deleteIfExists(clickRequestFile);
            }
            catch (IOException ignored)
            {
            }
        }
    }
}
