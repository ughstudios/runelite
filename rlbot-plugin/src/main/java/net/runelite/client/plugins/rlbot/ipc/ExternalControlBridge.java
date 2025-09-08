package net.runelite.client.plugins.rlbot.ipc;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.runelite.client.plugins.rlbot.RLBotLogger;

/**
 * File-based IPC bridge for driving actions from an external controller (e.g., OpenAI Gym).
 *
 * Protocol:
 * - action_space.json: { "state_dim": int, "actions": ["Name0", ...] }
 * - obs.json: { "seq": long, "timestamp": long, "state": [float...], "last_reward": float,
 *              "last_action_index": int|null, "last_action_name": String|null }
 * - action.json: { "seq": long, "action": int }
 */
public final class ExternalControlBridge {
    private final RLBotLogger logger;
    private final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private final Path dir;
    private final Path actionSpaceFile;
    private final Path obsFile;
    private final Path actionFile;

    private volatile long obsSeq = 0L;
    private volatile long lastActionSeqRead = -1L;
    private List<String> actionNames;
    private int stateDim = -1;

    public ExternalControlBridge(RLBotLogger logger, String directory) {
        this.logger = logger;
        this.dir = Paths.get(directory);
        this.actionSpaceFile = dir.resolve("action_space.json");
        this.obsFile = dir.resolve("obs.json");
        this.actionFile = dir.resolve("action.json");
        ensureDir();
    }

    private void ensureDir() {
        try { Files.createDirectories(dir); } catch (IOException ignored) {}
    }

    public void writeActionSpaceIfNeeded(List<String> actionNames, int stateDim) {
        try {
            if (actionNames == null || actionNames.isEmpty() || stateDim <= 0) return;
            boolean changed = this.actionNames == null || this.stateDim != stateDim || !this.actionNames.equals(actionNames);
            if (!changed && Files.exists(actionSpaceFile)) return;
            this.actionNames = actionNames;
            this.stateDim = stateDim;
            Map<String, Object> space = new HashMap<>();
            space.put("state_dim", stateDim);
            space.put("actions", actionNames);
            writeJson(actionSpaceFile, space);
            logger.info("[IPC] Wrote action_space.json with " + actionNames.size() + " actions, stateDim=" + stateDim);
        } catch (Exception e) {
            logger.warn("[IPC] Failed writing action space: " + e.getMessage());
        }
    }

    public long publishObservation(float[] state, Float lastReward, Integer lastActionIndex, String lastActionName) {
        try {
            ensureDir();
            long ts = System.currentTimeMillis();
            long nextSeq = ++obsSeq;
            Map<String, Object> obs = new HashMap<>();
            obs.put("seq", nextSeq);
            obs.put("timestamp", ts);
            obs.put("state", state);
            obs.put("last_reward", lastReward != null ? lastReward : 0.0f);
            obs.put("last_action_index", lastActionIndex);
            obs.put("last_action_name", lastActionName);
            writeJson(obsFile, obs);
            return nextSeq;
        } catch (Exception e) {
            logger.warn("[IPC] Failed publishing observation: " + e.getMessage());
            return obsSeq;
        }
    }

    public Integer tryReadAction() {
        try {
            if (!Files.exists(actionFile)) return null;
            String s = Files.readString(actionFile, StandardCharsets.UTF_8);
            if (s == null || s.isEmpty()) return null;
            Map<?, ?> m = gson.fromJson(s, Map.class);
            if (m == null) return null;
            Number seqN = (Number) m.get("seq");
            Number actN = (Number) m.get("action");
            if (seqN == null || actN == null) return null;
            long seq = seqN.longValue();
            int act = actN.intValue();
            if (seq <= lastActionSeqRead) return null; // already handled
            lastActionSeqRead = seq;
            return act;
        } catch (Exception e) {
            logger.warn("[IPC] Failed reading action: " + e.getMessage());
            return null;
        }
    }

    private void writeJson(Path file, Object obj) throws IOException {
        String tmpName = file.getFileName().toString() + ".tmp";
        Path tmp = file.getParent().resolve(tmpName);
        Files.writeString(tmp, gson.toJson(obj), StandardCharsets.UTF_8);
        Files.move(tmp, file, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
    }
}

