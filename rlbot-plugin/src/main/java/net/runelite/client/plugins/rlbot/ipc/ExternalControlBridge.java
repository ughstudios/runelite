package net.runelite.client.plugins.rlbot.ipc;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
    private final List<Path> readDirs;

    private volatile long obsSeq = 0L;
    private volatile long lastActionSeqRead = -1L;
    private List<String> actionNames;
    private int stateDim = -1;
    private java.nio.file.attribute.FileTime lastActionFileTime;

    public ExternalControlBridge(RLBotLogger logger, String directory) {
        this.logger = logger;
        this.dir = Paths.get(directory);
        this.actionSpaceFile = dir.resolve("action_space.json");
        this.obsFile = dir.resolve("obs.json");
        this.actionFile = dir.resolve("action.json");
        this.readDirs = buildReadDirs(directory);
        ensureDir();
    }

    private void ensureDir() {
        try { Files.createDirectories(dir); } catch (IOException ignored) {}
    }

    public void resetActionSeq() {
        this.lastActionSeqRead = -1L;
        this.lastActionFileTime = null;
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

    public long publishObservation(float[] state, List<String> stateNames, Float lastReward, Integer lastActionIndex, String lastActionName) {
        return publishObservation(state, stateNames, lastReward, lastActionIndex, lastActionName, false);
    }

    public long publishObservation(float[] state, List<String> stateNames, Float lastReward, Integer lastActionIndex, String lastActionName, boolean done) {
        try {
            ensureDir();
            long ts = System.currentTimeMillis();
            long nextSeq = ++obsSeq;
            Map<String, Object> obs = new HashMap<>();
            obs.put("seq", nextSeq);
            obs.put("timestamp", ts);
            obs.put("state", state);
            if (stateNames != null && !stateNames.isEmpty()) {
                obs.put("state_names", stateNames);
            }
            obs.put("last_reward", lastReward != null ? lastReward : 0.0f);
            obs.put("last_action_index", lastActionIndex);
            obs.put("last_action_name", lastActionName);
            obs.put("done", done);
            writeJson(obsFile, obs);
            logger.debug("[IPC] Published obs+done seq=" + nextSeq + " lastIdx=" + lastActionIndex + " (" + lastActionName + ") r=" + (lastReward != null ? lastReward : 0.0f) + ", done=" + done);
            return nextSeq;
        } catch (Exception e) {
            logger.warn("[IPC] Failed publishing observation (with done): " + e.getMessage());
            return obsSeq;
        }
    }

    public Integer tryReadAction() {
        for (Path candidateDir : readDirs) {
            Integer act = tryReadActionFrom(candidateDir.resolve("action.json"));
            if (act != null) {
                return act;
            }
        }
        return null;
    }

    private Integer tryReadActionFrom(Path file) {
        try {
            if (file == null || !Files.exists(file)) return null;
            String s = Files.readString(file, StandardCharsets.UTF_8);
            if (s == null || s.isEmpty()) return null;
            Map<?, ?> m = gson.fromJson(s, Map.class);
            if (m == null) return null;
            Number seqN = (Number) m.get("seq");
            Number actN = (Number) m.get("action");
            if (seqN == null || actN == null) return null;
            long seq = seqN.longValue();
            int act = actN.intValue();
            FileTime currentTime = Files.getLastModifiedTime(file);
            if (seq <= lastActionSeqRead) {
                if (lastActionFileTime == null || currentTime.compareTo(lastActionFileTime) > 0) {
                    lastActionSeqRead = seq;
                    lastActionFileTime = currentTime;
                    logger.info("[IPC] Read action seq=" + seq + " -> action=" + act + " (dir=" + file.getParent() + ")");
                    return act;
                }
                return null;
            }
            lastActionSeqRead = seq;
            lastActionFileTime = currentTime;
            logger.info("[IPC] Read action seq=" + seq + " -> action=" + act + " (dir=" + file.getParent() + ")");
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

    private List<Path> buildReadDirs(String configured) {
        List<Path> dirs = new ArrayList<>();
        addDir(dirs, dir);
        try {
            Path repoDir = Paths.get("rlbot-ipc").toAbsolutePath().normalize();
            addDir(dirs, repoDir);
        } catch (Exception ignored) {}
        try {
            Path homeDir = Paths.get(System.getProperty("user.home"), ".runelite", "rlbot-ipc");
            addDir(dirs, homeDir);
        } catch (Exception ignored) {}
        try {
            Path tmpDir = Paths.get(System.getProperty("java.io.tmpdir"), "rlbot-ipc");
            addDir(dirs, tmpDir);
        } catch (Exception ignored) {}
        if (configured != null && !configured.isBlank()) {
            try {
                Path configuredPath = Paths.get(configured);
                if (!configuredPath.isAbsolute()) {
                    configuredPath = Paths.get("").toAbsolutePath().normalize().resolve(configuredPath);
                }
                addDir(dirs, configuredPath);
            } catch (Exception ignored) {}
        }
        return dirs;
    }

    private void addDir(List<Path> dirs, Path candidate) {
        if (candidate == null) {
            return;
        }
        Path normalized = candidate.toAbsolutePath().normalize();
        for (Path existing : dirs) {
            if (Objects.equals(normalized, existing)) {
                return;
            }
        }
        dirs.add(normalized);
    }
}
