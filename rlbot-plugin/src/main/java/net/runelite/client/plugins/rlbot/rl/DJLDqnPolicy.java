package net.runelite.client.plugins.rlbot.rl;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Lightweight, dependency-free DQN-like policy.
 *
 * Replaces the previous DJL-based implementation so the plugin
 * compiles without pulling in external ML libraries. This keeps a
 * simple linear function approximator per action and trains it with
 * a TD(0) update. It is intentionally minimal and only aims to
 * preserve the API shape used by RLBotAgent.
 */
public final class DJLDqnPolicy implements AutoCloseable {
    private final int stateDim;
    private final int numActions;

    // Linear weights per action: Q(s, a) = dot(weights[a], s)
    private final float[][] weights;

    // Simple Adam-like parameters omitted; use constant LR for stability
    private final float learningRate = 0.01f;

    // Reusable buffer to avoid allocations on predict
    private final float[] qBuffer;

    public DJLDqnPolicy(int stateDim, int numActions) {
        this.stateDim = Math.max(1, stateDim);
        this.numActions = Math.max(1, numActions);
        this.weights = new float[this.numActions][this.stateDim];
        this.qBuffer = new float[this.numActions];
    }

    // Compute Q-values for all actions for a given state
    public float[] predictQ(float[] state) {
        if (state == null || state.length != stateDim) {
            // Return zeros on malformed input
            Arrays.fill(qBuffer, 0f);
            return qBuffer.clone();
        }
        for (int a = 0; a < numActions; a++) {
            float q = 0f;
            float[] wa = weights[a];
            for (int j = 0; j < stateDim; j++) {
                q += wa[j] * state[j];
            }
            qBuffer[a] = q;
        }
        return qBuffer.clone();
    }

    // One-step TD(0) update with linear function approximator.
    public float updateBatch(
        float[][] states, int[] actions, float[] rewards,
        float[][] nextStates, boolean[] done, float gamma
    ) {
        if (states == null || actions == null || rewards == null || nextStates == null || done == null) return 0f;
        int n = Math.min(states.length,
                Math.min(actions.length, Math.min(rewards.length, Math.min(nextStates.length, done.length))));
        if (n <= 0) return 0f;
        gamma = Math.max(0f, Math.min(1f, gamma));
        double sumSqErr = 0.0;

        for (int i = 0; i < n; i++) {
            float[] s = states[i];
            float[] ns = nextStates[i];
            int a = Math.max(0, Math.min(numActions - 1, actions[i]));
            float r = rewards[i];
            boolean terminal = done[i];

            // Current Q estimate for taken action
            float pred = 0f;
            float[] wa = weights[a];
            if (s != null && s.length == stateDim) {
                for (int j = 0; j < stateDim; j++) pred += wa[j] * s[j];
            }

            // Max over next actions
            float nextMax = 0f;
            if (!terminal && ns != null && ns.length == stateDim) {
                for (int a2 = 0; a2 < numActions; a2++) {
                    float q = 0f;
                    float[] wb = weights[a2];
                    for (int j = 0; j < stateDim; j++) q += wb[j] * ns[j];
                    if (a2 == 0 || q > nextMax) nextMax = q;
                }
            }

            float target = terminal ? r : r + gamma * nextMax;
            float err = target - pred;
            sumSqErr += (double) err * (double) err;

            // Gradient descent step: d/dw = -err * s
            if (s != null && s.length == stateDim) {
                for (int j = 0; j < stateDim; j++) {
                    wa[j] += learningRate * err * s[j];
                }
            }
        }
        return (float) (sumSqErr / n);
    }

    // Kept for API compatibility; no moving target in this minimal version
    public void syncTarget() { /* no-op */ }

    // Periodically persist a tiny text file with weights
    public void saveIfNeeded(long steps) {
        if (steps <= 0 || steps % 5000L != 0) return;
        try {
            Path outDir = Paths.get("models");
            Files.createDirectories(outDir);
            saveTo(outDir);
        } catch (Exception ignored) { }
    }

    public boolean saveTo(Path dir) {
        try {
            Files.createDirectories(dir);
            Path file = dir.resolve("rlbot-dqn.txt");
            try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
                w.write(stateDim + "," + numActions);
                w.newLine();
                for (int a = 0; a < numActions; a++) {
                    for (int j = 0; j < stateDim; j++) {
                        if (j > 0) w.write(",");
                        w.write(Float.toString(weights[a][j]));
                    }
                    w.newLine();
                }
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    public boolean loadFrom(Path dir) {
        try {
            Path file = dir.resolve("rlbot-dqn.txt");
            if (!Files.exists(file)) return false;
            try (BufferedReader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
                String header = r.readLine();
                if (header == null) return false;
                String[] hp = header.split(",");
                int sd = Integer.parseInt(hp[0]);
                int na = Integer.parseInt(hp[1]);
                if (sd != stateDim || na != numActions) return false; // shape mismatch
                for (int a = 0; a < numActions; a++) {
                    String line = r.readLine();
                    if (line == null) return false;
                    String[] parts = line.split(",");
                    if (parts.length != stateDim) return false;
                    for (int j = 0; j < stateDim; j++) {
                        weights[a][j] = Float.parseFloat(parts[j]);
                    }
                }
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public void close() { /* no resources */ }
}

