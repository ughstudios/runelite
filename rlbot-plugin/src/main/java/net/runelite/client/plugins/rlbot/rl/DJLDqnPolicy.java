package net.runelite.client.plugins.rlbot.rl;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

/** DJL-based DQN with online model and placeholder target sync. */
public final class DJLDqnPolicy implements AutoCloseable {
    private final int stateDim;
    private final int numActions;
    private final Model online;
    private final Model target;
    private final NDManager baseManager;
    private final Trainer trainer;
    private final Loss loss = Loss.l2Loss();

    public DJLDqnPolicy(int stateDim, int numActions) {
        this.stateDim = stateDim;
        this.numActions = numActions;
        this.baseManager = NDManager.newBaseManager(Device.cpu());
        this.online = Model.newInstance("rlbot-dqn", Device.cpu());
        this.target = Model.newInstance("rlbot-dqn-target", Device.cpu());
        Block net = new SequentialBlock()
            .add(Linear.builder().setUnits(64).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(64).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(numActions).build());
        online.setBlock(net);
        // target starts with same architecture
        Block net2 = new SequentialBlock()
            .add(Linear.builder().setUnits(64).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(64).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(numActions).build());
        target.setBlock(net2);

        DefaultTrainingConfig cfg = new DefaultTrainingConfig(loss)
            .optOptimizer(Optimizer.adam().build());
        this.trainer = online.newTrainer(cfg);
        this.trainer.initialize(new Shape(1, stateDim));
    }

    public float[] predictQ(float[] state) throws Exception {
        try (Predictor<float[], float[]> pred = online.newPredictor(new NoBatchifyTranslator<float[], float[]>() {
            @Override public float[] processOutput(TranslatorContext ctx, NDList list) { return list.head().toFloatArray(); }
            @Override public NDList processInput(TranslatorContext ctx, float[] input) {
                NDArray x = baseManager.create(input).reshape(1, stateDim);
                return new NDList(x);
            }
        })) {
            return pred.predict(state);
        }
    }

    public float updateBatch(float[][] states, int[] actions, float[] rewards, float[][] nextStates, boolean[] done, float gamma) {
        NDManager m = trainer.getManager();
        NDArray s = m.create(states);
        NDArray ns = m.create(nextStates);
        // forward online
        NDArray qPred = trainer.forward(new NDList(s)).singletonOrThrow(); // [B, A]
        // forward target (no grad)
        ParameterStore ps = new ParameterStore(m, false);
        NDArray qNext = target.getBlock().forward(ps, new NDList(ns), false).singletonOrThrow();
        NDArray maxNext = qNext.max(new int[]{1}, true); // [B,1]
        float[] doneMask = new float[done.length];
        for (int i = 0; i < done.length; i++) doneMask[i] = done[i] ? 0f : 1f;
        NDArray notDone = m.create(doneMask).reshape(-1, 1); // 1 if not done
        NDArray y = m.create(rewards).reshape(-1, 1).add(maxNext.mul(gamma).mul(notDone)); // [B,1]
        // build action mask
        float[][] maskArr = new float[actions.length][numActions];
        for (int i = 0; i < actions.length; i++) maskArr[i][actions[i]] = 1f;
        NDArray mask = m.create(maskArr); // [B,A]
        NDArray oneMinusMask = mask.neg().add(1f);
        NDArray yExpanded = y.repeat(1, numActions); // [B,A]
        NDArray targetQ = qPred.mul(oneMinusMask).add(yExpanded.mul(mask));
        NDArray l = loss.evaluate(new NDList(targetQ), new NDList(qPred));
        try (GradientCollector gc = ai.djl.engine.Engine.getInstance().newGradientCollector()) {
            gc.backward(l);
        }
        trainer.step();
        NDArray lm = l.mean();
        float lossVal = lm.toFloatArray()[0];
        lm.close();
        return lossVal;
    }

    public void syncTarget() {
        // Copy parameters by saving online and loading into target to avoid sharing references
        try {
            java.nio.file.Path tmp = java.nio.file.Files.createTempDirectory("rlbot-dqn-sync");
            try {
                online.save(tmp, "rlbot-dqn-sync");
                target.load(tmp, "rlbot-dqn-sync");
            } finally {
                try {
                    java.nio.file.Files.walk(tmp)
                        .sorted(java.util.Comparator.reverseOrder())
                        .forEach(p -> { try { java.nio.file.Files.deleteIfExists(p); } catch (Exception ignored) {} });
                } catch (Exception ignored) {}
            }
        } catch (Exception ignored) {}
    }

    public void saveIfNeeded(long steps) {
        try {
            if (steps % 5000L == 0 && steps > 0) {
                java.nio.file.Path out = java.nio.file.Paths.get(System.getProperty("user.home"), ".rlbot", "models");
                java.nio.file.Files.createDirectories(out);
                online.save(out, "rlbot-dqn");
            }
        } catch (Exception ignored) {}
    }

    @Override
    public void close() {
        trainer.close();
        online.close();
        target.close();
        baseManager.close();
    }
}


