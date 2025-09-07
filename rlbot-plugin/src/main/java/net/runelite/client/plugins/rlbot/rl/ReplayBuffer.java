package net.runelite.client.plugins.rlbot.rl;

import java.util.ArrayList;
import java.util.List;

public final class ReplayBuffer {
    private final List<Transition> data;
    private final int capacity;
    private int rrIndex = 0; // round-robin index for overwrite/sample
    private final java.util.Random random = new java.util.Random();

    public ReplayBuffer(int capacity) {
        this.capacity = Math.max(1, capacity);
        this.data = new ArrayList<>(this.capacity);
        this.rrIndex = 0;
    }

    public void add(Transition t) {
        if (data.size() < capacity) {
            data.add(t);
        } else {
            data.set(rrIndex % capacity, t);
            rrIndex = (rrIndex + 1) % capacity;
        }
    }

    public int size() { return data.size(); }

    public List<Transition> sample(int batchSize) {
        int n = Math.min(batchSize, data.size());
        List<Transition> batch = new ArrayList<>(n);
        if (n <= 0) return batch;
        // Uniform random sample without replacement for better i.i.d. batches
        java.util.HashSet<Integer> picked = new java.util.HashSet<>(n * 2);
        while (batch.size() < n) {
            int idx = random.nextInt(data.size());
            if (picked.add(idx)) {
                batch.add(data.get(idx));
            }
        }
        return batch;
    }
}


