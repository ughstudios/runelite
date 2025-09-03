package net.runelite.client.plugins.rlbot.rl;

import java.util.ArrayList;
import java.util.List;

public final class ReplayBuffer {
    private final List<Transition> data;
    private final int capacity;
    private int rrIndex = 0; // round-robin index for overwrite/sample

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
        for (int i = 0; i < n; i++) {
            batch.add(data.get((rrIndex + i) % data.size()));
        }
        return batch;
    }
}


