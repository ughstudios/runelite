package net.runelite.client.plugins.rlbot.rl;

public final class Transition {
    public final float[] state;
    public final int action;
    public final float reward;
    public final float[] nextState;
    public final boolean done;

    public Transition(float[] state, int action, float reward, float[] nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}


