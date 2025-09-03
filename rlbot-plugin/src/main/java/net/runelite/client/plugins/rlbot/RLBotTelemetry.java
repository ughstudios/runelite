package net.runelite.client.plugins.rlbot;

import javax.inject.Singleton;

@Singleton
public class RLBotTelemetry {
    private volatile String mode = "Idle";
    private volatile String targetName = "";
    private volatile int distanceTiles = -1;
    private volatile long busyRemainingMs = 0L;

    public String getMode() { return mode; }
    public void setMode(String mode) { this.mode = mode != null ? mode : ""; }

    public String getTargetName() { return targetName; }
    public void setTargetName(String targetName) { this.targetName = targetName != null ? targetName : ""; }

    public int getDistanceTiles() { return distanceTiles; }
    public void setDistanceTiles(int distanceTiles) { this.distanceTiles = distanceTiles; }

    public long getBusyRemainingMs() { return busyRemainingMs; }
    public void setBusyRemainingMs(long busyRemainingMs) { this.busyRemainingMs = Math.max(0L, busyRemainingMs); }

    // Simple RL episode counters
    private volatile long episodeSteps = 0L;
    private volatile long episodeReward = 0L;
    public long getEpisodeSteps() { return episodeSteps; }
    public long getEpisodeReward() { return episodeReward; }
    public void incEpisodeSteps() { episodeSteps++; }
    public void addReward(long r) { episodeReward += r; }
    public void resetEpisode() { episodeSteps = 0L; episodeReward = 0L; }
}


