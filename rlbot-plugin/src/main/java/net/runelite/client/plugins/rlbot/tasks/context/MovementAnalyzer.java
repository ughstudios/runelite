package net.runelite.client.plugins.rlbot.tasks.context;

import net.runelite.api.Player;
import net.runelite.client.plugins.rlbot.tasks.TaskContext;

public final class MovementAnalyzer {
    private final BusyTracker busy;

    public MovementAnalyzer(BusyTracker busy) {
        this.busy = busy;
    }

    public boolean isPlayerWalking(TaskContext ctx) {
        return busy.isPlayerWalking(ctx);
    }

    public long getLastMoveMs() {
        return busy.getLastMoveMs();
    }

    public String getWalkingDebugInfo(TaskContext ctx) {
        Player p = ctx.client.getLocalPlayer();
        if (p == null) return "No player";

        try {
            long now = System.currentTimeMillis();
            int anim = p.getAnimation();
            int poseAnim = p.getPoseAnimation();
            int graphic = p.getGraphic();
            boolean isInteracting = p.getInteracting() != null;
            boolean isRunning = ctx.client.getVarpValue(173) == 1;
            String worldLocation = p.getWorldLocation() != null ? p.getWorldLocation().toString() : "null";

            ProcessBuilder pb = new ProcessBuilder(
                "python3",
                "runelite/runelite-client/src/main/java/net/runelite/client/plugins/rlbot/tasks/PlayerWalkingDetector.py",
                "get_walking_debug_info",
                String.valueOf(busy.getLastMoveMs()),
                String.valueOf(anim),
                String.valueOf(poseAnim),
                String.valueOf(graphic),
                String.valueOf(isInteracting),
                String.valueOf(isRunning),
                worldLocation,
                String.valueOf(now)
            );

            Process process = pb.start();
            java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(process.getInputStream())
            );

            String result = reader.readLine();
            process.waitFor();

            if (result != null) {
                return result;
            } else {
                return getWalkingDebugInfoJava(ctx);
            }
        } catch (Exception e) {
            return getWalkingDebugInfoJava(ctx);
        }
    }

    private String getWalkingDebugInfoJava(TaskContext ctx) {
        Player p = ctx.client.getLocalPlayer();
        if (p == null) return "No player";

        long now = System.currentTimeMillis();
        long timeSinceMove = now - busy.getLastMoveMs();
        boolean positionChanged = timeSinceMove < 2000;

        int anim = p.getAnimation();
        int poseAnim = p.getPoseAnimation();
        int graphic = p.getGraphic();
        boolean isInteracting = p.getInteracting() != null;
        boolean isRunning = ctx.client.getVarpValue(173) == 1;

        return String.format("Pos:%s Time:%dms A:%d P:%d G:%d Int:%s Run:%s",
            p.getWorldLocation(),
            timeSinceMove,
            anim,
            poseAnim,
            graphic,
            isInteracting ? "Y" : "N",
            isRunning ? "Y" : "N");
    }
}

