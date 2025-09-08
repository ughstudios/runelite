package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.GameObject;

/**
 * Simplified generic object click helper. Not currently used by tasks, but
 * kept for compatibility with earlier versions and potential reuse.
 */
public final class ObjectClicker {
    private ObjectClicker() {}

    public static class ObjectType {
        public final String[] namePatterns;
        public final String[] preferredActions;
        public final boolean useConvexHull;
        public final int clickOffsetY;

        public ObjectType(String[] namePatterns, String[] preferredActions, boolean useConvexHull, int clickOffsetY) {
            this.namePatterns = namePatterns;
            this.preferredActions = preferredActions;
            this.useConvexHull = useConvexHull;
            this.clickOffsetY = clickOffsetY;
        }
    }

    public static final ObjectType TREE = new ObjectType(
        new String[]{"tree", "oak", "willow", "yew", "maple"},
        new String[]{"Chop down", "Chop", "Cut", "Cut down"},
        true,
        0
    );

    public static final ObjectType BANK = new ObjectType(
        new String[]{"bank booth", "bank chest", "bank", "deposit box", "bank deposit box"},
        new String[]{"Bank", "Deposit", "Use", "Open"},
        true,
        -14
    );

    public static boolean clickObject(TaskContext context, GameObject go, ObjectType type) {
        if (context == null || go == null) return false;
        try {
            // Project to a clickable point
            java.awt.Point p;
            if (type != null && type.useConvexHull) {
                p = ObjectFinder.projectToClickablePoint(context, go);
            } else {
                p = ObjectFinder.projectToCanvas(context, go);
            }
            if (p == null) return false;
            if (type != null && type.clickOffsetY != 0) {
                p = new java.awt.Point(p.x, Math.max(0, p.y + type.clickOffsetY));
            }

            // Choose label to validate against (first preferred available)
            String actionLabel = "Use";
            if (type != null && type.preferredActions != null) {
                try {
                    net.runelite.api.ObjectComposition comp = context.client.getObjectDefinition(go.getId());
                    String[] acts = comp != null ? comp.getActions() : null;
                    if (acts != null) {
                        for (String pref : type.preferredActions) {
                            for (String a : acts) {
                                if (a != null && a.equalsIgnoreCase(pref)) { actionLabel = a; throw new RuntimeException("found"); }
                            }
                        }
                    }
                } catch (RuntimeException marker) { /* found */ } catch (Exception ignored) {}
            }

            final java.awt.Point fp = p;
            final String fl = actionLabel;
            final java.util.concurrent.atomic.AtomicBoolean ok = new java.util.concurrent.atomic.AtomicBoolean(false);
            final java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
            context.clientThread.invoke(() -> {
                try {
                    ok.set(context.input.moveAndClickWithValidation(fp, fl));
                } catch (Exception ignored) {
                    ok.set(false);
                } finally { latch.countDown(); }
            });
            try { latch.await(600, java.util.concurrent.TimeUnit.MILLISECONDS); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            if (ok.get()) {
                context.setBusyForMs(250);
            }
            return ok.get();
        } catch (Exception e) {
            return false;
        }
    }
}

