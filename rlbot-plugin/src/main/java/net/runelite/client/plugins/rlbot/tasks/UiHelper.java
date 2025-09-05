package net.runelite.client.plugins.rlbot.tasks;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.List;
import com.google.gson.Gson;
import java.lang.reflect.Type;
import com.google.gson.reflect.TypeToken;
import java.awt.event.KeyEvent;
import net.runelite.api.widgets.Widget;
import net.runelite.api.MenuAction;
import net.runelite.api.widgets.WidgetID;

final class UiHelper {
    private static long lastUiCloseAttemptMs = 0L;
    private static final long UI_CLOSE_COOLDOWN_MS = 1500L;
    private static long lastUiDumpMs = 0L;
    private static final long UI_DUMP_COOLDOWN_MS = 2000L;
    private static long lastBlockerLoadMs = 0L;
    private static final long BLOCKER_LOAD_COOLDOWN_MS = 5000L;
    private static final Set<Integer> configuredWidgetIds = new HashSet<>();
    private static final Set<String> configuredNameSubs = new HashSet<>();
    private static long lastJsonLoadMs = 0L;
    private static final long JSON_LOAD_COOLDOWN_MS = 5000L;
    private static final List<BlockerRule> jsonRules = new java.util.ArrayList<>();
    private UiHelper() {}

    static void closeObstructions(TaskContext ctx) {
        try {
            long now = System.currentTimeMillis();
            if (now - lastUiCloseAttemptMs < UI_CLOSE_COOLDOWN_MS) {
                return; // cooldown to avoid spam
            }

            // First, collapse side panels (inventory) if they are covering the viewport
            if (maybeCollapseInventorySidePanel(ctx)) {
                ctx.logger.info("[UI] Collapsed inventory side panel");
                ctx.setBusyForMs(150);
                lastUiCloseAttemptMs = now;
                return;
            }

            loadJsonBlockerConfig(ctx);
            if (closeConfiguredJsonBlockers(ctx)) {
                ctx.logger.info("[UI] Closed JSON-configured blocker");
                ctx.setBusyForMs(150);
                lastUiCloseAttemptMs = now;
                return;
            }

            loadBlockerConfig(ctx);
            if (closeConfiguredBlockers(ctx)) {
                ctx.logger.info("[UI] Closed configured blocker");
                ctx.setBusyForMs(150);
                lastUiCloseAttemptMs = now;
                return;
            }
            if (clickCloseIfVisible(ctx)) {
                ctx.logger.info("[UI] Clicked a visible Close button");
                ctx.setBusyForMs(150);
                lastUiCloseAttemptMs = now;
                return;
            }
            // No explicit close found: dump current visible widgets for manual labeling
            logVisibleWidgets(ctx);
            // Do NOT send ESC blindly if nothing looks closable; avoid spam when nothing is open
            ctx.logger.debug("[UI] No Close button visible; skipping ESC");
            lastUiCloseAttemptMs = now;
        } catch (Exception ignored) {
        }
    }

    static boolean clickCloseIfVisible(TaskContext ctx) {
        try {
            // Prefer exact group targeting to avoid hitting non-close 'X' widgets (e.g., Quantity X)
            if (clickCloseForGroup(ctx, WidgetID.BANK_GROUP_ID)) return true;
            if (clickCloseForGroup(ctx, WidgetID.DEPOSIT_BOX_GROUP_ID)) return true;

            // Explicit: close bank main by concrete widget id if present (user-provided)
            if (closeBankMainById(ctx, 786434)) return true; // 12<<16 | 2

            // Heuristic search: scan recent top-layer component ids that commonly have Close
            int[] commonGroupIds = new int[] { 162, 593, 219, 164, 216, 193, 15, 12 };
            for (int groupId : commonGroupIds) {
                for (int child = 0; child < 300; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null || w.isHidden()) continue;
                    if (attemptClickIfClose(ctx, w)) return true;
                    if (dfsClickClose(ctx, w, 5000)) return true;
                }
            }
            // Full scan fallback: bounded group/child ranges with DFS
            for (int groupId = 0; groupId <= 800; groupId++) {
                for (int child = 0; child < 500; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null || w.isHidden()) continue;
                    if (attemptClickIfClose(ctx, w)) return true;
                    if (dfsClickClose(ctx, w, 8000)) return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static boolean clickCloseForGroup(TaskContext ctx, int groupId) {
        try {
            for (int child = 0; child < 500; child++) {
                Widget w = ctx.client.getWidget(groupId, child);
                if (w == null || w.isHidden()) continue;
                // Strong filter: action/name contains Close, or small 'X' near top-right of bank area
                if (!isCloseWidget(w)) continue;
                if (attemptClickIfClose(ctx, w)) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    /**
     * Explicitly close the Bank main interface by known root widget id (e.g., 786434).
     * Heuristics: search subtree for a small button near top-right, or a widget with
     * action/name/text indicating Close, then click it and issue CC_OP.
     */
    static boolean closeBankMainById(TaskContext ctx, int rootWidgetId) {
        try {
            Widget root = findWidgetById(ctx, rootWidgetId);
            if (root == null || root.isHidden()) return false;
            Rectangle rb = root.getBounds();
            if (rb == null || rb.width <= 0 || rb.height <= 0) return false;

            // First pass: exact close predicate
            Widget target = findCloseInSubtree(ctx, root);
            if (target == null) {
                // Second pass: heuristic small button near top-right of root
                target = findTopRightSmallButton(root);
            }
            if (target == null) return false;

            Rectangle b = target.getBounds();
            if (b == null || b.width <= 0 || b.height <= 0) return false;
            int wid = target.getId();
            Point center = new Point(b.x + b.width / 2, b.y + b.height / 2);
            ctx.input.smoothMouseMove(center);
            ctx.setBusyForMs(150);
            ctx.input.clickAt(center);
            ctx.setBusyForMs(200);
            int op = 1;
            String[] actions = target.getActions();
            if (actions != null) {
                for (int i = 0; i < actions.length; i++) {
                    String a = actions[i];
                    if (a != null && a.toLowerCase().contains("close")) { op = i + 1; break; }
                }
            }
            final int opFinal = op;
            final int widFinal = wid;
            ctx.clientThread.invoke(() -> {
                try {
                    ctx.client.menuAction(-1, widFinal, MenuAction.CC_OP, opFinal, -1, "Close", "");
                } catch (Exception ignored) {}
            });
            ctx.setBusyForMs(300);
            ctx.logger.info("[UI] Closed bank via explicit widget id=" + rootWidgetId + " targetId=" + wid);
            return true;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static Widget findWidgetById(TaskContext ctx, int id) {
        try {
            for (int groupId = 0; groupId <= 800; groupId++) {
                for (int child = 0; child < 600; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null) continue;
                    if (w.getId() == id) return w;
                }
            }
        } catch (Exception ignored) {}
        return null;
    }

    private static Widget findCloseInSubtree(TaskContext ctx, Widget root) {
        Deque<Widget> stack = new ArrayDeque<>();
        Set<Integer> seen = new HashSet<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Widget w = stack.pop();
            if (w == null) continue;
            if (!seen.add(w.getId())) continue;
            if (!w.isHidden() && isCloseWidget(w)) return w;
            enqueueChildren(stack, w.getChildren());
            enqueueChildren(stack, w.getStaticChildren());
            enqueueChildren(stack, w.getDynamicChildren());
            enqueueChildren(stack, w.getNestedChildren());
        }
        return null;
    }

    private static Widget findTopRightSmallButton(Widget root) {
        Rectangle rb = root.getBounds();
        if (rb == null) return null;
        Deque<Widget> stack = new ArrayDeque<>();
        Set<Integer> seen = new HashSet<>();
        stack.push(root);
        Widget best = null;
        int bestScore = Integer.MAX_VALUE;
        while (!stack.isEmpty()) {
            Widget w = stack.pop();
            if (w == null) continue;
            if (!seen.add(w.getId())) continue;
            Rectangle b = w.getBounds();
            if (b != null && b.width > 0 && b.height > 0) {
                boolean small = b.width <= 30 && b.height <= 30;
                if (small) {
                    int dx = Math.abs((rb.x + rb.width) - (b.x + b.width));
                    int dy = Math.abs((rb.y) - (b.y));
                    int score = dx + dy;
                    if (score < bestScore) { bestScore = score; best = w; }
                }
            }
            enqueueChildren(stack, w.getChildren());
            enqueueChildren(stack, w.getStaticChildren());
            enqueueChildren(stack, w.getDynamicChildren());
            enqueueChildren(stack, w.getNestedChildren());
        }
        return best;
    }

    private static boolean dfsClickClose(TaskContext ctx, Widget root, int maxNodes) {
        try {
            Deque<Widget> stack = new ArrayDeque<>();
            Set<Integer> seen = new HashSet<>();
            stack.push(root);
            int visited = 0;
            while (!stack.isEmpty() && visited < Math.max(100, maxNodes)) {
                Widget w = stack.pop();
                if (w == null) continue;
                int id = w.getId();
                if (!seen.add(id)) continue;
                visited++;
                if (!w.isHidden() && attemptClickIfClose(ctx, w)) return true;
                enqueueChildren(stack, w.getChildren());
                enqueueChildren(stack, w.getStaticChildren());
                enqueueChildren(stack, w.getDynamicChildren());
                enqueueChildren(stack, w.getNestedChildren());
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static void enqueueChildren(Deque<Widget> stack, Widget[] arr) {
        if (arr == null) return;
        for (Widget c : arr) {
            if (c != null) stack.push(c);
        }
    }

    private static boolean attemptClickIfClose(TaskContext ctx, Widget w) {
        if (!isCloseWidget(w)) return false;
        Rectangle b = w.getBounds();
        if (b == null || b.width <= 0 || b.height <= 0) return false;
        int widgetId = w.getId();
        ctx.logger.info("[UI] Closing via menuAction CC_OP on widgetId=" + widgetId);
        
        // First, try to click on the widget to ensure it's properly targeted
        try {
            Point center = new Point(b.x + b.width / 2, b.y + b.height / 2);
            ctx.input.smoothMouseMove(center);
            ctx.setBusyForMs(200); // Wait for mouse movement
            ctx.input.clickAt(center);
            ctx.setBusyForMs(300); // Wait for click to register
        } catch (Exception e) {
            ctx.logger.warn("[UI] Failed to click on close widget: " + e.getMessage());
        }
        
        // Then invoke the menu action; prefer the widget's explicit action index when present
        ctx.clientThread.invoke(() -> {
            try {
                int op = 1; // default CC_OP index
                String[] actions = w.getActions();
                if (actions != null) {
                    for (int i = 0; i < actions.length; i++) {
                        String a = actions[i];
                        if (a != null && a.toLowerCase().contains("close")) { op = i + 1; break; }
                    }
                }
                ctx.client.menuAction(-1, widgetId, MenuAction.CC_OP, op, -1, "Close", "");
            } catch (Exception ignored) {}
        });
        
        // Longer delay to ensure the close action registers
        ctx.setBusyForMs(500);
        return true;
    }

    /**
     * Collapse the inventory side panel if it's visible in Pre-EoC resizable layout and may occlude clicks.
     * Uses known widget ids: ToplevelPreEoc.SIDE_CONTAINER (164,96) and toggles STONE3 tab.
     */
    private static boolean maybeCollapseInventorySidePanel(TaskContext ctx) {
        try {
            // Detect side container visibility and size
            Widget side = ctx.client.getWidget(164, 96); // ToplevelPreEoc.SIDE_CONTAINER
            if (side == null || side.isHidden()) return false;
            Rectangle b = side.getBounds();
            if (b == null || b.width <= 0 || b.height <= 0) return false;
            // If the side container is wide enough, it's likely expanded
            if (b.width < 180) return false; // small width likely collapsed

            // Toggle the inventory tab button to collapse the panel
            Widget invTab = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.RESIZABLE_VIEWPORT_BOTTOM_LINE_INVENTORY_TAB);
            if (invTab == null || invTab.isHidden()) {
                invTab = ctx.client.getWidget(net.runelite.api.widgets.WidgetInfo.FIXED_VIEWPORT_INVENTORY_TAB);
            }
            if (invTab != null && !invTab.isHidden()) {
                final int wid = invTab.getId();
                ctx.clientThread.invoke(() -> {
                    try {
                        ctx.client.menuAction(-1, wid, MenuAction.CC_OP, 1, -1, "Toggle", "");
                    } catch (Exception ignored) {}
                });
                ctx.setBusyForMs(200);
                return true;
            }
        } catch (Exception ignored) {}
        return false;
    }

    private static boolean isCloseWidget(Widget w) {
        try {
            String[] acts = w.getActions();
            if (acts != null) {
                for (String a : acts) {
                    if (a != null) {
                        String al = a.toLowerCase();
                        if (al.contains("close")) return true;
                    }
                }
            }
            String name = w.getName();
            if (name != null && name.toLowerCase().contains("close")) return true;
            String text = w.getText();
            if (text != null && text.trim().equalsIgnoreCase("x")) {
                // Heuristic: only treat bare "X" as a close button if it's a small button
                // near the top of the interface (avoid Quantity "X" near bank bottom).
                java.awt.Rectangle b = w.getBounds();
                if (b != null) {
                    boolean smallButton = b.width <= 30 && b.height <= 30;
                    boolean nearTop = b.y <= 150; // top strip of the interface
                    if (smallButton && nearTop) return true;
                }
                return false; // otherwise, do not consider it a close button
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    // ===== JSON blocker config =====

    static final class RulesFile { List<BlockerRule> rules; }
    static final class BlockerRule { MatchSpec match; MatchSpec close; }
    static final class MatchSpec { Integer id; Integer group; Integer child; String nameContains; Boolean esc; }

    private static void loadJsonBlockerConfig(TaskContext ctx) {
        long now = System.currentTimeMillis();
        if (now - lastJsonLoadMs < JSON_LOAD_COOLDOWN_MS) return;
        lastJsonLoadMs = now;
        try {
            Path explicit = null;
            String prop = System.getProperty("rlbot.ui.blockers.path");
            if (prop != null && !prop.trim().isEmpty()) {
                explicit = Paths.get(prop.trim());
            }
            Path cwd = Paths.get("runelite", "rlbot", "config", "rlbot-ui-blockers.json");
            Path home = null; // Don't use home directory
            Path[] candidates = new Path[] { explicit, cwd, home };
            for (Path p : candidates) {
                if (p == null) continue;
                if (Files.exists(p)) {
                    String json = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
                    parseJsonRules(json);
                    ctx.logger.debug("[UI] Loaded JSON blocker config from " + p);
                    return;
                }
            }
            // If no file found, clear rules
            jsonRules.clear();
        } catch (Exception ignored) {
        }
    }

    private static void parseJsonRules(String json) {
        try {
            Gson gson = new Gson();
            // Accept either {"rules":[...]} or bare array [...]
            if (json.trim().startsWith("{")) {
                Type t = new TypeToken<RulesFile>(){}.getType();
                RulesFile rf = gson.fromJson(json, t);
                jsonRules.clear();
                if (rf != null && rf.rules != null) jsonRules.addAll(rf.rules);
            } else {
                Type t = new TypeToken<List<BlockerRule>>(){}.getType();
                List<BlockerRule> rs = gson.fromJson(json, t);
                jsonRules.clear();
                if (rs != null) jsonRules.addAll(rs);
            }
        } catch (Exception ignored) {
            jsonRules.clear();
        }
    }

    private static boolean closeConfiguredJsonBlockers(TaskContext ctx) {
        if (jsonRules.isEmpty()) return false;
        try {
            for (BlockerRule rule : jsonRules) {
                if (rule == null || rule.match == null) continue;
                Widget root = findFirstMatch(ctx, rule.match);
                if (root == null) continue;
                // If close spec provided
                if (rule.close != null) {
                    if (Boolean.TRUE.equals(rule.close.esc)) {
                        ctx.input.pressKey(KeyEvent.VK_ESCAPE);
                        return true;
                    }
                    Widget target = findInSubtreeBySpec(ctx, root, rule.close);
                    if (target == null) target = findGlobalBySpec(ctx, rule.close);
                    if (target != null) {
                        int widgetId = target.getId();
                        ctx.logger.info("[UI] Closing JSON-matched widget via CC_OP id=" + widgetId);
                        int wid = widgetId;
                        ctx.clientThread.invoke(() -> {
                            try {
                                ctx.client.menuAction(-1, wid, net.runelite.api.MenuAction.CC_OP, 1, -1, "Close", "");
                            } catch (Exception ignored) {}
                        });
                        ctx.setBusyForMs(150);
                        return true;
                    }
                }
                // Fallback: attempt default close on root
                if (attemptClickIfClose(ctx, root)) return true;
                if (dfsClickClose(ctx, root, 4000)) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static Widget findFirstMatch(TaskContext ctx, MatchSpec spec) {
        try {
            for (int groupId = 0; groupId <= 800; groupId++) {
                for (int child = 0; child < 500; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null || w.isHidden()) continue;
                    if (matchesSpec(w, spec)) return w;
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private static boolean matchesSpec(Widget w, MatchSpec spec) {
        try {
            if (spec.id != null && w.getId() == spec.id) return true;
            if (spec.group != null && spec.child != null) {
                int id = (spec.group << 16) | (spec.child & 0xFFFF);
                if (w.getId() == id) return true;
            }
            if (spec.nameContains != null && !spec.nameContains.trim().isEmpty()) {
                String sub = spec.nameContains.toLowerCase();
                String name = safe(w.getName()).toLowerCase();
                String text = safe(w.getText()).toLowerCase();
                if (name.contains(sub) || text.contains(sub)) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static Widget findInSubtreeBySpec(TaskContext ctx, Widget root, MatchSpec spec) {
        try {
            Deque<Widget> stack = new ArrayDeque<>();
            Set<Integer> seen = new HashSet<>();
            stack.push(root);
            while (!stack.isEmpty()) {
                Widget w = stack.pop();
                if (w == null) continue;
                int id = w.getId();
                if (!seen.add(id)) continue;
                if (!w.isHidden() && matchesSpec(w, spec)) return w;
                enqueueChildren(stack, w.getChildren());
                enqueueChildren(stack, w.getStaticChildren());
                enqueueChildren(stack, w.getDynamicChildren());
                enqueueChildren(stack, w.getNestedChildren());
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private static Widget findGlobalBySpec(TaskContext ctx, MatchSpec spec) {
        try {
            for (int groupId = 0; groupId <= 800; groupId++) {
                for (int child = 0; child < 500; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null || w.isHidden()) continue;
                    if (matchesSpec(w, spec)) return w;
                }
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private static boolean closeConfiguredBlockers(TaskContext ctx) {
        try {
            // If nothing configured, skip fast
            if (configuredWidgetIds.isEmpty() && configuredNameSubs.isEmpty()) return false;
            boolean any = false;
            for (int groupId = 0; groupId <= 800; groupId++) {
                for (int child = 0; child < 500; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null || w.isHidden()) continue;
                    if (!isConfiguredBlocker(w)) continue;
                    any = true;
                    // Try to find a close within this widget's subtree first
                    if (attemptClickIfClose(ctx, w)) return true;
                    if (dfsClickClose(ctx, w, 4000)) return true;
                }
            }
            return false;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean isConfiguredBlocker(Widget w) {
        try {
            int id = w.getId();
            if (configuredWidgetIds.contains(id)) return true;
            String name = w.getName();
            String text = w.getText();
            String hay = ((name == null ? "" : name) + "\n" + (text == null ? "" : text)).toLowerCase();
            for (String sub : configuredNameSubs) {
                if (sub.isEmpty()) continue;
                if (hay.contains(sub)) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private static void loadBlockerConfig(TaskContext ctx) {
        long now = System.currentTimeMillis();
        if (now - lastBlockerLoadMs < BLOCKER_LOAD_COOLDOWN_MS) return;
        lastBlockerLoadMs = now;
        try {
            // Load from file in ~/.runelite/rlbot-ui-blockers.txt if present
            Path p = Paths.get(System.getProperty("user.home"), ".runelite", "rlbot-ui-blockers.txt");
            if (Files.exists(p)) {
                List<String> lines = Files.readAllLines(p, StandardCharsets.UTF_8);
                parseBlockerLines(lines);
                ctx.logger.debug("[UI] Loaded blocker config from " + p);
                return;
            }
        } catch (Exception ignored) {
        }
        // No file found; keep whatever is already configured (possibly empty)
    }

    private static void parseBlockerLines(List<String> lines) {
        configuredWidgetIds.clear();
        configuredNameSubs.clear();
        if (lines == null) return;
        for (String raw : lines) {
            if (raw == null) continue;
            String s = raw.trim();
            if (s.isEmpty() || s.startsWith("#")) continue;
            String lower = s.toLowerCase();
            try {
                if (lower.startsWith("name:")) {
                    String sub = lower.substring(5).trim();
                    if (!sub.isEmpty()) configuredNameSubs.add(sub);
                } else if (lower.startsWith("id:")) {
                    String rest = lower.substring(3).trim();
                    Integer id = parseWidgetId(rest);
                    if (id != null) configuredWidgetIds.add(id);
                } else {
                    // heuristic: allow bare id or gid:child
                    Integer id = parseWidgetId(lower);
                    if (id != null) configuredWidgetIds.add(id);
                    else if (!lower.isEmpty()) configuredNameSubs.add(lower);
                }
            } catch (Exception ignored) {
            }
        }
    }

    private static Integer parseWidgetId(String s) {
        if (s == null) return null;
        s = s.trim();
        if (s.isEmpty()) return null;
        if (s.contains(":")) {
            String[] parts = s.split(":", 2);
            try {
                int gid = Integer.parseInt(parts[0].trim());
                int child = Integer.parseInt(parts[1].trim());
                return (gid << 16) | child;
            } catch (NumberFormatException e) {
                return null;
            }
        }
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    static void logVisibleWidgets(TaskContext ctx) {
        try {
            long now = System.currentTimeMillis();
            if (now - lastUiDumpMs < UI_DUMP_COOLDOWN_MS) return;
            lastUiDumpMs = now;
            int logged = 0;
            // Quiet console: write widget dump to file only
            ctx.logger.file("[UI] Dumping visible widgets (for manual blocker labeling)");
            for (int groupId = 0; groupId <= 800; groupId++) {
                for (int child = 0; child < 500; child++) {
                    Widget w = ctx.client.getWidget(groupId, child);
                    if (w == null || w.isHidden()) continue;
                    Rectangle b = w.getBounds();
                    if (b == null || b.width <= 0 || b.height <= 0) continue;
                    String name = safe(w.getName());
                    String text = safe(w.getText());
                    String acts = "";
                    String[] arr = w.getActions();
                    if (arr != null) {
                        acts = Arrays.stream(arr).filter(Objects::nonNull).limit(6).collect(Collectors.joining("|"));
                    }
                    ctx.logger.file(String.format(
                        "[UI-WIDGET] gid=%d child=%d id=%d hidden=%s bounds=(%d,%d %dx%d) name=\"%s\" text=\"%s\" actions=[%s]",
                        groupId, child, w.getId(), w.isHidden(), b.x, b.y, b.width, b.height, name, text, acts
                    ));
                    if (++logged >= 4000) {
                        ctx.logger.file("[UI] Reached widget log cap (4000); stopping dump");
                        return;
                    }
                }
            }
            if (logged == 0) {
                ctx.logger.file("[UI] No visible widgets to dump");
            } else {
                ctx.logger.file("[UI] Logged " + logged + " widgets");
            }
        } catch (Exception ignored) {
        }
    }

    private static String safe(String s) {
        if (s == null) return "";
        String t = s.replace('\n', ' ').replace('\r', ' ').trim();
        if (t.length() > 120) t = t.substring(0, 120) + "…";
        return t;
    }
}


