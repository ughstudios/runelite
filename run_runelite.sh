#!/bin/bash

# Ensure JDK 17 is used
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"

# Resolve real script directory even when invoked via a symlink (e.g., from /Applications)
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
while [ -h "$SCRIPT_SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
  TARGET="$(readlink "$SCRIPT_SOURCE")"
  case "$TARGET" in
    /*) SCRIPT_SOURCE="$TARGET" ;;
    *)  SCRIPT_SOURCE="$DIR/$TARGET" ;;
  esac
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
# Use the user's default RuneLite home so sideloaded plugins live under ~/.runelite
RUNELITE_HOME="$HOME/.runelite"
SIDELOAD_DIR="$RUNELITE_HOME/sideloaded-plugins"
mkdir -p "$SIDELOAD_DIR"

cleanup_rlbot_sideloads() {
  local target_dir="$1"
  if [ -d "$target_dir" ]; then
    find "$target_dir" -maxdepth 1 -type f \
      \( -name 'rlbot-plugin-*.jar' -o -name 'rlbot-macro-plugin-*.jar' \) \
      -print -delete 2>/dev/null || true
  fi
}

cleanup_rlbot_sideloads "$SIDELOAD_DIR"
cleanup_rlbot_sideloads "$SCRIPT_DIR/runelite/sideloaded-plugins"

# Copy latest rlbot-plugin jar into sideload dir
PLUGIN_JAR=$(ls -t "$SCRIPT_DIR/rlbot-plugin/target"/rlbot-plugin-*-shaded.jar 2>/dev/null | head -n1 || true)
if [ -z "$PLUGIN_JAR" ]; then
  PLUGIN_JAR=$(ls -t "$SCRIPT_DIR/rlbot-plugin/target"/rlbot-plugin-*.jar 2>/dev/null | head -n1 || true)
fi
if [ -n "$PLUGIN_JAR" ]; then
  echo "Copying RLBot plugin jar to $SIDELOAD_DIR"
  cp -f "$PLUGIN_JAR" "$SIDELOAD_DIR/"
fi

# Locate the shaded client jar (prefer upstream under runelite/, fallback to local module)
JAR=""
SEARCH_DIRS=(
  "$SCRIPT_DIR/runelite/runelite-client/target"
  "$SCRIPT_DIR/runelite-client/target"
)
for DIR in "${SEARCH_DIRS[@]}"; do
  if [ -d "$DIR" ]; then
    CANDIDATE=$(ls -t "$DIR"/client-*-shaded.jar 2>/dev/null | head -n1 || true)
    if [ -n "$CANDIDATE" ]; then
      JAR="$CANDIDATE"
      break
    fi
  fi
done

if [ -z "$JAR" ] || [ ! -f "$JAR" ]; then
  echo "Could not find RuneLite shaded client jar."
  echo "Build one of the following:"
  echo "  1) Upstream (preferred path): place client-*-shaded.jar under runelite/runelite-client/target/"
  echo "  2) Local module: mvn -DskipTests -pl runelite-client -Pshade clean package (from repo root)"
  exit 1
fi

# Optionally strip the built-in RLBot from the client jar so only the sideloaded RLBot (Dev) exists
if command -v unzip >/dev/null 2>&1 && command -v zip >/dev/null 2>&1; then
  if unzip -l "$JAR" | grep -q "net/runelite/client/plugins/rlbot/RLBotPlugin.class"; then
    echo "Stripping built-in RLBot from client jar (keeping sideloaded RLBot (Dev))..."
    TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t rlstrip)"
    cp -f "$JAR" "$JAR.backup"
    unzip -q "$JAR" -d "$TMPDIR"
    rm -rf "$TMPDIR/net/runelite/client/plugins/rlbot"
    (cd "$TMPDIR" && zip -qr "$JAR" .)
    rm -rf "$TMPDIR"
  fi
fi

# macOS fullscreen adapter requires apple eawt package access on newer JDKs
exec java \
  -Drlbot.gymIpcDir="$SCRIPT_DIR/rlbot-ipc" \
  --add-exports java.desktop/com.apple.eawt=ALL-UNNAMED \
  --add-exports java.desktop/com.apple.eawt.event=ALL-UNNAMED \
  --add-opens java.desktop/com.apple.eawt=ALL-UNNAMED \
  --add-opens java.desktop/com.apple.eawt.event=ALL-UNNAMED \
  -ea \
  -jar "$JAR" --developer-mode --debug
