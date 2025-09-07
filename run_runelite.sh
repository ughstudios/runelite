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
JAR="$SCRIPT_DIR/runelite/runelite-client/target/client-1.11.16-SNAPSHOT-shaded.jar"

if [ ! -f "$JAR" ]; then
  echo "Could not find client jar at $JAR"
  exit 1
fi

# macOS fullscreen adapter requires apple eawt package access on newer JDKs
exec java \
  --add-exports java.desktop/com.apple.eawt=ALL-UNNAMED \
  --add-exports java.desktop/com.apple.eawt.event=ALL-UNNAMED \
  --add-opens java.desktop/com.apple.eawt=ALL-UNNAMED \
  --add-opens java.desktop/com.apple.eawt.event=ALL-UNNAMED \
  -ea \
  -jar "$JAR"
