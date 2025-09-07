#!/bin/bash
# Build the standalone RLBot plugin JAR using Maven, and copy to RuneLite sideloaded-plugins
set -e

echo "🔨 Building RLBot plugin JAR (standalone)..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Ensure Java 17 (RuneLite requires JDK17 toolchain)
JAVA_17_PATH=$( /usr/libexec/java_home -v 17 2>/dev/null || true )
if [ -z "$JAVA_17_PATH" ]; then
  echo "❌ JDK 17 not found. Install with: brew install --cask temurin17" >&2
  exit 1
fi
export JAVA_HOME="$JAVA_17_PATH"
export PATH="$JAVA_HOME/bin:$PATH"
echo "Using JAVA_HOME=$JAVA_HOME"

# Ensure local RuneLite shaded client JAR is installed to Maven local for compilation
RL_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RL_CLIENT_SHADED="$RL_ROOT_DIR/runelite/runelite-client/target/client-1.11.16-SNAPSHOT-shaded.jar"
RL_API_JAR="$RL_ROOT_DIR/runelite-api/target/runelite-api-1.11.16-SNAPSHOT.jar"
if [ -f "$RL_CLIENT_SHADED" ]; then
  echo "Installing local RuneLite shaded client to Maven repo: $RL_CLIENT_SHADED"
  mvn -q install:install-file -Dfile="$RL_CLIENT_SHADED" -DgroupId=net.runelite -DartifactId=client -Dversion=1.11.16-SNAPSHOT -Dpackaging=jar -DgeneratePom=true
else
  echo "⚠️  Shaded client jar not found at $RL_CLIENT_SHADED; attempting to build it..."
  (cd "$RL_ROOT_DIR/runelite/runelite-client" && mvn -q -DskipTests -Dmaven.compiler.failOnError=false -Pshade clean package) || true
  if [ -f "$RL_CLIENT_SHADED" ]; then
    mvn -q install:install-file -Dfile="$RL_CLIENT_SHADED" -DgroupId=net.runelite -DartifactId=client -Dversion=1.11.16-SNAPSHOT -Dpackaging=jar -DgeneratePom=true
  else
    echo "❌ Could not locate or build RuneLite shaded client jar; plugin build may fail."
  fi
fi

# Also install runelite-api to local Maven if available (compile-scope provided)
if [ -f "$RL_API_JAR" ]; then
  echo "Installing local RuneLite API to Maven repo: $RL_API_JAR"
  mvn -q install:install-file -Dfile="$RL_API_JAR" -DgroupId=net.runelite -DartifactId=runelite-api -Dversion=1.11.16-SNAPSHOT -Dpackaging=jar -DgeneratePom=true || true
else
  echo "ℹ️  RuneLite API jar not found at $RL_API_JAR; assuming it exists in local Maven or will resolve remotely."
fi

# Build the rlbot-plugin with Maven (ensure correct working directory)
(
  cd "$SCRIPT_DIR"
  mvn -q -DskipTests clean package
)

# Copy to user sideload dir (~/.runelite/sideloaded-plugins)
LOCAL_SIDELOAD_DIR="$HOME/.runelite/sideloaded-plugins"
mkdir -p "$LOCAL_SIDELOAD_DIR"

# Optional backup of previous jar
if [ -f "$LOCAL_SIDELOAD_DIR/rlbot-plugin-1.0.0.jar" ]; then
  cp -f "$LOCAL_SIDELOAD_DIR/rlbot-plugin-1.0.0.jar" "$LOCAL_SIDELOAD_DIR/rlbot-plugin-1.0.0.jar.backup" || true
fi

cp "$SCRIPT_DIR/target/rlbot-plugin-1.0.0.jar" "$LOCAL_SIDELOAD_DIR/"

echo "✅ RLBot plugin JAR created: $SCRIPT_DIR/target/rlbot-plugin-1.0.0.jar"
echo "📁 Copied to: $LOCAL_SIDELOAD_DIR/"

echo "⚡ Auto-reload: If RuneLite is running in developer mode, changes will hot-reload automatically."
