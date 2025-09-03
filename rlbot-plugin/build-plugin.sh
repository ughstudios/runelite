#!/bin/bash
# Build RLBot plugin JAR from compiled RuneLite classes

echo "🔨 Building RLBot plugin JAR..."

# Ensure RuneLite client is compiled
cd runelite
mvn compile -pl runelite-client -DskipTests -Dmaven.compiler.failOnError=false
cd ..

# Create plugin JAR
cd rlbot-plugin

# Create target directory structure
mkdir -p target/classes/net/runelite/client/plugins/rlbot

# Copy compiled RLBot classes from RuneLite
cp -r ../runelite/runelite-client/target/classes/net/runelite/client/plugins/rlbot/* target/classes/net/runelite/client/plugins/rlbot/

# Copy plugin metadata
cp -r src/main/resources/* target/classes/

# Create JAR
jar cf target/rlbot-plugin-1.0.0.jar -C target/classes .

# Copy to RuneLite sideloaded plugins directory
mkdir -p ../runelite/sideloaded-plugins
cp target/rlbot-plugin-1.0.0.jar ../runelite/sideloaded-plugins/

echo "✅ RLBot plugin JAR created: target/rlbot-plugin-1.0.0.jar"
echo "📦 Plugin copied to: ../runelite/sideloaded-plugins/"
echo "💡 Restart RuneLite to load the plugin"
