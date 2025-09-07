#!/bin/bash
# Hot reload script for RLBot plugin

echo "🔥 Hot reloading RLBot plugin..."

# Build the plugin JAR
./rlbot-plugin/build-plugin.sh

if [ $? -eq 0 ]; then
    echo "✅ Plugin built successfully!"
    echo "📦 Plugin JAR: rlbot-plugin/target/rlbot-plugin-1.0.0.jar"
    echo "📁 Sideloaded to: runelite/sideloaded-plugins/ and ~/.runelite/sideloaded-plugins/"
    echo ""
    echo "⚡ If RuneLite is running (developer mode), the plugin will hot-reload automatically."
    echo "   Otherwise start RuneLite and enable RLBot in settings."
else
    echo "❌ Plugin build failed!"
    exit 1
fi
