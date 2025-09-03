#!/bin/bash
# Hot reload script for RLBot plugin

echo "🔥 Hot reloading RLBot plugin..."

# Build the plugin JAR
./rlbot-plugin/build-plugin.sh

if [ $? -eq 0 ]; then
    echo "✅ Plugin built successfully!"
    echo "📦 Plugin JAR: rlbot-plugin/target/rlbot-plugin-1.0.0.jar"
    echo "📁 Sideloaded to: runelite/sideloaded-plugins/"
    echo ""
    echo "💡 To load the plugin:"
    echo "   1. Restart RuneLite, OR"
    echo "   2. Go to RuneLite Settings > External Plugins and enable RLBot"
    echo ""
    echo "⚡ Hot reload complete! Changes are ready to test."
else
    echo "❌ Plugin build failed!"
    exit 1
fi
