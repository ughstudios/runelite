# RLBot Branch Guide 🌿

## 📋 **Available Branches**

### 🟢 **`master` Branch (Current)**
- **Status**: ✅ Working original system
- **Description**: Your original RLBot agent with manual code and basic RL
- **Use Case**: Stable, reliable bot that works out of the box

### 🚀 **`enhanced-rl-system` Branch**
- **Status**: ⚠️ Enhanced but has DJL compatibility issues
- **Description**: Neural network-driven system with curriculum learning
- **Use Case**: Advanced RL features, reduced manual dependencies

## 🔄 **How to Switch Branches**

### **Switch to Enhanced RL System:**
```bash
git checkout enhanced-rl-system
cd rlbot-plugin && mvn clean package
```

### **Switch Back to Original System:**
```bash
git checkout master
cd rlbot-plugin && mvn clean package
```

## 🎯 **Quick Start (Original System)**

Since you're currently on `master`, you can run your original working system:

```bash
cd rlbot-plugin
mvn clean package
cp target/rlbot-plugin-1.0.0.jar ~/.runelite/plugins/
```

Then in RuneLite:
1. Enable "RLBot" plugin
2. Configure settings:
   - ✅ **"Enable RL Agent"**
   - ✅ **"Show Overlay"**
   - Set **"Agent Interval (ms)"** to 250-500
   - Set **"RL Epsilon"** for exploration (0-100%)

## 🔧 **Configuration You'll Actually See**

In the original system, you'll see these settings in RuneLite:

**Basic Settings:**
- **"Enable RL Agent"** - Turn on/off the RL system
- **"Show Overlay"** - Show the status overlay
- **"Agent Interval (ms)"** - How often the agent acts

**RL Parameters:**
- **"RL Epsilon"** - Exploration rate (0-100%)
- **"RL Alpha"** - Learning rate (1-100%)
- **"RL Gamma"** - Discount factor (0-100%)
- **"RL Batch Size"** - Training batch size (8-256)
- **"RL Replay Capacity"** - Memory buffer size (1000-200000)

**Advanced:**
- **"Debug Logging"** - Enable detailed logs
- **"Performance Logging"** - Log timing metrics
- **"Save Screenshots"** - Save game screenshots

## 🚨 **Current Issue with Enhanced Branch**

The enhanced RL system has a DJL/PyTorch compatibility issue in the RuneLite environment:
```
java.lang.NoClassDefFoundError: Could not initialize class ai.djl.engine.Engine
```

This is a known issue with PyTorch native libraries in plugin environments.

## 💡 **Recommendation**

**For now, use the `master` branch** which has your original working system. The enhanced RL features can be developed further in the `enhanced-rl-system` branch once the DJL compatibility issues are resolved.

## 📊 **What Works on Master**

- ✅ Original RLBot agent with manual task selection + RL assistance
- ✅ Basic DQN neural network (using existing DJLDqnPolicy)
- ✅ Experience replay and model training
- ✅ Action masking and preference logic
- ✅ All UI controls and overlays
- ✅ Manual triggers for testing

**Your original system is ready to use right now!** 🎯
