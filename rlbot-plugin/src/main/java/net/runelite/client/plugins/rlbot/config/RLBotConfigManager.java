package net.runelite.client.plugins.rlbot.config;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import net.runelite.api.coords.WorldPoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Consolidated configuration manager for RLBot plugin.
 * Handles all JSON data in a single file to reduce file I/O operations.
 */
public class RLBotConfigManager {
    private static final Logger logger = LoggerFactory.getLogger(RLBotConfigManager.class);
    private static final String CONFIG_FILE = "rlbot-config.json";
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    
    // Configuration data
    private static final Map<WorldPoint, TreeLocation> discoveredTrees = new ConcurrentHashMap<>();
    private static final Map<WorldPoint, Long> depletedTrees = new ConcurrentHashMap<>();
    private static final Map<WorldPoint, BankLocation> discoveredBanks = new ConcurrentHashMap<>();
    private static final Set<WorldPoint> blacklistedBanks = ConcurrentHashMap.newKeySet();
    private static final Set<Integer> bannedObjects = ConcurrentHashMap.newKeySet();
    
    private static long lastSaveTime = 0;
    private static final long SAVE_COOLDOWN_MS = 5000; // Save every 5 seconds max
    
    /**
     * Data class for tree locations
     */
    public static class TreeLocation {
        public int x, y, plane;
        public String name;
        public long discoveredAt;
        
        public TreeLocation(WorldPoint wp, String name) {
            this.x = wp.getX();
            this.y = wp.getY();
            this.plane = wp.getPlane();
            this.name = name;
            this.discoveredAt = System.currentTimeMillis();
        }
        
        public WorldPoint toWorldPoint() {
            return new WorldPoint(x, y, plane);
        }
    }
    
    /**
     * Data class for bank locations
     */
    public static class BankLocation {
        public int x, y, plane;
        public String name;
        public long discoveredAt;
        
        public BankLocation(WorldPoint wp, String name) {
            this.x = wp.getX();
            this.y = wp.getY();
            this.plane = wp.getPlane();
            this.name = name;
            this.discoveredAt = System.currentTimeMillis();
        }
        
        public WorldPoint toWorldPoint() {
            return new WorldPoint(x, y, plane);
        }
    }
    
    /**
     * Main configuration container
     */
    public static class RLBotConfig {
        public List<TreeLocation> trees = new ArrayList<>();
        public List<Map<String, Object>> depletedTrees = new ArrayList<>();
        public List<BankLocation> banks = new ArrayList<>();
        public List<Map<String, Object>> blacklistedBanks = new ArrayList<>();
        public List<Map<String, Object>> bannedObjects = new ArrayList<>();
    }
    
    static {
        loadConfig();
    }
    
    /**
     * Load configuration from JSON file
     */
    private static void loadConfig() {
        try {
            File configFile = new File(CONFIG_FILE);
            if (!configFile.exists()) {
                logger.info("RLBot config file not found, creating new one");
                saveConfig();
                return;
            }
            
            try (FileReader reader = new FileReader(configFile)) {
                RLBotConfig config = gson.fromJson(reader, RLBotConfig.class);
                if (config == null) {
                    config = new RLBotConfig();
                }
                
                // Load trees
                discoveredTrees.clear();
                if (config.trees != null) {
                    for (TreeLocation tree : config.trees) {
                        discoveredTrees.put(tree.toWorldPoint(), tree);
                    }
                }
                
                // Load depleted trees
                depletedTrees.clear();
                if (config.depletedTrees != null) {
                    for (Map<String, Object> depleted : config.depletedTrees) {
                        int x = ((Number) depleted.get("x")).intValue();
                        int y = ((Number) depleted.get("y")).intValue();
                        int plane = ((Number) depleted.get("plane")).intValue();
                        long until = ((Number) depleted.get("until")).longValue();
                        WorldPoint wp = new WorldPoint(x, y, plane);
                        depletedTrees.put(wp, until);
                    }
                }
                
                // Load banks
                discoveredBanks.clear();
                if (config.banks != null) {
                    for (BankLocation bank : config.banks) {
                        discoveredBanks.put(bank.toWorldPoint(), bank);
                    }
                }
                
                // Load blacklisted banks
                blacklistedBanks.clear();
                if (config.blacklistedBanks != null) {
                    for (Map<String, Object> blacklisted : config.blacklistedBanks) {
                        int x = ((Number) blacklisted.get("x")).intValue();
                        int y = ((Number) blacklisted.get("y")).intValue();
                        int plane = ((Number) blacklisted.get("plane")).intValue();
                        WorldPoint wp = new WorldPoint(x, y, plane);
                        blacklistedBanks.add(wp);
                    }
                }
                
                // Load banned objects
                bannedObjects.clear();
                if (config.bannedObjects != null) {
                    for (Map<String, Object> banned : config.bannedObjects) {
                        int id = ((Number) banned.get("id")).intValue();
                        bannedObjects.add(id);
                    }
                }
                
                logger.info("Loaded RLBot config: {} trees, {} depleted trees, {} banks, {} blacklisted banks, {} banned objects", 
                    discoveredTrees.size(), depletedTrees.size(), discoveredBanks.size(), blacklistedBanks.size(), bannedObjects.size());
            }
        } catch (Exception e) {
            logger.error("Error loading RLBot config", e);
        }
    }
    
    /**
     * Save configuration to JSON file
     */
    private static void saveConfig() {
        long now = System.currentTimeMillis();
        if (now - lastSaveTime < SAVE_COOLDOWN_MS) {
            return;
        }
        lastSaveTime = now;
        
        try {
            RLBotConfig config = new RLBotConfig();
            
            // Save trees
            config.trees = new ArrayList<>(discoveredTrees.values());
            
            // Save depleted trees
            config.depletedTrees = new ArrayList<>();
            for (Map.Entry<WorldPoint, Long> entry : depletedTrees.entrySet()) {
                WorldPoint wp = entry.getKey();
                Map<String, Object> depleted = new ConcurrentHashMap<>();
                depleted.put("x", wp.getX());
                depleted.put("y", wp.getY());
                depleted.put("plane", wp.getPlane());
                depleted.put("until", entry.getValue());
                config.depletedTrees.add(depleted);
            }
            
            // Save banks
            config.banks = new ArrayList<>(discoveredBanks.values());
            
            // Save blacklisted banks
            config.blacklistedBanks = new ArrayList<>();
            for (WorldPoint wp : blacklistedBanks) {
                Map<String, Object> blacklisted = new ConcurrentHashMap<>();
                blacklisted.put("x", wp.getX());
                blacklisted.put("y", wp.getY());
                blacklisted.put("plane", wp.getPlane());
                config.blacklistedBanks.add(blacklisted);
            }
            
            // Save banned objects
            config.bannedObjects = new ArrayList<>();
            for (Integer id : bannedObjects) {
                Map<String, Object> banned = new ConcurrentHashMap<>();
                banned.put("id", id);
                config.bannedObjects.add(banned);
            }
            
            try (FileWriter writer = new FileWriter(CONFIG_FILE)) {
                gson.toJson(config, writer);
            }
            
            logger.debug("Saved RLBot config: {} trees, {} depleted trees, {} banks, {} blacklisted banks, {} banned objects", 
                discoveredTrees.size(), depletedTrees.size(), discoveredBanks.size(), blacklistedBanks.size(), bannedObjects.size());
        } catch (Exception e) {
            logger.error("Error saving RLBot config", e);
        }
    }
    
    // Tree management methods
    public static void addTree(WorldPoint wp, String name) {
        discoveredTrees.put(wp, new TreeLocation(wp, name));
        saveConfig();
    }
    
    public static boolean hasTree(WorldPoint wp) {
        return discoveredTrees.containsKey(wp);
    }
    
    public static List<TreeLocation> getTrees() {
        return new ArrayList<>(discoveredTrees.values());
    }
    
    public static void markTreeDepleted(WorldPoint wp, long until) {
        depletedTrees.put(wp, until);
        saveConfig();
    }
    
    public static boolean isTreeDepleted(WorldPoint wp) {
        Long until = depletedTrees.get(wp);
        if (until == null) return false;
        if (System.currentTimeMillis() > until) {
            depletedTrees.remove(wp);
            return false;
        }
        return true;
    }
    
    public static void cleanupExpiredDepletedTrees() {
        long now = System.currentTimeMillis();
        depletedTrees.entrySet().removeIf(entry -> entry.getValue() < now);
    }
    
    // Bank management methods
    public static void addBank(WorldPoint wp, String name) {
        discoveredBanks.put(wp, new BankLocation(wp, name));
        saveConfig();
    }
    
    public static boolean hasBank(WorldPoint wp) {
        return discoveredBanks.containsKey(wp);
    }
    
    public static List<BankLocation> getBanks() {
        return new ArrayList<>(discoveredBanks.values());
    }
    
    public static void blacklistBank(WorldPoint wp) {
        blacklistedBanks.add(wp);
        saveConfig();
    }
    
    public static boolean isBankBlacklisted(WorldPoint wp) {
        return blacklistedBanks.contains(wp);
    }
    
    public static void removeBankFromBlacklist(WorldPoint wp) {
        blacklistedBanks.remove(wp);
        saveConfig();
    }
    
    // Banned object management methods
    public static void banObject(int objectId) {
        bannedObjects.add(objectId);
        saveConfig();
    }
    
    public static boolean isObjectBanned(int objectId) {
        return bannedObjects.contains(objectId);
    }
    
    public static void removeObjectFromBan(int objectId) {
        bannedObjects.remove(objectId);
        saveConfig();
    }
}
