package net.runelite.client.plugins.rlbot.rewards;

import net.runelite.api.Client;
import net.runelite.api.Item;
import net.runelite.api.ItemContainer;
import net.runelite.api.InventoryID;
import net.runelite.api.ItemComposition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility class for calculating rewards based on log quality and experience values.
 * Higher quality logs that give more experience should provide higher rewards.
 */
public class LogQualityRewards {
    private static final Logger logger = LoggerFactory.getLogger(LogQualityRewards.class);
    
    // Experience values for different log types (from OSRS data)
    private static final Map<String, Double> LOG_EXPERIENCE_VALUES = new HashMap<>();
    private static final Map<String, Integer> LOG_QUALITY_TIERS = new HashMap<>();
    
    static {
        // Initialize experience values and quality tiers
        LOG_EXPERIENCE_VALUES.put("logs", 25.0);
        LOG_EXPERIENCE_VALUES.put("oak logs", 37.5);
        LOG_EXPERIENCE_VALUES.put("willow logs", 67.5);
        LOG_EXPERIENCE_VALUES.put("teak logs", 85.0);
        LOG_EXPERIENCE_VALUES.put("maple logs", 100.0);
        LOG_EXPERIENCE_VALUES.put("mahogany logs", 125.5);
        LOG_EXPERIENCE_VALUES.put("yew logs", 175.0);
        LOG_EXPERIENCE_VALUES.put("magic logs", 250.0);
        LOG_EXPERIENCE_VALUES.put("redwood logs", 380.0);
        
        // Quality tiers (higher number = better quality)
        LOG_QUALITY_TIERS.put("logs", 1);
        LOG_QUALITY_TIERS.put("oak logs", 2);
        LOG_QUALITY_TIERS.put("willow logs", 3);
        LOG_QUALITY_TIERS.put("teak logs", 4);
        LOG_QUALITY_TIERS.put("maple logs", 5);
        LOG_QUALITY_TIERS.put("mahogany logs", 6);
        LOG_QUALITY_TIERS.put("yew logs", 7);
        LOG_QUALITY_TIERS.put("magic logs", 8);
        LOG_QUALITY_TIERS.put("redwood logs", 9);
    }
    
    /**
     * Calculate bonus reward multiplier based on log quality.
     * Higher quality logs get progressively higher multipliers.
     * 
     * @param logName The name of the log type
     * @return Bonus multiplier (1.0 = no bonus, higher values = better bonus)
     */
    public static double getLogQualityMultiplier(String logName) {
        if (logName == null) return 1.0;
        
        String lowerName = logName.toLowerCase();
        Integer tier = LOG_QUALITY_TIERS.get(lowerName);
        
        if (tier == null) {
            // Check for partial matches (e.g., "oak" in "oak logs")
            for (Map.Entry<String, Integer> entry : LOG_QUALITY_TIERS.entrySet()) {
                if (lowerName.contains(entry.getKey().split(" ")[0])) {
                    tier = entry.getValue();
                    break;
                }
            }
        }
        
        if (tier == null) return 1.0;
        
        // Progressive multiplier: tier 1 = 1.0x, tier 2 = 1.3x, tier 3 = 1.7x, etc.
        return 1.0 + (tier - 1) * 0.4;
    }
    
    /**
     * Get the experience value for a specific log type.
     * 
     * @param logName The name of the log type
     * @return Experience value, or 0 if unknown
     */
    public static double getLogExperienceValue(String logName) {
        if (logName == null) return 0.0;
        
        String lowerName = logName.toLowerCase();
        Double exp = LOG_EXPERIENCE_VALUES.get(lowerName);
        
        if (exp == null) {
            // Check for partial matches
            for (Map.Entry<String, Double> entry : LOG_EXPERIENCE_VALUES.entrySet()) {
                if (lowerName.contains(entry.getKey().split(" ")[0])) {
                    exp = entry.getValue();
                    break;
                }
            }
        }
        
        return exp != null ? exp : 0.0;
    }
    
    /**
     * Calculate quality-based reward for gaining logs in inventory.
     * Analyzes current inventory contents to determine what types of logs were gained.
     * 
     * @param client The RuneScape client
     * @param logCount Number of logs gained
     * @return Quality-adjusted reward value
     */
    public static float calculateLogQualityReward(Client client, int logCount) {
        if (logCount <= 0) return 0.0f;
        
        try {
            ItemContainer inv = client.getItemContainer(InventoryID.INVENTORY);
            if (inv == null) return (float) logCount * 1.0f; // Default reward if no inventory access
            
            Item[] items = inv.getItems();
            if (items == null) return (float) logCount * 1.0f;
            
            // Find the highest quality log in inventory to determine reward
            double bestMultiplier = 1.0;
            String bestLogType = "unknown";
            
            for (Item item : items) {
                if (item == null || item.getId() <= 0) continue;
                
                try {
                    ItemComposition itemComp = client.getItemDefinition(item.getId());
                    if (itemComp == null) continue;
                    
                    String itemName = itemComp.getName();
                    if (itemName == null) continue;
                    
                    // Check if this is a log item
                    String lowerName = itemName.toLowerCase();
                    if (lowerName.contains("logs") || lowerName.equals("logs")) {
                        double multiplier = getLogQualityMultiplier(itemName);
                        if (multiplier > bestMultiplier) {
                            bestMultiplier = multiplier;
                            bestLogType = itemName;
                        }
                    }
                } catch (Exception ignored) {}
            }
            
            float baseReward = (float) logCount * 2.0f; // Base reward per log
            float qualityReward = (float) (baseReward * bestMultiplier);
            
            if (bestMultiplier > 1.0) {
                logger.info("[LogQuality] Enhanced reward for " + bestLogType + ": " + 
                           String.format("%.2f", baseReward) + " -> " + String.format("%.2f", qualityReward) + 
                           " (multiplier: " + String.format("%.1f", bestMultiplier) + "x)");
            }
            
            return qualityReward;
            
        } catch (Exception e) {
            logger.warn("[LogQuality] Error calculating log quality reward: " + e.getMessage());
            return (float) logCount * 1.0f; // Fallback to basic reward
        }
    }
    
    /**
     * Calculate experience-based reward bonus.
     * Rewards the agent more for actions that lead to higher XP gains.
     * 
     * @param xpGained Amount of woodcutting XP gained
     * @return Bonus reward based on XP value
     */
    public static float calculateExperienceReward(int xpGained) {
        if (xpGained <= 0) return 0.0f;
        
        // Scale reward based on XP gained
        // Higher XP gains (from better logs) get exponentially better rewards
        float baseXpReward = xpGained * 0.1f;
        
        // Bonus multiplier for high-value XP gains
        if (xpGained >= 175) { // Yew or better
            baseXpReward *= 2.0f;
            logger.info("[LogQuality] High-value XP bonus applied for " + xpGained + " XP");
        } else if (xpGained >= 100) { // Maple or better
            baseXpReward *= 1.5f;
            logger.info("[LogQuality] Medium-value XP bonus applied for " + xpGained + " XP");
        } else if (xpGained >= 67) { // Willow or better
            baseXpReward *= 1.2f;
        }
        
        return Math.min(8.0f, baseXpReward); // Cap at 8.0 to prevent excessive rewards
    }
    
    /**
     * Get quality tier of a log type for decision making.
     * 
     * @param logName The name of the log type
     * @return Quality tier (1-9, higher is better), or 0 if unknown
     */
    public static int getLogQualityTier(String logName) {
        if (logName == null) return 0;
        
        String lowerName = logName.toLowerCase();
        Integer tier = LOG_QUALITY_TIERS.get(lowerName);
        
        if (tier == null) {
            // Check for partial matches
            for (Map.Entry<String, Integer> entry : LOG_QUALITY_TIERS.entrySet()) {
                if (lowerName.contains(entry.getKey().split(" ")[0])) {
                    tier = entry.getValue();
                    break;
                }
            }
        }
        
        return tier != null ? tier : 0;
    }
}
