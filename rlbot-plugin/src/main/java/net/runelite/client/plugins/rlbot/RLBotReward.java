package net.runelite.client.plugins.rlbot;

import static net.runelite.client.plugins.rlbot.RLBotConstants.*;

import net.runelite.client.plugins.rlbot.RLBotState.Snapshot;

/**
 * Reward shaping for the RL agent.
 */
final class RLBotReward
{
    private RLBotReward() {}

    private static final float STEP_PENALTY = 0.01f;

    static float compute(Snapshot prev, Snapshot cur, Integer prevWoodcutXp)
    {
        float reward = -STEP_PENALTY;
        if (prev == null || cur == null)
        {
            return reward;
        }

        int deltaFree = prev.freeSlots - cur.freeSlots;
        if (deltaFree > 0)
        {
            reward += deltaFree * 1.0f; // gained logs
        }
        if (cur.bankOpen && cur.freeSlots > prev.freeSlots)
        {
            reward += Math.min(4.0f, (cur.freeSlots - prev.freeSlots) * 1.5f);
        }
        if (cur.bankOpen && cur.inventoryFull)
        {
            reward += 0.5f;
        }
        if (!cur.bankOpen && cur.needsBank())
        {
            reward -= 0.2f;
        }

        if (cur.woodcutting)
        {
            reward += 0.2f;
            if (!prev.woodcutting)
            {
                reward += 1.5f;
            }
        }

        if (prevWoodcutXp != null && cur.woodcutXp > prevWoodcutXp)
        {
            reward += (cur.woodcutXp - prevWoodcutXp) * 0.05f;
        }

        if (cur.needsBank() && cur.bankDistance != null && prev.bankDistance != null)
        {
            if (cur.bankDistance < prev.bankDistance)
            {
                reward += 0.3f;
            }
            else if (cur.bankDistance > prev.bankDistance)
            {
                reward -= 0.2f;
            }
        }

        if (!cur.needsBank() && cur.treeDistance != null && prev.treeDistance != null)
        {
            if (cur.treeDistance < prev.treeDistance)
            {
                reward += 0.3f;
            }
            else if (cur.treeDistance > prev.treeDistance)
            {
                reward -= 0.2f;
            }
        }

        return reward;
    }
}

