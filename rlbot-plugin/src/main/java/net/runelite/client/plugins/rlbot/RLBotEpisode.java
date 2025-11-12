package net.runelite.client.plugins.rlbot;

import static net.runelite.client.plugins.rlbot.RLBotConstants.*;

import net.runelite.client.plugins.rlbot.RLBotState.Snapshot;

/**
 * Episode termination policy for the RL agent.
 */
final class RLBotEpisode
{
    private RLBotEpisode() {}

    static boolean shouldTerminateEpisode(Snapshot prev, Snapshot cur, long episodeStartMs, long nowMs)
    {
        if (cur == null)
        {
            return false;
        }

        // Time-based cap
        if (nowMs - episodeStartMs >= Math.max(10_000, EPISODE_MAX_MS))
        {
            return true;
        }

        // End on a successful bank deposit (inventory significantly emptied while bank is open)
        if (EPISODE_END_ON_BANK_DEPOSIT && prev != null)
        {
            boolean depositHappened = cur.bankOpen && prev.freeSlots < cur.freeSlots && cur.freeSlots >= 24;
            if (depositHappened)
            {
                return true;
            }
        }
        return false;
    }
}

