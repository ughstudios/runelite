package net.runelite.client.plugins.rlbot.tasks;

import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;

/**
 * Composite task that handles the full banking flow when inventory is near-full:
 * - If bank is open, deposit inventory.
 * - Otherwise, approach a bank hotspot and stage a bank click (hover -> click across ticks).
 *
 * This reduces action fragmentation for the RL agent by combining navigation and
 * deposit into a single macro.
 */
public class BankWhenFullTask implements Task
{
    private final NavigateToBankHotspotTask navigate = new NavigateToBankHotspotTask();
    private final BankDepositTask deposit = new BankDepositTask();

    private enum Phase { IDLE, NAVIGATE, CLICK_STAGE, OPEN_WAIT, DEPOSIT, CLOSE }
    private Phase phase = Phase.IDLE;
    private long phaseStartMs = 0L;
    private int clickRetries = 0;
    private static final long NAVIGATE_MAX_MS = 12000L;
    private static final long OPEN_WAIT_MAX_MS = 6000L;
    private static final int CLICK_RETRY_MAX = 3;

    public boolean isActive()
    {
        return phase != Phase.IDLE;
    }

    @Override
    public void run(TaskContext context)
    {
        long now = System.currentTimeMillis();

        // If no longer need banking, reset state.
        if (!context.isInventoryNearFull() && !context.isInventoryFull() && !isBankOpen(context))
        {
            reset();
            return;
        }

        // Priority: if a bank is interactable/visible now, clear any busy lock and attempt to open it immediately.
        // This prevents over-navigation when the booth is already within clicking range.
        try
        {
            if (!isBankOpen(context))
            {
                net.runelite.api.GameObject bankObj = ObjectFinder.findNearestBankInteractable(context);
                if (bankObj != null)
                {
                    java.awt.Point click = ObjectFinder.projectToCanvas(context, bankObj);
                    if (click != null)
                    {
                        context.clearBusyLock();
                        // Drive an immediate open attempt via deposit task (which will click-to-open when closed)
                        deposit.run(context);
                        if (isBankOpen(context))
                        {
                            transition(Phase.DEPOSIT);
                            return;
                        }
                    }
                }
            }
        }
        catch (Exception ignored)
        {
        }

        // Respect busy locks to allow staged clicks and movement to settle (unless we just tried an open above).
        if (context.isBusy() && !context.timedOutSince(1500))
        {
            return;
        }

        // Ensure we enter the flow when inventory is near/full.
        if (phase == Phase.IDLE && (context.isInventoryNearFull() || context.isInventoryFull()))
        {
            transition(Phase.NAVIGATE);
        }

        // If bank UI is open at any point, skip to DEPOSIT.
        if (isBankOpen(context))
        {
            transition(Phase.DEPOSIT);
        }

        switch (phase)
        {
            case NAVIGATE:
            {
                // Opportunistically try to stage a bank click if a bank is interactable now before moving.
                deposit.run(context);
                if (isBankOpen(context))
                {
                    transition(Phase.DEPOSIT);
                    return;
                }
                if (BankClicker.isPending())
                {
                    transition(Phase.CLICK_STAGE);
                    return;
                }
                // Otherwise, continue moving toward a bank. Clear any artificial busy lock to ensure progress.
                context.clearBusyLock();
                navigate.run(context);
                if (now - phaseStartMs > NAVIGATE_MAX_MS)
                {
                    // Give staging another try after timeout
                    transition(Phase.CLICK_STAGE);
                }
                return;
            }
            case CLICK_STAGE:
            {
                // Continue staged hover->click or attempt a new stage
                deposit.run(context);
                if (isBankOpen(context))
                {
                    transition(Phase.DEPOSIT);
                    return;
                }
                if (BankClicker.isPending())
                {
                    // Waiting for next tick to click; remain in this phase
                    return;
                }
                // If no pending and not open, either retry staging or navigate again
                if (++clickRetries >= CLICK_RETRY_MAX)
                {
                    clickRetries = 0;
                    transition(Phase.NAVIGATE);
                }
                return;
            }
            case OPEN_WAIT:
            {
                // Legacy placeholder: we rely on deposit.run to perform open; keep an upper bound wait
                if (isBankOpen(context))
                {
                    transition(Phase.DEPOSIT);
                    return;
                }
                if (now - phaseStartMs > OPEN_WAIT_MAX_MS)
                {
                    transition(Phase.NAVIGATE);
                }
                return;
            }
            case DEPOSIT:
            {
                // Bank is open: deposit inventory. BankDepositTask handles clicking deposit-all and closing if empty.
                deposit.run(context);
                if (context.getInventoryFreeSlots() == 28)
                {
                    transition(Phase.CLOSE);
                }
                return;
            }
            case CLOSE:
            {
                // Let BankDepositTask close via ESC/close button; once closed, we are done.
                if (!isBankOpen(context))
                {
                    reset();
                    return;
                }
                // Nudge deposit again to drive close if still open.
                deposit.run(context);
                return;
            }
            case IDLE:
            default:
                // Nothing to do; wait for need to bank
                return;
        }
    }

    private void transition(Phase next)
    {
        this.phase = next;
        this.phaseStartMs = System.currentTimeMillis();
    }

    private void reset()
    {
        this.phase = Phase.IDLE;
        this.phaseStartMs = 0L;
        this.clickRetries = 0;
    }

    private boolean isBankOpen(TaskContext context)
    {
        try
        {
            Widget bank = context.client.getWidget(WidgetInfo.BANK_CONTAINER);
            return bank != null && !bank.isHidden();
        }
        catch (Exception e)
        {
            return false;
        }
    }
}
