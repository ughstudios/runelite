package net.runelite.client.plugins.rlbot.tasks;

import java.awt.Rectangle;
import java.awt.event.KeyEvent;
import net.runelite.api.GameObject;
import net.runelite.api.MenuAction;
import net.runelite.api.Player;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;

/**
 * Minimal bank interaction task: open a nearby bank and deposit the entire
 * inventory when full.
 */
public class BankDepositTask implements Task
{
    // Revert to simple flow; no long-running async sequence here

    @Override
    public void run(TaskContext context)
    {
        if (context.isBusy() && !context.timedOutSince(3000))
        {
            return;
        }

        if (isBankOpen(context))
        {
            depositInventory(context);
            return;
        }

        GameObject bankObject = ObjectFinder.findNearestBankInteractable(context);
        if (bankObject == null)
        {
            return;
        }

        BankDiscovery.scanAndDiscoverBanks(context);
        BankDiscovery.setLastTargetedBank(bankObject.getWorldLocation());

        // Staged hover -> click across ticks (mirrors TreeClicker flow)
        BankClicker.Result clickRes = BankClicker.clickBank(context, bankObject);
        if (clickRes == BankClicker.Result.STAGED)
        {
            // Hover staged this tick; click will occur on next tick
            return;
        }
        if (clickRes == BankClicker.Result.CLICKED)
        {
            context.setBusyForMs(300);
            return;
        }
        // If projection failed or click failed, try direct menu interaction as a last resort
        if (clickRes == BankClicker.Result.NONE || clickRes == BankClicker.Result.FAILED)
        {
            boolean interacted = context.input.interactWithGameObject(bankObject, "Bank");
            if (interacted)
            {
                context.setBusyForMs(250);
                return;
            }
        }
        context.logger.warn("[BankDepositTask] Bank click failed (" + clickRes + ") and direct interact failed");
    }

    private void depositInventory(TaskContext context)
    {
        if (context.getInventoryFreeSlots() == 28)
        {
            // Inventory already empty – close the bank UI.
            if (!UiHelper.closeBankMainById(context, WidgetInfo.BANK_CONTAINER.getId()))
            {
                if (!UiHelper.clickCloseIfVisible(context))
                {
                    context.input.pressKey(KeyEvent.VK_ESCAPE);
                }
            }
            context.setBusyForMs(200);
            return;
        }

        Widget depositAll = context.client.getWidget(WidgetInfo.BANK_DEPOSIT_INVENTORY);
        if (depositAll != null && !depositAll.isHidden())
        {
            Rectangle bounds = depositAll.getBounds();
            if (bounds != null && bounds.width > 0)
            {
                int widgetId = depositAll.getId();
                context.clientThread.invoke(() -> {
                    try
                    {
                        context.client.menuAction(-1, widgetId, MenuAction.CC_OP, 1, -1, "Deposit inventory", "");
                    }
                    catch (Exception e)
                    {
                        context.logger.error("[BankDeposit] menuAction failed: " + e.getMessage());
                    }
                });
                context.setBusyForMs(400);
                return;
            }
        }

        // Fallback: click centre of deposit-all widget region if the direct action failed
        if (depositAll != null && depositAll.getBounds() != null)
        {
            Rectangle b = depositAll.getBounds();
            java.awt.Point p = new java.awt.Point(b.x + b.width / 2, b.y + b.height / 2);
            context.input.smoothMouseMove(p);
            if (context.input.clickAt(p))
            {
                context.setBusyForMs(400);
            }
            else
            {
                context.logger.warn("[BankDeposit] Fallback click failed at " + p);
            }
        }
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

    // openAndDeposit flow removed; we rely on BankClicker and UI deposit
}
