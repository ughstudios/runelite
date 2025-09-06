package net.runelite.client.plugins.rlbot;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

@Singleton
public class RLBotControlWindow extends JFrame {
	private final RLBotAgent agent;

	@Inject
	public RLBotControlWindow(RLBotAgent agent) {
		super("RLBot Controls");
		this.agent = agent;
		setLayout(new BorderLayout());
		setSize(500, 150);
		setAlwaysOnTop(true);
		setLocationByPlatform(true);
		setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);

		JPanel header = new JPanel(new FlowLayout(FlowLayout.LEFT));
		header.add(new JLabel("Manual Actions"));
		add(header, BorderLayout.NORTH);

		JPanel buttons = new JPanel(new FlowLayout(FlowLayout.LEFT));
		JButton btnNavTrees = new JButton("Navigate Trees");
		JButton btnChopTree = new JButton("Chop Tree");
		JButton btnNavBank = new JButton("Navigate Bank");
		JButton btnBankDeposit = new JButton("Bank Deposit");
		JButton btnCrossWilderness = new JButton("Cross Wilderness");
		JButton btnCrossWildernessOut = new JButton("Cross Out");
		JButton btnExplore = new JButton("Explore");
		JButton btnRotate = new JButton("Rotate Camera");
		JButton btnRotL = new JButton("←");
		JButton btnRotR = new JButton("→");
		JButton btnRotU = new JButton("↑");
		JButton btnRotD = new JButton("↓");
		javax.swing.JTextField rotateSteps = new javax.swing.JTextField("4", 3);
		javax.swing.JTextField exploreTiles = new javax.swing.JTextField("6", 3);
		// Manual canvas click controls
		javax.swing.JLabel lblClick = new javax.swing.JLabel("Click X/Y:");
		javax.swing.JTextField clickX = new javax.swing.JTextField("400", 4);
		javax.swing.JTextField clickY = new javax.swing.JTextField("300", 4);
		javax.swing.JTextField clickAction = new javax.swing.JTextField("Chop down", 10);
		JButton btnClickXY = new JButton("Click X/Y");
		JButton btnNorth = new JButton("N");
		JButton btnSouth = new JButton("S");
		JButton btnEast = new JButton("E");
		JButton btnWest = new JButton("W");
		buttons.add(btnNavTrees);
		buttons.add(btnChopTree);
		buttons.add(btnNavBank);
		buttons.add(btnBankDeposit);
		buttons.add(btnCrossWilderness);
		buttons.add(btnCrossWildernessOut);
		buttons.add(btnExplore);
		buttons.add(exploreTiles);
		buttons.add(btnNorth);
		buttons.add(btnSouth);
		buttons.add(btnEast);
		buttons.add(btnWest);
		buttons.add(btnRotate);
		buttons.add(rotateSteps);
		buttons.add(btnRotL);
		buttons.add(btnRotR);
		buttons.add(btnRotU);
		buttons.add(btnRotD);
		// Add manual canvas click controls to panel
		buttons.add(lblClick);
		buttons.add(clickX);
		buttons.add(clickY);
		buttons.add(clickAction);
		buttons.add(btnClickXY);
		add(buttons, BorderLayout.CENTER);

		btnNavTrees.addActionListener(e -> safe(() -> agent.triggerNavigateTrees()));
		btnChopTree.addActionListener(e -> safe(() -> agent.triggerChopTree()));
		btnNavBank.addActionListener(e -> safe(() -> agent.triggerNavigateBank()));
		btnBankDeposit.addActionListener(e -> safe(() -> agent.triggerBankDeposit()));
		btnCrossWilderness.addActionListener(e -> safe(() -> agent.triggerCrossWilderness()));
		btnCrossWildernessOut.addActionListener(e -> safe(() -> agent.triggerCrossWildernessOut()));
		btnExplore.addActionListener(e -> safe(() -> agent.triggerExplore()));
		btnRotate.addActionListener(e -> safe(() -> agent.triggerRotateCamera()));
		btnRotL.addActionListener(e -> safe(() -> agent.triggerRotateDirection("LEFT", parseInt(rotateSteps.getText(), 4))));
		btnRotR.addActionListener(e -> safe(() -> agent.triggerRotateDirection("RIGHT", parseInt(rotateSteps.getText(), 4))));
		btnRotU.addActionListener(e -> safe(() -> agent.triggerRotateDirection("UP", parseInt(rotateSteps.getText(), 4))));
		btnRotD.addActionListener(e -> safe(() -> agent.triggerRotateDirection("DOWN", parseInt(rotateSteps.getText(), 4))));
		btnNorth.addActionListener(e -> safe(() -> agent.triggerExploreCardinal("NORTH", parseInt(exploreTiles.getText(), 6))));
		btnSouth.addActionListener(e -> safe(() -> agent.triggerExploreCardinal("SOUTH", parseInt(exploreTiles.getText(), 6))));
		btnEast.addActionListener(e -> safe(() -> agent.triggerExploreCardinal("EAST", parseInt(exploreTiles.getText(), 6))));
		btnWest.addActionListener(e -> safe(() -> agent.triggerExploreCardinal("WEST", parseInt(exploreTiles.getText(), 6))));
		btnClickXY.addActionListener(e -> safe(() -> agent.triggerClickAtXY(
			parseInt(clickX.getText(), 400),
			parseInt(clickY.getText(), 300),
			clickAction.getText() != null ? clickAction.getText().trim() : "Chop down"
		)));

		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				setVisible(false);
			}
		});
	}

	private void safe(Runnable r) {
		try { r.run(); } catch (Exception ignored) {}
	}

	private static int parseInt(String s, int def) {
		try { return Integer.parseInt(s.trim()); } catch (Exception e) { return def; }
	}

	public void showWindow() {
		SwingUtilities.invokeLater(() -> setVisible(true));
	}

	public void hideWindow() {
		SwingUtilities.invokeLater(() -> setVisible(false));
	}
}
