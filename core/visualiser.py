"""
env/visualiser.py
-----------------
Live per-step debugging dashboard for a single TradingEnv episode.

Usage
-----
    from core.visualiser import EpisodeVisualiser

    vis = EpisodeVisualiser(window=120, pause=0.01)
    obs, _ = env.reset()
    vis.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        vis.update(env, reward, action=action)   # pass action each step

    vis.save("episode.png")
    vis.close()

Layout
------
    ┌──────────────────────────────────────────┐
    │  [0]  Price  +  trade entry/exit markers │
    ├──────────────────────────────────────────┤
    │  [1]  Action strip  (colour per step)    │  ← thin coloured bar
    ├──────────────────────────────────────────┤
    │  [2]  Equity curve vs initial balance    │
    ├────────────────────┬─────────────────────┤
    │  [3]  Reward/step  │  [4] Position slots │
    └────────────────────┴─────────────────────┘

Action colours
--------------
    HOLD  → muted grey
    BUY   → green
    SELL  → red
    CLOSE → amber

Notes
-----
- Works only in the main process (matplotlib GUI).
  Do NOT use inside SubprocVecEnv workers.
- Call reset() at the start of each episode to clear history.
- action kwarg to update() accepts np.ndarray([direction, lot_tier])
  or None (treated as HOLD) — compatible with both PPO and strategies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action index constants (mirrors simulator.Action)
# ---------------------------------------------------------------------------
_A_HOLD      = 0
_A_BUY       = 1
_A_SELL      = 2
_A_CLOSE_L   = 3  # CLOSE_LONG
_A_CLOSE_S   = 4  # CLOSE_SHORT

_ACTION_LABELS = {
    _A_HOLD:    "HOLD",
    _A_BUY:     "BUY",
    _A_SELL:    "SELL",
    _A_CLOSE_L: "CLOSE_L",
    _A_CLOSE_S: "CLOSE_S",
}

# ---------------------------------------------------------------------------
# Colour palette  (dark terminal aesthetic)
# ---------------------------------------------------------------------------
_BG     = "#0d0f14"
_PANEL  = "#13161e"
_GRID   = "#1e2230"
_TEXT   = "#c8ccd8"
_MUTED  = "#4a5068"
_GREEN  = "#3ddc97"
_RED    = "#f06474"
_AMBER  = "#f5c542"
_BLUE   = "#5b9cf6"
_PURPLE = "#a78bfa"
_WHITE  = "#e8eaf0"

# Colour per action index — used for the strip and its legend
_ACTION_COLOURS = {
    _A_HOLD:    _MUTED,
    _A_BUY:     _GREEN,
    _A_SELL:    _RED,
    _A_CLOSE_L: _AMBER,
    _A_CLOSE_S: _PURPLE,
}


def _import_matplotlib():
    """Lazy import so the module can be imported without matplotlib installed."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")       # swap to "Qt5Agg" if TkAgg unavailable
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        return plt, gridspec, mcolors, mpatches, Line2D
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for EpisodeVisualiser. "
            "Install it with:  pip install matplotlib"
        ) from e


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EpisodeVisualiser:
    """
    Standalone live dashboard for a TradingEnv episode.

    Parameters
    ----------
    window : int
        Number of price bars shown in the price/action panels at any one
        time.  Older bars scroll off the left edge.
    pause : float
        Seconds to pause after each redraw.
        0.001 is barely perceptible; 0.05 is comfortable for watching.
    figsize : tuple[float, float]
        Figure dimensions in inches.
    """

    def __init__(
        self,
        window:  int                 = 120,
        pause:   float               = 0.01,
        figsize: tuple[float, float] = (18, 11),
    ):
        self.window  = window
        self.pause   = pause
        self.figsize = figsize

        # Accumulated episode history
        self._prices:   list[float] = []
        self._equities: list[float] = []
        self._rewards:  list[float] = []
        self._steps:    list[int]   = []
        self._actions:  list[int]   = []   # direction index per step

        # Trade markers: (step, price)
        self._buy_markers:   list[tuple[int, float]] = []
        self._sell_markers:  list[tuple[int, float]] = []
        self._close_markers: list[tuple[int, float]] = []

        # Internal tracking state
        self._last_trade_count: int = 0
        self._last_n_positions: int = 0

        # Matplotlib objects (created lazily on first update)
        self._fig         = None
        self._axes: list  = []
        self._plt         = None
        self._mcolors     = None
        self._mpatches    = None
        self._Line2D      = None

        # Episode-level constants set on first update
        self._initial_balance: float = 0.0
        self._max_positions:   int   = 0
        self._symbol_name:     str   = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated history.  Call at the start of each episode."""
        self._prices.clear()
        self._equities.clear()
        self._rewards.clear()
        self._steps.clear()
        self._actions.clear()
        self._buy_markers.clear()
        self._sell_markers.clear()
        self._close_markers.clear()
        self._last_trade_count = 0
        self._last_n_positions = 0

        if self._fig is not None:
            for ax in self._axes:
                ax.cla()
            self._plt.draw()

        logger.debug("EpisodeVisualiser reset.")

    def update(
        self,
        env:    "TradingEnv",
        reward: float,
        action: Optional[np.ndarray] = None,
    ) -> None:
        """
        Ingest one step's worth of data and redraw the dashboard.

        Call immediately after env.step() on every step.

        Parameters
        ----------
        env    : The live TradingEnv (read-only access to its state).
        reward : Scalar reward returned by env.step() this step.
        action : np.ndarray([direction_idx, lot_tier_idx]) or None.
                 When None the action is recorded as HOLD (index 0).
        """
        step  = env._step
        price = env._current_price()

        unrealized = env._sim.total_unrealized_pnl(price)
        equity     = env._balance + unrealized

        # Record direction index (0=HOLD if action is None)
        direction_idx = int(action[0]) if action is not None else _A_HOLD

        self._prices.append(price)
        self._equities.append(equity)
        self._rewards.append(float(reward))
        self._steps.append(step)
        self._actions.append(direction_idx)

        # Detect newly closed trade
        n_trades = len(env._episode_trades)
        if n_trades > self._last_trade_count:
            latest = env._episode_trades[-1]
            self._close_markers.append((step, float(latest.exit_price)))
            self._last_trade_count = n_trades

        # Detect newly opened position
        n_open = len(env._sim._positions)
        if n_open > self._last_n_positions:
            newest = env._sim._positions[-1]
            if newest.direction.name == "LONG":
                self._buy_markers.append((step, float(newest.entry_price)))
            else:
                self._sell_markers.append((step, float(newest.entry_price)))
        self._last_n_positions = n_open

        if self._fig is None:
            self._build_figure(env)

        self._redraw(env)

    def save(self, path: str) -> None:
        """Save the current dashboard frame to a PNG file."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=150, bbox_inches="tight",
                              facecolor=_BG)
            logger.info("Visualiser frame saved → %s", path)

    def close(self) -> None:
        """Close the matplotlib window."""
        if self._plt is not None and self._fig is not None:
            self._plt.close(self._fig)
            self._fig = None
        logger.debug("EpisodeVisualiser closed.")

    # ------------------------------------------------------------------
    # Private: figure construction
    # ------------------------------------------------------------------

    def _build_figure(self, env: "TradingEnv") -> None:
        plt, gridspec, mcolors, mpatches, Line2D = _import_matplotlib()
        self._plt      = plt
        self._mcolors  = mcolors
        self._mpatches = mpatches
        self._Line2D   = Line2D

        fig = plt.figure(figsize=self.figsize, facecolor=_BG)
        fig.canvas.manager.set_window_title(
            f"ForexRL Debug — {env.spec.name}"
        )

        # 5 rows: price | action strip | equity | reward+positions
        gs = gridspec.GridSpec(
            5, 2,
            figure        = fig,
            hspace        = 0.50,
            wspace        = 0.30,
            left          = 0.06,
            right         = 0.97,
            top           = 0.93,
            bottom        = 0.06,
            height_ratios = [2.2, 0.25, 1.4, 1.4, 1.4],
        )

        ax_price     = fig.add_subplot(gs[0, :])   # full-width
        ax_action    = fig.add_subplot(gs[1, :])   # full-width thin strip
        ax_equity    = fig.add_subplot(gs[2, :])   # full-width
        ax_reward    = fig.add_subplot(gs[3, 0])   # bottom-left
        ax_positions = fig.add_subplot(gs[3, 1])   # bottom-right

        self._axes = [ax_price, ax_action, ax_equity, ax_reward, ax_positions]
        self._fig  = fig

        for ax in self._axes:
            ax.set_facecolor(_PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            ax.tick_params(colors=_MUTED, labelsize=8)

        # Action strip — minimal chrome
        ax_action.set_xticks([])
        ax_action.set_yticks([])
        ax_action.set_ylabel("action", color=_MUTED, fontsize=7,
                             rotation=0, labelpad=28, va="center")

        self._initial_balance = env.initial_balance
        self._max_positions   = env.max_positions
        self._symbol_name     = env.spec.name

        plt.ion()
        plt.show(block=False)

    # ------------------------------------------------------------------
    # Private: per-step redraw
    # ------------------------------------------------------------------

    def _redraw(self, env: "TradingEnv") -> None:
        ax_price, ax_action, ax_equity, ax_reward, ax_positions = self._axes

        steps    = np.array(self._steps)
        prices   = np.array(self._prices)
        equities = np.array(self._equities)
        rewards  = np.array(self._rewards)
        actions  = np.array(self._actions, dtype=np.int32)

        # Visible window (price + action strip share the same x-window)
        vis_start  = max(0, len(steps) - self.window)
        vis_steps  = steps[vis_start:]
        vis_prices = prices[vis_start:]
        vis_acts   = actions[vis_start:]
        win_origin = int(vis_steps[0]) if len(vis_steps) else 0

        # ----------------------------------------------------------------
        # [0] Price + trade markers
        # ----------------------------------------------------------------
        ax_price.cla()
        ax_price.set_facecolor(_PANEL)
        ax_price.set_title(
            f"{self._symbol_name}  —  Price & Trades",
            color=_TEXT, fontsize=10, fontweight="bold", pad=6,
        )
        ax_price.plot(vis_steps, vis_prices, color=_BLUE,
                      linewidth=1.0, alpha=0.9, zorder=2)
        ax_price.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)

        for s, p in self._buy_markers:
            if s >= win_origin:
                ax_price.scatter(s, p, marker="^", color=_GREEN,
                                 s=90, zorder=5, linewidths=0)

        for s, p in self._sell_markers:
            if s >= win_origin:
                ax_price.scatter(s, p, marker="v", color=_RED,
                                 s=90, zorder=5, linewidths=0)

        for s, p in self._close_markers:
            if s >= win_origin:
                ax_price.scatter(s, p, marker="D", color=_AMBER,
                                 s=55, zorder=5, linewidths=0)

        # Dashed horizontal lines for open position entries
        for pos in env._sim._positions:
            ax_price.axhline(
                pos.entry_price,
                color     = _GREEN if pos.direction.name == "LONG" else _RED,
                linewidth = 0.8,
                linestyle = "--",
                alpha     = 0.55,
                zorder    = 1,
            )

        legend_elements = [
            self._Line2D([0], [0], marker="^", color="w",
                         markerfacecolor=_GREEN,  markersize=8, label="Buy open"),
            self._Line2D([0], [0], marker="v", color="w",
                         markerfacecolor=_RED,    markersize=8, label="Sell open"),
            self._Line2D([0], [0], marker="D", color="w",
                         markerfacecolor=_AMBER,  markersize=7, label="Close"),
        ]
        ax_price.legend(
            handles=legend_elements, loc="upper left",
            facecolor=_BG, edgecolor=_GRID,
            labelcolor=_TEXT, fontsize=7, framealpha=0.8,
        )
        for spine in ax_price.spines.values():
            spine.set_edgecolor(_GRID)
        ax_price.tick_params(colors=_MUTED, labelsize=8)

        # ----------------------------------------------------------------
        # [1] Action strip
        # ----------------------------------------------------------------
        ax_action.cla()
        ax_action.set_facecolor(_PANEL)
        ax_action.set_xticks([])
        ax_action.set_yticks([])
        ax_action.set_ylabel("action", color=_MUTED, fontsize=7,
                             rotation=0, labelpad=28, va="center")
        for spine in ax_action.spines.values():
            spine.set_edgecolor(_GRID)

        if len(vis_steps) > 0:
            # Build a colour array — one colour per visible step
            strip_colours = [_ACTION_COLOURS[a] for a in vis_acts]

            # Draw as thin vertical bars filling the full height [0, 1]
            ax_action.bar(
                vis_steps,
                np.ones(len(vis_steps)),      # all height=1
                width  = 1.0,
                color  = strip_colours,
                align  = "center",
                bottom = 0,
                linewidth = 0,
            )
            ax_action.set_xlim(vis_steps[0] - 0.5, vis_steps[-1] + 0.5)
            ax_action.set_ylim(0, 1)

            # Inline legend — coloured squares + labels at the right edge
            legend_x  = vis_steps[-1] + 0.5 + (vis_steps[-1] - vis_steps[0]) * 0.01
            items = [
                (_ACTION_COLOURS[_A_HOLD],    "HOLD"),
                (_ACTION_COLOURS[_A_BUY],     "BUY"),
                (_ACTION_COLOURS[_A_SELL],    "SELL"),
                (_ACTION_COLOURS[_A_CLOSE_L], "CLOSE_L"),
                (_ACTION_COLOURS[_A_CLOSE_S], "CLOSE_S"),
            ]
            patches = [
                self._mpatches.Patch(color=c, label=lbl)
                for c, lbl in items
            ]
            ax_action.legend(
                handles   = patches,
                loc       = "center right",
                ncol      = 5,
                facecolor = _BG,
                edgecolor = _GRID,
                labelcolor= _TEXT,
                fontsize  = 6.5,
                framealpha= 0.85,
                borderpad = 0.4,
                handlelength = 0.8,
                handleheight = 0.8,
            )

        # ----------------------------------------------------------------
        # [2] Equity curve
        # ----------------------------------------------------------------
        ax_equity.cla()
        ax_equity.set_facecolor(_PANEL)
        ax_equity.set_title("Equity", color=_TEXT, fontsize=9, pad=4)
        ax_equity.plot(steps, equities, color=_PURPLE,
                       linewidth=1.2, alpha=0.95, zorder=3)
        ax_equity.axhline(
            self._initial_balance,
            color=_MUTED, linewidth=0.8, linestyle="--", alpha=0.7,
        )
        ax_equity.fill_between(
            steps, equities, self._initial_balance,
            where=(equities >= self._initial_balance),
            alpha=0.15, color=_GREEN, interpolate=True,
        )
        ax_equity.fill_between(
            steps, equities, self._initial_balance,
            where=(equities < self._initial_balance),
            alpha=0.15, color=_RED, interpolate=True,
        )
        if len(equities):
            cur_eq = equities[-1]
            colour = _GREEN if cur_eq >= self._initial_balance else _RED
            ax_equity.annotate(
                f"${cur_eq:,.2f}",
                xy=(steps[-1], cur_eq),
                xytext=(4, 0), textcoords="offset points",
                color=colour, fontsize=8, fontweight="bold",
            )
        ax_equity.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
        for spine in ax_equity.spines.values():
            spine.set_edgecolor(_GRID)
        ax_equity.tick_params(colors=_MUTED, labelsize=8)

        # ----------------------------------------------------------------
        # [3] Reward per step
        # ----------------------------------------------------------------
        ax_reward.cla()
        ax_reward.set_facecolor(_PANEL)
        ax_reward.set_title("Reward / Step", color=_TEXT, fontsize=9, pad=4)

        if len(rewards):
            bar_colours = np.where(rewards >= 0, _GREEN, _RED)
            ax_reward.bar(steps, rewards, color=bar_colours,
                          width=1.0, alpha=0.75, zorder=2)
            ax_reward.axhline(0, color=_MUTED, linewidth=0.7)

            if len(rewards) >= 10:
                kernel  = min(20, len(rewards))
                rolling = np.convolve(
                    rewards, np.ones(kernel) / kernel, mode="valid"
                )
                ax_reward.plot(
                    steps[kernel - 1:], rolling,
                    color=_AMBER, linewidth=1.0, alpha=0.85, zorder=3,
                )

        ax_reward.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
        for spine in ax_reward.spines.values():
            spine.set_edgecolor(_GRID)
        ax_reward.tick_params(colors=_MUTED, labelsize=8)

        # ----------------------------------------------------------------
        # [4] Position slot heatmap
        # ----------------------------------------------------------------
        ax_positions.cla()
        ax_positions.set_facecolor(_PANEL)
        ax_positions.set_title("Open Positions", color=_TEXT, fontsize=9, pad=4)

        price     = env._current_price()
        pos_vec   = env._sim.position_state_vector(price, self._max_positions)
        slot_size = 5
        n_slots   = self._max_positions
        matrix    = pos_vec.reshape(n_slots, slot_size)

        cmap = self._mcolors.LinearSegmentedColormap.from_list(
            "rdgr", [_RED, _PANEL, _GREEN]
        )
        ax_positions.imshow(matrix, aspect="auto", cmap=cmap,
                            vmin=-1.0, vmax=1.0)

        col_labels = ["filled", "dir", "lots", "Δentry", "uPnL"]
        row_labels = [f"Slot {i + 1}" for i in range(n_slots)]
        ax_positions.set_xticks(range(slot_size))
        ax_positions.set_xticklabels(col_labels, fontsize=7, color=_MUTED)
        ax_positions.set_yticks(range(n_slots))
        ax_positions.set_yticklabels(row_labels, fontsize=7, color=_MUTED)
        ax_positions.tick_params(length=0)

        for r in range(n_slots):
            for c in range(slot_size):
                val = matrix[r, c]
                ax_positions.text(
                    c, r, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=6.5,
                    color=_WHITE if abs(val) > 0.3 else _MUTED,
                )

        for spine in ax_positions.spines.values():
            spine.set_edgecolor(_GRID)

        # ----------------------------------------------------------------
        # Figure-level status strip
        # ----------------------------------------------------------------
        n_open   = env._sim.n_positions
        n_trades = len(env._episode_trades)
        upnl     = env._sim.total_unrealized_pnl(price)
        bal      = env._balance

        # Show current action name prominently in the title
        current_action_label = _ACTION_LABELS.get(
            self._actions[-1] if self._actions else _A_HOLD, "HOLD"
        )
        action_colour = _ACTION_COLOURS.get(
            self._actions[-1] if self._actions else _A_HOLD, _MUTED
        )

        self._fig.suptitle(
            f"Step {env._step}  |  Balance ${bal:,.2f}  |  "
            f"uPnL ${upnl:+.2f}  |  Open {n_open}  |  Trades {n_trades}"
            f"  |  Action: {current_action_label}",
            color=_TEXT, fontsize=9, y=0.98,
        )

        self._plt.pause(self.pause)