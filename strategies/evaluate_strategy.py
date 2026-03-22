from __future__ import annotations

from core.evaluator import Evaluator
from strategies.base import BaseStrategy


def evaluate_strategy(
    strategy:     BaseStrategy,
    data_path:    str,
    config_path:  str   = "config/config.yaml",
    n_episodes:   int   = 1,
    save_results: bool  = True,
    results_dir:  str   = "data/evaluation",
    visualise:    bool  = False,
    vis_window:   int   = 120,
    vis_pause:    float = 0.01,
) -> dict:
    return Evaluator(config_path).run(
        agent       = strategy,
        data_path   = data_path,
        n_episodes  = n_episodes,
        save        = save_results,
        results_dir = results_dir,
        visualise   = visualise,
        vis_window  = vis_window,
        vis_pause   = vis_pause,
    )
