from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from agents.ppo_agent import PPOAgent
from core.evaluator import Evaluator

log = logging.getLogger(__name__)


def evaluate(
    model_path:    str,
    data_path:     str,
    config_path:   str   = "config/config.yaml",
    n_episodes:    int   = 10,
    render:        bool  = False,
    save_results_: bool  = True,
    results_dir:   str   = "data/evaluation",
    export_onnx_:  bool  = False,
    onnx_path:     str   = "models/model.onnx",
    visualise:     bool  = False,
    vis_window:    int   = 120,
    vis_pause:     float = 0.01,
) -> dict:
    agent     = PPOAgent(model_path)
    evaluator = Evaluator(config_path)

    if export_onnx_:
        from agents.train import export_onnx
        from env.preprocessor import load_csv, load_npy, preprocess
        path    = Path(data_path)
        raw     = load_npy(path) if path.suffix.lower() == ".npy" else load_csv(path)[0]
        obs_dim = (
            evaluator.env_cfg["window_size"] * preprocess(raw).shape[1]
            + evaluator.env_cfg.get("max_positions", 3) * 5
            + 2
        )
        export_onnx(agent.model, onnx_path, obs_dim)

    return evaluator.run(
        agent       = agent,
        data_path   = data_path,
        n_episodes  = n_episodes,
        save        = save_results_,
        results_dir = results_dir,
        visualise   = visualise,
        vis_window  = vis_window,
        vis_pause   = vis_pause,
    )
