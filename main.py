#!/usr/bin/env python3
"""
Forex RL Trader CLI

Usage:
    python main.py download  --symbols EURUSD GBPUSD --start 2023-01-01 --end 2024-12-31
    python main.py generate  --symbol EURUSD --samples 10000
    python main.py train     --data data/raw --symbols EURUSD USDJPY --timesteps 500000
    python main.py evaluate  --model models/ppo_final.zip --data data/raw/EURUSD.csv
    python main.py baseline  --strategy ma_cross --data data/raw/EURUSD.csv
    python main.py baseline  --strategy random   --data data/raw/EURUSD.csv --visualise
"""

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> int:
    try:
        from data.downloader import MT5Downloader
    except ImportError:
        logging.error("MetaTrader5 not installed.  Run: pip install MetaTrader5")
        return 1

    log = logging.getLogger("download")
    downloader = MT5Downloader()

    try:
        if not downloader.connect():
            log.error("Failed to connect to MT5 terminal.")
            return 1

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = downloader.download_multiple(
            symbols    = args.symbols,
            timeframe  = args.timeframe,
            start_date = args.start,
            end_date   = args.end,
            output_dir = str(output_dir),
        )

        log.info("Downloaded %d symbol(s):", len(results))
        for symbol, df in results.items():
            log.info("  %s: %d bars", symbol, len(df))

        return 0

    finally:
        downloader.disconnect()


def cmd_generate(args: argparse.Namespace) -> int:
    from data.generator import generate_and_save

    log = logging.getLogger("generate")
    log.info(
        "Generating %d synthetic candles for %s (seed=%d) → %s",
        args.samples, args.symbol, args.seed, args.output,
    )

    generate_and_save(
        output_dir = args.output,
        symbol     = args.symbol,
        n_samples  = args.samples,
        seed       = args.seed,
    )

    log.info("Done.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from agents.train import train

    log = logging.getLogger("train")
    log.info("Starting training run.")
    log.info("  data dir : %s", args.data)
    log.info("  symbols  : %s", args.symbols or "all files in dir")
    log.info("  config   : %s", args.config)
    log.info("  timesteps: %s", args.timesteps or "from config")
    log.info("  seed     : %d", args.seed)

    train(
        config_path      = args.config,
        data_dir         = args.data,
        symbols          = args.symbols,
        model_path       = args.model,
        output_dir       = args.output,
        total_timesteps  = args.timesteps,
        seed             = args.seed,
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from agents.evaluate import evaluate

    log = logging.getLogger("evaluate")
    log.info("Evaluating model: %s", args.model)
    log.info("  data     : %s", args.data)
    log.info("  episodes : %d", args.episodes)

    evaluate(
        model_path   = args.model,
        data_path    = args.data,
        config_path  = args.config,
        n_episodes   = args.episodes,
        render       = args.render,
        save_results = not args.no_save,
        results_dir  = args.output,
        export_onnx_ = args.export_onnx,
        onnx_path    = args.onnx_path,
        visualise    = args.visualise,
        vis_window   = args.vis_window,
        vis_pause    = args.vis_pause,
    )
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from strategies.evaluate_strategy import evaluate_strategy

    log = logging.getLogger("baseline")

    # ------------------------------------------------------------------
    # Build the strategy from CLI args
    # ------------------------------------------------------------------
    strategy_name = args.strategy.lower()

    if strategy_name == "random":
        from strategies.baselines import RandomStrategy
        strategy = RandomStrategy(seed=args.seed, lot_tier=args.lot_tier)

    elif strategy_name == "ma_cross":
        from strategies.baselines import MACrossStrategy
        strategy = MACrossStrategy(
            fast     = args.fast,
            slow     = args.slow,
            lot_tier = args.lot_tier,
        )

    else:
        log.error(
            "Unknown strategy '%s'. Choices: random, ma_cross", strategy_name
        )
        return 1

    log.info(
        "Running baseline  strategy=%s  data=%s  episodes=%d",
        strategy.name, args.data, args.episodes,
    )

    evaluate_strategy(
        strategy     = strategy,
        data_path    = args.data,
        config_path  = args.config,
        n_episodes   = args.episodes,
        save_results = not args.no_save,
        results_dir  = args.output,
        visualise    = args.visualise,
        vis_window   = args.vis_window,
        vis_pause    = args.vis_pause,
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "forex-rl",
        description = "Forex Reinforcement Learning Trader",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  %(prog)s download --symbols EURUSD GBPUSD --start 2023-01-01 --end 2024-12-31
  %(prog)s generate --symbol EURUSD --samples 20000
  %(prog)s train    --data data/raw --symbols EURUSD USDJPY --timesteps 500000
  %(prog)s evaluate --model models/ppo_final.zip --data data/raw/EURUSD.csv
  %(prog)s baseline --strategy ma_cross --data data/raw/EURUSD.csv
  %(prog)s baseline --strategy ma_cross --data data/raw/EURUSD.csv --fast 5 --slow 20 --visualise
  %(prog)s baseline --strategy random   --data data/raw/EURUSD.csv --episodes 5
        """,
    )

    parser.add_argument(
        "--log-level",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Logging verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- download --------------------------------------------------------
    dl = sub.add_parser("download", help="Download historical data from MT5")
    dl.add_argument("--symbols",   nargs="+", default=["EURUSD"], metavar="SYM")
    dl.add_argument("--timeframe", default="H1",
                    choices=["M1","M5","M15","M30","H1","H4","D1","W1","MN1"])
    dl.add_argument("--start",  default="2024-01-01", help="Start date YYYY-MM-DD")
    dl.add_argument("--end",    default="2024-12-31", help="End date YYYY-MM-DD")
    dl.add_argument("--output", default="data/raw")
    dl.set_defaults(func=cmd_download)

    # ---- generate --------------------------------------------------------
    gen = sub.add_parser("generate", help="Generate synthetic forex data")
    gen.add_argument("--symbol",  default="EURUSD")
    gen.add_argument("--samples", type=int, default=10_000)
    gen.add_argument("--output",  default="data/raw")
    gen.add_argument("--seed",    type=int, default=42)
    gen.set_defaults(func=cmd_generate)

    # ---- train -----------------------------------------------------------
    tr = sub.add_parser("train", help="Train the RL agent")
    tr.add_argument("--data",       required=True,
                    help="Directory containing SYMBOL.csv files (data/raw)")
    tr.add_argument("--symbols",    nargs="*", default=None,
                    help="Symbols to train on (default: all files in --data)")
    tr.add_argument("--config",     default="config/config.yaml")
    tr.add_argument("--model",      default=None,
                    help="Existing model to continue training from")
    tr.add_argument("--output",     default="models")
    tr.add_argument("--timesteps",  type=int, default=None,
                    help="Total training timesteps (overrides config)")
    tr.add_argument("--seed",       type=int, default=42)
    tr.set_defaults(func=cmd_train)

    # ---- evaluate --------------------------------------------------------
    ev = sub.add_parser("evaluate", help="Evaluate a trained RL model")
    ev.add_argument("--model",    required=True, help="Path to .zip model file")
    ev.add_argument("--data",     required=True, help="Path to OHLCV CSV file")
    ev.add_argument("--config",   default="config/config.yaml")
    ev.add_argument("--episodes", type=int, default=10)
    ev.add_argument("--render",       action="store_true")
    ev.add_argument("--no-save",      action="store_true")
    ev.add_argument("--output",       default="data/evaluation")
    ev.add_argument("--export-onnx",  action="store_true")
    ev.add_argument("--onnx-path",    default="models/model.onnx")
    ev.add_argument("--visualise",    action="store_true",
                    help="Show live per-step debug dashboard")
    ev.add_argument("--vis-window",   type=int,   default=120, metavar="N")
    ev.add_argument("--vis-pause",    type=float, default=0.01, metavar="SEC")
    ev.set_defaults(func=cmd_evaluate)

    # ---- baseline --------------------------------------------------------
    bl = sub.add_parser("baseline", help="Run and evaluate a hand-coded strategy")
    bl.add_argument(
        "--strategy", required=True,
        choices=["random", "ma_cross"],
        help="Strategy to run",
    )
    bl.add_argument("--data",     required=True, help="Path to OHLCV CSV file")
    bl.add_argument("--config",   default="config/config.yaml")
    bl.add_argument("--episodes", type=int, default=1,
                    help="Episodes to run (default: 1 — full dataset pass)")
    bl.add_argument("--no-save",  action="store_true",
                    help="Skip saving results to disk")
    bl.add_argument("--output",   default="data/evaluation",
                    help="Directory for saved results")
    bl.add_argument("--seed",     type=int, default=42,
                    help="Random seed (RandomStrategy only)")
    # MA cross params
    bl.add_argument("--fast",     type=int, default=10,
                    help="Fast MA period (ma_cross only, default: 10)")
    bl.add_argument("--slow",     type=int, default=50,
                    help="Slow MA period (ma_cross only, default: 50)")
    # Shared lot tier
    bl.add_argument("--lot-tier", type=int, default=0, choices=[0, 1, 2],
                    metavar="{0,1,2}",
                    help="Lot size tier: 0=0.01  1=0.02  2=0.05 (default: 0)")
    # Visualiser
    bl.add_argument("--visualise",  action="store_true",
                    help="Show live per-step debug dashboard")
    bl.add_argument("--vis-window", type=int,   default=120, metavar="N")
    bl.add_argument("--vis-pause",  type=float, default=0.01, metavar="SEC")
    bl.set_defaults(func=cmd_baseline)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    setup_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())