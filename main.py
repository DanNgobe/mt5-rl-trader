#!/usr/bin/env python3
"""
Forex RL Trader CLI

Command-line interface for the forex reinforcement learning trading system.
Provides commands for data download, training, evaluation, and visualization.

Usage:
    python main.py download --symbol EURUSD --samples 10000
    python main.py train --data data/raw/EURUSD.npy --timesteps 100000
    python main.py evaluate --model models/ppo_trading_final.zip --data data/raw/EURUSD.npy
    python main.py generate --symbol EURUSD --samples 5000
"""

import argparse
import sys
from pathlib import Path


def cmd_download(args):
    """Download historical data from MetaTrader 5."""
    try:
        from data.downloader import MT5Downloader
    except ImportError:
        print("Error: MetaTrader5 not installed. Run: pip install MetaTrader5")
        return 1
    
    downloader = MT5Downloader()
    
    try:
        if not downloader.connect():
            print("Failed to connect to MT5")
            return 1
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = downloader.download_multiple(
            symbols=args.symbols,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            output_dir=str(output_dir),
        )
        
        print(f"\nSuccessfully downloaded {len(results)} symbols")
        for symbol, df in results.items():
            print(f"  - {symbol}: {len(df)} bars")
        
        return 0
    
    finally:
        downloader.disconnect()


def cmd_generate(args):
    """Generate synthetic forex data for testing."""
    from data.generator import generate_and_save
    
    generate_and_save(
        output_dir=args.output,
        symbol=args.symbol,
        n_samples=args.samples,
        seed=args.seed,
    )
    return 0


def cmd_train(args):
    """Train the RL agent."""
    from agents.train import train
    
    train(
        config_path=args.config,
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )
    return 0


def cmd_evaluate(args):
    """Evaluate a trained model."""
    from agents.evaluate import evaluate
    
    evaluate(
        model_path=args.model,
        data_path=args.data,
        config_path=args.config,
        n_episodes=args.episodes,
        render=args.render,
        save_results=not args.no_save,
        results_dir=args.output,
    )
    return 0


def cmd_generate_data(args):
    """Generate synthetic data for testing (alias for generate)."""
    return cmd_generate(args)


def main():
    parser = argparse.ArgumentParser(
        description="Forex RL Trader - Reinforcement Learning for Forex Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s download --symbol EURUSD --start 2024-01-01 --end 2024-12-31
  %(prog)s generate --symbol EURUSD --samples 10000
  %(prog)s train --data data/raw/EURUSD.npy --timesteps 100000
  %(prog)s evaluate --model models/ppo_trading_final.zip --data data/raw/EURUSD.npy --episodes 50
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download historical data from MetaTrader 5",
    )
    download_parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD"],
        help="Forex symbols to download",
    )
    download_parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"],
        help="Timeframe",
    )
    download_parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--end",
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory",
    )
    download_parser.set_defaults(func=cmd_download)
    
    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic forex data for testing",
    )
    generate_parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Currency pair symbol",
    )
    generate_parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of data points to generate",
    )
    generate_parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    generate_parser.set_defaults(func=cmd_generate)
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train the RL agent",
    )
    train_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    train_parser.add_argument(
        "--data",
        required=True,
        help="Path to OHLCV data file",
    )
    train_parser.add_argument(
        "--model",
        default=None,
        help="Path to existing model for continued training",
    )
    train_parser.add_argument(
        "--output",
        default="models",
        help="Output directory for trained model",
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
    )
    eval_parser.add_argument(
        "--model",
        required=True,
        help="Path to trained PPO model (.zip file)",
    )
    eval_parser.add_argument(
        "--data",
        required=True,
        help="Path to OHLCV data file",
    )
    eval_parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    eval_parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )
    eval_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file",
    )
    eval_parser.add_argument(
        "--output",
        default="data/evaluation",
        help="Output directory for results",
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Generate-data alias
    gen_data_parser = subparsers.add_parser(
        "generate-data",
        help="Generate synthetic data (alias for generate)",
    )
    gen_data_parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Currency pair symbol",
    )
    gen_data_parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of data points to generate",
    )
    gen_data_parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory",
    )
    gen_data_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    gen_data_parser.set_defaults(func=cmd_generate_data)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
