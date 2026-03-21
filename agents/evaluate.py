"""
Evaluation script for forex RL agent.

Runs backtesting on historical data and calculates performance metrics
including Sharpe ratio, maximum drawdown, win rate, and more.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.preprocessor import FeatureProcessor
from env.trading_env import TradingEnv


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> np.ndarray:
    """Load OHLCV data from numpy file."""
    path = Path(data_path)
    
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        return data["data"] if "data" in data else data[list(data.files)[0]]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_preprocessor(model_dir: str) -> FeatureProcessor:
    """
    Load preprocessor from saved parameters.
    
    Args:
        model_dir: Directory containing preprocessor.npy
        
    Returns:
        Configured FeatureProcessor
    """
    preprocessor_path = Path(model_dir) / "preprocessor.npy"
    
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    params = np.load(preprocessor_path, allow_pickle=True).item()
    
    # Create processor and restore parameters
    processor = FeatureProcessor()
    
    if params.get("price_reference") is not None:
        processor.price_scaler.reference_price = params["price_reference"]
        processor.price_scaler._fitted = True
    
    if params.get("price_mean") is not None:
        processor.price_scaler.mean = params["price_mean"]
        processor.price_scaler.std = params.get("price_std")
        processor.price_scaler._fitted = True
    
    if params.get("volume_min") is not None:
        processor.volume_scaler.min_vol = params["volume_min"]
        processor.volume_scaler.max_vol = params.get("volume_max")
        processor.volume_scaler._fitted = True
    
    return processor


def calculate_metrics(
    trades: list[dict],
    equity_curve: np.ndarray,
    initial_balance: float,
) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        trades: List of trade dictionaries with pnl, exit_reason, etc.
        equity_curve: Array of equity values over time
        initial_balance: Starting balance
        
    Returns:
        Dictionary of performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "final_balance": initial_balance,
            "total_return_pct": 0.0,
            "calmar_ratio": 0.0,
            "expectancy": 0.0,
        }
    
    pnls = np.array([t["pnl"] for t in trades])
    winning_pnls = pnls[pnls > 0]
    losing_pnls = pnls[pnls < 0]
    
    n_wins = len(winning_pnls)
    n_losses = len(losing_pnls)
    n_trades = len(trades)
    
    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    avg_win = float(np.mean(winning_pnls)) if n_wins > 0 else 0.0
    avg_loss = float(np.mean(losing_pnls)) if n_losses > 0 else 0.0
    
    gross_profit = float(np.sum(winning_pnls))
    gross_loss = abs(float(np.sum(losing_pnls)))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Equity curve metrics
    final_balance = float(equity_curve[-1]) if len(equity_curve) > 0 else initial_balance
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Maximum drawdown
    cumulative = np.cumsum(pnls) + initial_balance
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns))
    max_drawdown_pct = (max_drawdown / np.max(running_max)) * 100 if np.max(running_max) > 0 else 0.0
    
    # Sharpe ratio (assuming daily returns, annualized)
    if len(pnls) > 1:
        returns = np.diff(cumulative) / cumulative[:-1]
        sharpe_ratio = float(np.sqrt(252) * np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0
    
    # Expectancy (average profit per trade normalized by average loss)
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    loss_rate = n_losses / n_trades if n_trades > 0 else 0.0
    expectancy = (win_rate * avg_win / abs(avg_loss) - loss_rate) if avg_loss != 0 else 0.0
    
    return {
        "total_trades": n_trades,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "calmar_ratio": calmar_ratio,
        "expectancy": expectancy,
    }


def evaluate(
    model_path: str,
    data_path: str,
    config_path: str = "config/config.yaml",
    n_episodes: int = 10,
    render: bool = False,
    save_results: bool = True,
    results_dir: str = "data/evaluation",
) -> dict:
    """
    Evaluate trained model on historical data.
    
    Args:
        model_path: Path to trained PPO model (.zip file)
        data_path: Path to OHLCV data file
        config_path: Path to configuration file
        n_episodes: Number of evaluation episodes
        render: Whether to render environment during evaluation
        save_results: Whether to save results to file
        results_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results and metrics
    """
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    print(f"Loading data from {data_path}")
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Try to load preprocessor
    model_dir = Path(model_path).parent
    try:
        processor = load_preprocessor(str(model_dir))
        print("Loaded preprocessor from model directory")
        processed_data = processor.transform(data)
    except FileNotFoundError:
        print("Preprocessor not found, applying fresh preprocessing...")
        preproc_config = config["preprocessing"]
        processor = FeatureProcessor(
            price_method=preproc_config["price_method"],
            volume_method=preproc_config["volume_method"],
            add_technical_indicators=preproc_config["add_technical_indicators"],
        )
        processed_data = processor.fit_transform(data)
    
    print(f"Processed data shape: {processed_data.shape}")
    
    # Create environment
    env_config = config["environment"]
    env = TradingEnv(
        data=processed_data,
        window_size=env_config["window_size"],
        initial_balance=env_config["initial_balance"],
        spread=env_config["spread"],
        slippage_prob=env_config["slippage_prob"],
        slippage_range=tuple(env_config["slippage_range"]),
        render_mode="human" if render else None,
    )
    env = DummyVecEnv([lambda: env])
    
    # Set model environment
    model.set_env(env)
    
    # Run evaluation episodes
    print(f"\nRunning {n_episodes} evaluation episodes...")
    print("-" * 50)
    
    all_trades = []
    all_equity_curves = []
    episode_stats = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_trades = []
        equity_curve = [env.envs[0].equity]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Collect trade info if a trade was closed
            if info[0].get("closed_trade") is not None:
                trade = info[0]["closed_trade"]
                episode_trades.append({
                    "pnl": float(trade.pnl),
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price),
                    "size": float(trade.size),
                    "exit_reason": trade.exit_reason,
                    "slippage": float(trade.slippage),
                    "spread": float(trade.spread),
                })
            
            equity_curve.append(env.envs[0].equity)
        
        # Get episode statistics
        stats = env.envs[0].get_episode_stats()
        
        all_trades.extend(episode_trades)
        all_equity_curves.append(np.array(equity_curve))
        episode_stats.append(stats)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Trades={stats['total_trades']}, "
              f"P&L=${stats['total_pnl']:.2f}, "
              f"Win Rate={stats['win_rate']*100:.1f}%")
    
    # Calculate aggregate metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    metrics = calculate_metrics(
        all_trades,
        all_equity_curves[-1],  # Use last episode's equity curve
        env_config["initial_balance"],
    )
    
    # Print metrics
    print(f"\nTotal Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"\nTotal P&L: ${metrics['total_pnl']:.2f}")
    print(f"Average P&L per Trade: ${metrics['avg_pnl']:.2f}")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"\nFinal Balance: ${metrics['final_balance']:.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Expectancy: {metrics['expectancy']:.2f}")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = results_path / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Save trades as CSV
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_file = results_path / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"Trades saved to: {trades_file}")
        
        # Save equity curve
        equity_file = results_path / f"equity_curve_{timestamp}.npy"
        np.save(equity_file, all_equity_curves[-1])
        print(f"Equity curve saved to: {equity_file}")
        
        # Save full results summary
        summary = {
            "timestamp": timestamp,
            "model_path": str(model_path),
            "data_path": str(data_path),
            "n_episodes": n_episodes,
            "metrics": metrics,
            "episode_stats": [
                {
                    "episode": i + 1,
                    "total_trades": s["total_trades"],
                    "total_pnl": s["total_pnl"],
                    "win_rate": s["win_rate"],
                    "max_drawdown": s["max_drawdown"],
                }
                for i, s in enumerate(episode_stats)
            ],
        }
        summary_file = results_path / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
    
    return {
        "metrics": metrics,
        "trades": all_trades,
        "equity_curves": all_equity_curves,
        "episode_stats": episode_stats,
    }


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate forex RL agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to OHLCV data file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        data_path=args.data,
        config_path=args.config,
        n_episodes=args.episodes,
        render=args.render,
        save_results=not args.no_save,
        results_dir=args.output,
    )


if __name__ == "__main__":
    main()
