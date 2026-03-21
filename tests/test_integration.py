"""
End-to-end integration tests for the forex RL trading system.

Tests the complete pipeline from data generation through training to evaluation.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from agents.evaluate import evaluate, load_config
from agents.train import train, load_data
from env.preprocessor import FeatureProcessor, normalize_ohlcv
from env.trading_env import TradingEnv
from env.simulator import TradeSimulator, PositionType


def generate_synthetic_data(
    n_samples: int = 1000,
    base_price: float = 1.1000,
    volatility: float = 0.001,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic OHLCV data for testing.
    
    Args:
        n_samples: Number of data points
        base_price: Starting price
        volatility: Price volatility
        seed: Random seed
        
    Returns:
        OHLCV array of shape (n_samples, 5)
    """
    np.random.seed(seed)
    
    # Generate returns and prices
    returns = np.random.randn(n_samples) * volatility
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close prices
    opens = np.roll(prices, 1)
    opens[0] = base_price
    
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.randn(n_samples) * volatility))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.randn(n_samples) * volatility))
    
    # Generate volume
    volume = np.random.randint(500, 2000, n_samples)
    
    # Stack into OHLCV array
    data = np.column_stack([opens, highs, lows, prices, volume])
    
    return data.astype(np.float64)


class TestEndToEndPipeline:
    """End-to-end tests for the complete trading pipeline."""
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(n_samples=100, base_price=1.5000)
        
        assert data.shape == (100, 5)
        assert data.dtype == np.float64
        
        # Check price is around base
        assert np.isclose(np.mean(data[:, 3]), 1.5000, rtol=0.1)
        
        # Check high >= low
        assert np.all(data[:, 1] >= data[:, 2])
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        data = generate_synthetic_data(n_samples=100)
        
        processor = FeatureProcessor(
            price_method="log_return",
            volume_method="log",
            add_technical_indicators=True,
        )
        
        processed = processor.fit_transform(data)
        
        # Should have 9 features (4 prices + 1 volume + 4 indicators)
        assert processed.shape == (100, 9)
        
        # First row prices should be close to 0 (log return of reference)
        assert np.allclose(processed[0, :4], 0.0, atol=1e-6)
    
    def test_environment_step(self):
        """Test environment can run a full episode."""
        data = generate_synthetic_data(n_samples=100)
        
        processor = FeatureProcessor()
        processed = processor.fit_transform(data)
        
        env = TradingEnv(
            processed,
            window_size=10,
            initial_balance=10000.0,
            spread=0.0001,
            slippage_prob=0.0,
        )
        
        obs, _ = env.reset()
        
        # Run random actions until episode ends
        steps = 0
        max_steps = 1000
        while steps < max_steps:
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated or truncated:
                break
        
        # Should have completed some steps
        assert steps > 0
        
        # Should have valid observation
        assert obs.shape == env.observation_space.shape
    
    def test_environment_with_profitable_trade(self):
        """Test environment correctly handles a profitable trade."""
        # Create data with clear upward trend
        n_samples = 50
        base_price = 1.1000
        prices = base_price * (1 + np.arange(n_samples) * 0.001)  # Steady increase
        
        data = np.column_stack([
            prices,  # open
            prices * 1.0005,  # high
            prices * 0.9995,  # low
            prices,  # close
            np.ones(n_samples) * 1000,  # volume
        ])
        
        processor = FeatureProcessor(price_method="relative")
        processed = processor.fit_transform(data)
        
        env = TradingEnv(
            processed,
            window_size=5,
            initial_balance=10000.0,
            spread=0.0,
            slippage_prob=0.0,
        )
        
        env.reset()
        
        # Buy and hold
        env.step(1)  # BUY
        
        # Hold through the uptrend
        for _ in range(20):
            env.step(0)  # HOLD
        
        # Check unrealized profit
        if env.simulator.has_position:
            pnl = env.simulator.position.unrealized_pnl(env._get_current_price())
            # Should have positive unrealized P&L in uptrend
            assert pnl > 0
    
    def test_training_pipeline(self):
        """Test training pipeline with synthetic data."""
        # Generate test data
        data = generate_synthetic_data(n_samples=200)
        
        # Create temporary directory for test artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save data
            data_path = tmpdir / "test_data.npy"
            np.save(data_path, data)
            
            # Create minimal config
            config = {
                "environment": {
                    "window_size": 10,
                    "initial_balance": 10000.0,
                    "spread": 0.0001,
                    "slippage_prob": 0.0,
                    "slippage_range": [0.00001, 0.0001],
                },
                "data": {
                    "raw_data_dir": str(tmpdir),
                    "processed_data_dir": str(tmpdir / "processed"),
                    "columns": ["open", "high", "low", "close", "volume"],
                },
                "preprocessing": {
                    "price_method": "log_return",
                    "volume_method": "log",
                    "add_technical_indicators": False,
                },
                "agent": {
                    "algorithm": "PPO",
                    "policy": "MlpPolicy",
                    "net_arch": [64, 64],
                    "learning_rate": 0.001,
                    "n_steps": 128,
                    "batch_size": 32,
                    "n_epochs": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                "training": {
                    "total_timesteps": 256,  # Very short for testing
                    "seed": 42,
                    "log_dir": str(tmpdir / "logs"),
                    "model_dir": str(tmpdir / "models"),
                    "save_freq": 100000,
                    "eval_freq": 0,
                    "n_eval_episodes": 1,
                    "verbose": 0,
                },
            }
            
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            # Run training
            model = train(
                config_path=str(config_path),
                data_path=str(data_path),
                output_dir=str(tmpdir / "models"),
                total_timesteps=256,
                seed=42,
            )
            
            # Verify model was created
            assert model is not None
            
            # Verify model directory exists
            model_dir = tmpdir / "models"
            assert model_dir.exists()
            
            # Check for saved model
            model_files = list(model_dir.glob("*.zip"))
            assert len(model_files) > 0
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline with trained model."""
        # Generate test data
        data = generate_synthetic_data(n_samples=200)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save data
            data_path = tmpdir / "test_data.npy"
            np.save(data_path, data)
            
            # Create config
            config = {
                "environment": {
                    "window_size": 10,
                    "initial_balance": 10000.0,
                    "spread": 0.0001,
                    "slippage_prob": 0.0,
                    "slippage_range": [0.00001, 0.0001],
                },
                "data": {
                    "raw_data_dir": str(tmpdir),
                    "processed_data_dir": str(tmpdir / "processed"),
                    "columns": ["open", "high", "low", "close", "volume"],
                },
                "preprocessing": {
                    "price_method": "log_return",
                    "volume_method": "log",
                    "add_technical_indicators": False,
                },
                "agent": {
                    "algorithm": "PPO",
                    "policy": "MlpPolicy",
                    "net_arch": [64, 64],
                    "learning_rate": 0.001,
                    "n_steps": 128,
                    "batch_size": 32,
                    "n_epochs": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                "training": {
                    "total_timesteps": 256,
                    "seed": 42,
                    "log_dir": str(tmpdir / "logs"),
                    "model_dir": str(tmpdir / "models"),
                    "save_freq": 100000,
                    "eval_freq": 0,
                    "n_eval_episodes": 1,
                    "verbose": 0,
                },
                "evaluation": {
                    "n_episodes": 2,
                    "render": False,
                    "save_results": True,
                    "results_dir": str(tmpdir / "results"),
                },
            }
            
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            # Train model
            model = train(
                config_path=str(config_path),
                data_path=str(data_path),
                output_dir=str(tmpdir / "models"),
                total_timesteps=256,
                seed=42,
            )
            
            # Find saved model
            model_files = list((tmpdir / "models").glob("*.zip"))
            model_path = model_files[0]
            
            # Run evaluation
            results = evaluate(
                model_path=str(model_path),
                data_path=str(data_path),
                config_path=str(config_path),
                n_episodes=2,
                render=False,
                save_results=True,
                results_dir=str(tmpdir / "results"),
            )
            
            # Verify results
            assert "metrics" in results
            assert "trades" in results
            assert "equity_curves" in results
            
            metrics = results["metrics"]
            assert "total_trades" in metrics
            assert "win_rate" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
            
            # Verify results were saved
            results_dir = tmpdir / "results"
            assert results_dir.exists()
            
            # Check for saved files
            json_files = list(results_dir.glob("*.json"))
            assert len(json_files) > 0
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline: data -> preprocess -> train -> evaluate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Step 1: Generate data
            print("Generating synthetic data...")
            data = generate_synthetic_data(n_samples=300, seed=42)
            data_path = tmpdir / "data.npy"
            np.save(data_path, data)
            
            # Step 2: Preprocess
            print("Preprocessing data...")
            processor = FeatureProcessor(
                price_method="log_return",
                volume_method="log",
                add_technical_indicators=False,
            )
            processed = processor.fit_transform(data)
            processed_path = tmpdir / "processed.npy"
            np.save(processed_path, processed)
            
            # Step 3: Create config
            print("Creating configuration...")
            config = {
                "environment": {
                    "window_size": 10,
                    "initial_balance": 10000.0,
                    "spread": 0.0,
                    "slippage_prob": 0.0,
                    "slippage_range": [0.0, 0.0],
                },
                "data": {
                    "raw_data_dir": str(tmpdir),
                    "processed_data_dir": str(tmpdir),
                    "columns": ["open", "high", "low", "close", "volume"],
                },
                "preprocessing": {
                    "price_method": "log_return",
                    "volume_method": "log",
                    "add_technical_indicators": False,
                },
                "agent": {
                    "algorithm": "PPO",
                    "policy": "MlpPolicy",
                    "net_arch": [32, 32],
                    "learning_rate": 0.001,
                    "n_steps": 64,
                    "batch_size": 16,
                    "n_epochs": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                "training": {
                    "total_timesteps": 128,
                    "seed": 42,
                    "log_dir": str(tmpdir / "logs"),
                    "model_dir": str(tmpdir / "models"),
                    "save_freq": 100000,
                    "eval_freq": 0,
                    "n_eval_episodes": 1,
                    "verbose": 0,
                },
                "evaluation": {
                    "n_episodes": 2,
                    "render": False,
                    "save_results": True,
                    "results_dir": str(tmpdir / "results"),
                },
            }
            
            config_path = tmpdir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            # Step 4: Train
            print("Training model...")
            model = train(
                config_path=str(config_path),
                data_path=str(data_path),
                output_dir=str(tmpdir / "models"),
                total_timesteps=128,
                seed=42,
            )
            
            assert model is not None
            print("Training complete!")
            
            # Step 5: Evaluate
            print("Evaluating model...")
            model_files = list((tmpdir / "models").glob("*.zip"))
            model_path = model_files[0]
            
            results = evaluate(
                model_path=str(model_path),
                data_path=str(data_path),
                config_path=str(config_path),
                n_episodes=2,
                render=False,
                save_results=False,
                results_dir=str(tmpdir / "results"),
            )
            
            print("Evaluation complete!")
            
            # Verify pipeline produced expected outputs
            assert results is not None
            assert "metrics" in results
            print(f"Final metrics: {results['metrics']}")
            
            print("\n✓ Full pipeline integration test PASSED!")


def run_quick_test():
    """Run a quick sanity check of the entire system."""
    print("=" * 60)
    print("RUNNING QUICK SYSTEM SANITY CHECK")
    print("=" * 60)
    
    # Generate small dataset
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(n_samples=100)
    print(f"   Data shape: {data.shape}")
    print(f"   Price range: {data[:, 3].min():.5f} - {data[:, 3].max():.5f}")
    
    # Preprocess
    print("\n2. Preprocessing data...")
    processor = FeatureProcessor()
    processed = processor.fit_transform(data)
    print(f"   Processed shape: {processed.shape}")
    
    # Create and run environment
    print("\n3. Running trading environment...")
    env = TradingEnv(processed, window_size=10, slippage_prob=0.0)
    obs, _ = env.reset()
    print(f"   Observation shape: {obs.shape}")
    
    # Run episode
    total_reward = 0
    n_steps = 0
    for _ in range(50):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            break
    
    print(f"   Steps completed: {n_steps}")
    print(f"   Total reward: {total_reward:.4f}")
    
    stats = env.get_episode_stats()
    print(f"   Trades: {stats['total_trades']}")
    print(f"   P&L: ${stats['total_pnl']:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ QUICK SANITY CHECK PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_quick_test()
