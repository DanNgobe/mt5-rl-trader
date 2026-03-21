"""
Data generator for creating synthetic forex data for testing.

Generates realistic-looking OHLCV data without requiring MT5 connection.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_forex_data(
    symbol: str = "EURUSD",
    n_samples: int = 10000,
    base_price: float = 1.1000,
    volatility: float = 0.0005,
    timeframe: str = "H1",
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Generate synthetic forex OHLCV data.
    
    Uses geometric Brownian motion with mean reversion to create
    realistic-looking price data.
    
    Args:
        symbol: Currency pair symbol
        n_samples: Number of data points to generate
        base_price: Starting price
        volatility: Price volatility per bar
        timeframe: Timeframe string (H1, H4, D1)
        seed: Random seed for reproducibility
        start_date: Start date for the data
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    np.random.seed(seed)
    
    # Timeframe to timedelta
    timeframe_map = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
    }
    td = timeframe_map.get(timeframe, timedelta(hours=1))
    
    # Generate timestamps (skip weekends for forex)
    start = pd.Timestamp(start_date)
    timestamps = []
    current = start
    
    while len(timestamps) < n_samples:
        # Skip weekends
        if current.dayofweek < 5:  # Monday = 0, Friday = 4
            timestamps.append(current)
        current += td
    
    # Generate prices using geometric Brownian motion with mean reversion
    # dS = mu*S*dt + sigma*S*dW + kappa*(S_mean - S)*dt
    mu = 0.0  # No drift (forex is roughly mean-reverting)
    sigma = volatility  # Volatility parameter
    kappa = 0.001  # Mean reversion speed
    S_mean = base_price  # Long-term mean
    
    prices = [base_price]
    for i in range(1, n_samples):
        dW = np.random.randn() * np.sqrt(volatility)
        S = prices[-1]
        
        # GBM with mean reversion
        dS = mu * S * volatility + sigma * S * dW + kappa * (S_mean - S) * volatility
        new_price = S + dS
        
        # Ensure positive price
        new_price = max(0.0001, new_price)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC from prices
    # Use random walk within bar for realistic OHLC
    opens = np.roll(prices, 1)
    opens[0] = base_price
    
    # Intra-bar volatility
    intra_bar_vol = volatility * 0.5
    
    highs = []
    lows = []
    for i in range(n_samples):
        # Random high/low around open/close
        bar_range = abs(prices[i] - opens[i]) + np.abs(np.random.randn()) * intra_bar_vol * prices[i]
        highs.append(max(opens[i], prices[i]) + np.random.uniform(0, 0.5) * bar_range)
        lows.append(min(opens[i], prices[i]) - np.random.uniform(0, 0.5) * bar_range)
    
    highs = np.array(highs)
    lows = np.array(lows)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))
    
    # Generate volume (higher during certain hours for realism)
    base_volume = 1000
    volume = base_volume + np.random.exponential(base_volume, n_samples)
    
    # Add some time-based variation (higher volume during London/NY overlap)
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        # London/NY overlap: 13:00-17:00 GMT
        if 13 <= hour <= 17:
            volume[i] *= 1.5
        # Asian session: lower volume
        elif 0 <= hour <= 6:
            volume[i] *= 0.5
    
    volume = volume.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volume,
    })
    
    return df


def generate_and_save(
    output_dir: str = "data/raw",
    symbol: str = "EURUSD",
    n_samples: int = 10000,
    seed: int = 42,
) -> str:
    """
    Generate synthetic data and save to file.
    
    Args:
        output_dir: Directory to save data
        symbol: Currency pair symbol
        n_samples: Number of data points
        seed: Random seed
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print(f"Generating {n_samples} bars of {symbol} data...")
    df = generate_forex_data(
        symbol=symbol,
        n_samples=n_samples,
        seed=seed,
    )
    
    # Save as numpy array (OHLCV only, without timestamp)
    ohlcv = df[["open", "high", "low", "close", "volume"]].values
    npy_path = output_path / f"{symbol}.npy"
    np.save(npy_path, ohlcv)
    print(f"Saved numpy array to: {npy_path}")
    
    # Also save as CSV for inspection
    csv_path = output_path / f"{symbol}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Print summary
    print(f"\nData summary:")
    print(f"  Symbol: {symbol}")
    print(f"  Bars: {n_samples}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    print(f"  Avg volume: {df['volume'].mean():.0f}")
    
    return str(npy_path)


def main():
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic forex data")
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Currency pair symbol",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of data points to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--base-price",
        type=float,
        default=None,
        help="Base price (default depends on symbol)",
    )
    
    args = parser.parse_args()
    
    # Default base prices for common pairs
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.2700,
        "USDJPY": 148.00,
        "USDCHF": 0.8800,
        "AUDUSD": 0.6600,
        "USDCAD": 1.3500,
        "EURGBP": 0.8600,
        "EURJPY": 162.00,
    }
    
    base_price = args.base_price or base_prices.get(args.symbol, 1.1000)
    
    generate_and_save(
        output_dir=args.output,
        symbol=args.symbol,
        n_samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
