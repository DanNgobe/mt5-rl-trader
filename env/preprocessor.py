"""
Preprocessor for multi-symbol forex OHLCV data.

Design goals
------------
- Price normalization must be pair-agnostic so a model trained on EURUSD
  generalises to USDJPY without seeing different raw price scales.
- Log returns achieve this: they express price movement as a fraction,
  independent of the absolute price level (1.08 vs 150).
- Volume is z-scored per file because liquidity profiles differ across
  symbols; normalising each file independently keeps volume features
  comparable within a symbol while still being scale-free.
- No technical indicators — the agent learns directly from price action.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected column order for all OHLCV files
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Core transforms — stateless, applied per-array
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D or 2-D price array to log returns.

    log_return[t] = log(price[t] / price[t-1])

    The first row is set to 0.0 (no predecessor).  The output is the same
    shape as the input, which keeps array indexing simple in the environment.

    This transform is completely stateless — no fitting required — and
    produces values that are directly comparable across EURUSD, USDJPY,
    GBPJPY, etc.
    """
    prices = np.asarray(prices, dtype=np.float64)

    if prices.ndim == 1:
        result        = np.empty_like(prices)
        result[0]     = 0.0
        result[1:]    = np.log(prices[1:] / prices[:-1])
        return result.astype(np.float32)

    # 2-D: apply column-wise
    result       = np.empty_like(prices)
    result[0, :] = 0.0
    result[1:, :] = np.log(prices[1:, :] / prices[:-1, :])
    return result.astype(np.float32)


def zscore_volume(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalise a volume series in-place (fit on the full array).

    Uses the population std.  A small epsilon prevents division by zero on
    flat/synthetic data.
    """
    volume = np.asarray(volume, dtype=np.float64).flatten()
    mean   = volume.mean()
    std    = volume.std() + 1e-8
    return ((volume - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def preprocess(
    data: np.ndarray,
    columns: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single OHLCV array.

    Steps
    -----
    1. Validate shape — must have at least 5 columns (OHLCV).
    2. Apply log returns to OHLC columns (columns 0–3).
    3. Z-score normalise volume (column 4).
    4. Pass through any extra columns (e.g. spread) unchanged.

    Args:
        data:    Raw OHLCV array of shape (n_samples, ≥5).
        columns: Column names for logging only (default: OHLCV_COLUMNS).

    Returns:
        Preprocessed float32 array of the same shape.
    """
    if columns is None:
        columns = OHLCV_COLUMNS

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {data.shape}")
    if data.shape[1] < 5:
        raise ValueError(
            f"Expected at least 5 columns (OHLCV), got {data.shape[1]}"
        )

    n_samples, n_cols = data.shape
    result            = np.empty_like(data, dtype=np.float32)

    # OHLC → log returns
    result[:, :4] = log_returns(data[:, :4])

    # Volume → z-score
    result[:, 4] = zscore_volume(data[:, 4])

    # Pass through any extra columns (e.g. broker spread column)
    if n_cols > 5:
        result[:, 5:] = data[:, 5:].astype(np.float32)

    logger.debug(
        "Preprocessed %d samples, %d features. "
        "OHLC → log returns; volume → z-score.",
        n_samples, n_cols,
    )
    return result


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    columns: Optional[list[str]] = None,
    datetime_col: Optional[str] = "time",
) -> tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    """
    Load a CSV file and return a raw OHLCV numpy array plus optional index.

    The CSV is expected to have columns matching `columns` (case-insensitive).
    Extra columns are dropped.  The datetime column, if present, becomes the
    returned index.

    Args:
        path:         Path to CSV file.
        columns:      Expected OHLCV column names.
        datetime_col: Name of datetime column to parse as index (or None).

    Returns:
        (data_array, datetime_index_or_None)
    """
    if columns is None:
        columns = OHLCV_COLUMNS

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Parse datetime index if available
    dt_index: Optional[pd.DatetimeIndex] = None
    if datetime_col and datetime_col.lower() in df.columns:
        dt_index = pd.to_datetime(df[datetime_col.lower()])
        df = df.drop(columns=[datetime_col.lower()])

    # Select and order required columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    data = df[columns].to_numpy(dtype=np.float64)

    logger.info("Loaded %d rows from %s", len(data), path.name)
    return data, dt_index


def load_npy(path: str | Path) -> np.ndarray:
    """Load a pre-saved .npy OHLCV array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    logger.info("Loaded %s from %s", data.shape, path.name)
    return data


def load_and_preprocess(
    path: str | Path,
    columns: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Convenience: load a CSV or .npy file and immediately preprocess it.

    This is the main entry point used by the training pipeline when
    iterating over data/raw/*.csv.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        raw = load_npy(path)
    else:
        raw, _ = load_csv(path, columns=columns)

    return preprocess(raw, columns=columns)


def load_symbol_files(
    data_dir: str | Path,
    symbols: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Load and preprocess all symbol files in a directory.

    Each file is normalised independently (per-file volume z-score).
    Price log returns are stateless so no cross-symbol state is needed.

    Args:
        data_dir: Directory containing SYMBOL.csv or SYMBOL.npy files.
        symbols:  Whitelist of symbol names (without extension).
                  If None, all .csv and .npy files are loaded.
        columns:  OHLCV column names.

    Returns:
        Dict mapping symbol name → preprocessed float32 array.
    """
    data_dir = Path(data_dir)
    results: dict[str, np.ndarray] = {}

    patterns = ["*.csv", "*.npy"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(data_dir.glob(pat)))

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    for fpath in files:
        symbol = fpath.stem.upper()
        if symbols is not None and symbol not in [s.upper() for s in symbols]:
            continue

        try:
            results[symbol] = load_and_preprocess(fpath, columns=columns)
            logger.info("Preprocessed %s → shape %s", symbol, results[symbol].shape)
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath.name, exc)

    if not results:
        raise ValueError(
            f"No symbol files could be loaded from {data_dir} "
            f"(requested: {symbols})"
        )

    return results