"""
Preprocessor for multi-symbol forex OHLCV data.

Design goals
------------
- Price normalisation is pair-agnostic via log returns.
- Technical indicators are precomputed over the full series at load time
  so the environment can index into them cheaply at each step.
- Every indicator output is normalised to a consistent scale so the MLP
  does not have to deal with mixed magnitudes.
- All indicator computation is controlled by the obs_config dict so the
  observation space can be fully configured from config.yaml.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Core price transforms
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns for a 1-D or 2-D price array. First row = 0."""
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim == 1:
        result     = np.empty_like(prices)
        result[0]  = 0.0
        result[1:] = np.log(prices[1:] / prices[:-1])
        return result.astype(np.float32)
    result        = np.empty_like(prices)
    result[0, :]  = 0.0
    result[1:, :] = np.log(prices[1:, :] / prices[:-1, :])
    return result.astype(np.float32)


def zscore_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise a volume series over the full array."""
    volume = np.asarray(volume, dtype=np.float64).flatten()
    std    = volume.std() + 1e-8
    return ((volume - volume.mean()) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Indicator builders — all return float32 arrays of shape (n,)
# Values are normalised to roughly [-1, 1] or [0, 1] ranges.
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via pandas for correctness."""
    return pd.Series(series).ewm(span=period, adjust=False).mean().to_numpy(dtype=np.float64)


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI normalised to [-1, 1].
    Raw RSI 0-100 → (RSI/50) - 1  so 50=0, overbought>0, oversold<0.
    """
    close  = np.asarray(close, dtype=np.float64)
    delta  = np.diff(close, prepend=close[0])
    gain   = np.where(delta > 0, delta, 0.0)
    loss   = np.where(delta < 0, -delta, 0.0)

    avg_gain = pd.Series(gain).ewm(com=period - 1, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(com=period - 1, adjust=False).mean().to_numpy()

    rs  = np.divide(avg_gain, avg_loss, where=avg_loss > 0, out=np.full_like(avg_loss, 100.0))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return ((rsi / 50.0) - 1.0).astype(np.float32)


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """
    ATR normalised by close price → dimensionless volatility ratio.
    Typical values ~0.001–0.01 for forex; kept as-is (already small scale).
    """
    high  = np.asarray(high,  dtype=np.float64)
    low   = np.asarray(low,   dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close),
                    np.abs(low  - prev_close)))

    atr = pd.Series(tr).ewm(com=period - 1, adjust=False).mean().to_numpy()
    return (atr / np.where(close > 0, close, 1.0)).astype(np.float32)


def compute_ema_ratio(close: np.ndarray, fast: int = 8, slow: int = 21) -> np.ndarray:
    """
    EMA(fast)/EMA(slow) - 1.
    Positive = fast above slow (uptrend), negative = downtrend.
    Clipped to [-0.1, 0.1] to bound outliers.
    """
    close     = np.asarray(close, dtype=np.float64)
    ema_fast  = _ema(close, fast)
    ema_slow  = _ema(close, slow)
    ratio     = ema_fast / np.where(ema_slow > 0, ema_slow, 1.0) - 1.0
    return np.clip(ratio, -0.1, 0.1).astype(np.float32)


def compute_bollinger_pct(close: np.ndarray, period: int = 20,
                          std_dev: float = 2.0) -> np.ndarray:
    """
    Bollinger %B: where price sits within the band.
    0 = lower band, 1 = upper band, 0.5 = midline.
    Normalised to [-1, 1]: (pct_b - 0.5) * 2.
    Clipped to [-1.5, 1.5] for breakouts.
    """
    s      = pd.Series(close.astype(np.float64))
    mid    = s.rolling(period, min_periods=1).mean()
    std    = s.rolling(period, min_periods=1).std(ddof=0).fillna(0)
    upper  = mid + std_dev * std
    lower  = mid - std_dev * std
    band   = (upper - lower).to_numpy()
    pct_b  = np.divide(close - lower.to_numpy(), band,
                       where=band > 0, out=np.full_like(band, 0.5))
    return np.clip((pct_b - 0.5) * 2.0, -1.5, 1.5).astype(np.float32)


def compute_momentum(close: np.ndarray, periods: list[int]) -> np.ndarray:
    """
    Log return over each period: log(close[t] / close[t-p]).
    Returns array of shape (n, len(periods)).
    First `p` rows for each period are 0.
    """
    close  = np.asarray(close, dtype=np.float64)
    n      = len(close)
    result = np.zeros((n, len(periods)), dtype=np.float32)
    for col, p in enumerate(periods):
        if p < n:
            result[p:, col] = np.log(close[p:] / close[:-p]).astype(np.float32)
    return result


def compute_session_features(dt_index: pd.DatetimeIndex,
                              n: int) -> np.ndarray:
    """
    Cyclical time encoding: hour_sin, hour_cos, dow_sin, dow_cos.
    Returns array of shape (n, 4).
    Falls back to zeros if no datetime index is available.
    """
    result = np.zeros((n, 4), dtype=np.float32)
    if dt_index is None or len(dt_index) != n:
        logger.warning("No datetime index — session features will be zero.")
        return result
    hours = dt_index.hour.to_numpy(dtype=np.float32)
    dows  = dt_index.dayofweek.to_numpy(dtype=np.float32)
    result[:, 0] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    result[:, 1] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    result[:, 2] = np.sin(2 * np.pi * dows  / 5).astype(np.float32)
    result[:, 3] = np.cos(2 * np.pi * dows  / 5).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Observation dimension calculator
# ---------------------------------------------------------------------------

def obs_dim_from_config(obs_cfg: dict, n_slots: int) -> int:
    """
    Compute the flat observation vector length.

    n_slots = len(lot_tiers) * 2  — passed in from TradingEnv so the
    preprocessor stays decoupled from the action space config.
    """
    ind = obs_cfg.get("indicators", {})
    dim = 0
    dim += len(obs_cfg.get("price_lags", [1, 2, 4, 8, 24]))
    if ind.get("rsi",      {}).get("enabled", True):  dim += 1
    if ind.get("atr",      {}).get("enabled", True):  dim += 1
    if ind.get("ema_ratio",{}).get("enabled", True):  dim += 1
    if ind.get("bollinger",{}).get("enabled", True):  dim += 1
    if ind.get("momentum", {}).get("enabled", True):
        dim += len(ind["momentum"].get("periods", [5, 20, 50]))
    if ind.get("session",  {}).get("enabled", True):  dim += 4
    dim += n_slots * 3   # [direction*lot_size, upnl_norm, bars_open_norm] per slot
    dim += 2             # balance_norm, equity_norm
    return dim


# ---------------------------------------------------------------------------
# Precompute all indicator arrays for one symbol
# ---------------------------------------------------------------------------

def build_obs_arrays(
    raw:       np.ndarray,
    obs_cfg:   dict,
    dt_index:  Optional[pd.DatetimeIndex] = None,
) -> dict[str, np.ndarray]:
    """
    Precompute every enabled indicator over the full price series.

    Returns a dict of named arrays all of shape (n,) or (n, k).
    The environment indexes into these at each step — no rolling
    computation happens during training.

    Args:
        raw:      Raw OHLCV array (n, ≥5), unprocessed prices.
        obs_cfg:  The 'observation' block from config.yaml.
        dt_index: Optional DatetimeIndex aligned with raw rows.

    Returns:
        Dict with keys: 'close_log_returns', and one key per enabled indicator.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n   = len(raw)
    ind = obs_cfg.get("indicators", {})

    close = raw[:, 3]
    high  = raw[:, 1]
    low   = raw[:, 2]

    arrays: dict[str, np.ndarray] = {}

    # Close log returns — always computed (needed for sparse lag indexing)
    arrays["close_log_returns"] = log_returns(close)

    if ind.get("rsi", {}).get("enabled", True):
        arrays["rsi"] = compute_rsi(close, ind["rsi"].get("period", 14))

    if ind.get("atr", {}).get("enabled", True):
        arrays["atr"] = compute_atr(high, low, close, ind["atr"].get("period", 14))

    if ind.get("ema_ratio", {}).get("enabled", True):
        cfg = ind.get("ema_ratio", {})
        arrays["ema_ratio"] = compute_ema_ratio(close,
                                                cfg.get("fast", 8),
                                                cfg.get("slow", 21))

    if ind.get("bollinger", {}).get("enabled", True):
        cfg = ind.get("bollinger", {})
        arrays["bollinger"] = compute_bollinger_pct(close,
                                                    cfg.get("period", 20),
                                                    cfg.get("std_dev", 2.0))

    if ind.get("momentum", {}).get("enabled", True):
        periods = ind["momentum"].get("periods", [5, 20, 50])
        arrays["momentum"] = compute_momentum(close, periods)

    if ind.get("session", {}).get("enabled", True):
        arrays["session"] = compute_session_features(dt_index, n)

    logger.info(
        "Built obs arrays: %s  (n=%d)",
        list(arrays.keys()), n,
    )
    return arrays


# ---------------------------------------------------------------------------
# Preprocessing pipeline (price normalisation only — unchanged)
# ---------------------------------------------------------------------------

def preprocess(
    data: np.ndarray,
    columns: Optional[list[str]] = None,
) -> np.ndarray:
    """OHLC → log returns, volume → z-score. Shape unchanged."""
    if columns is None:
        columns = OHLCV_COLUMNS
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"Expected 2-D array with ≥5 cols, got {data.shape}")
    n_cols = data.shape[1]
    result = np.empty_like(data, dtype=np.float32)
    result[:, :4] = log_returns(data[:, :4])
    result[:, 4]  = zscore_volume(data[:, 4])
    if n_cols > 5:
        result[:, 5:] = data[:, 5:].astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    columns: Optional[list[str]] = None,
    datetime_col: Optional[str] = "time",
) -> tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    """Load CSV → (raw_array, datetime_index_or_None)."""
    if columns is None:
        columns = OHLCV_COLUMNS
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    dt_index: Optional[pd.DatetimeIndex] = None
    if datetime_col and datetime_col.lower() in df.columns:
        dt_index = pd.DatetimeIndex(pd.to_datetime(df[datetime_col.lower()]))
        df = df.drop(columns=[datetime_col.lower()])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df[columns].to_numpy(dtype=np.float64), dt_index


def load_npy(path: str | Path) -> tuple[np.ndarray, None]:
    """Load .npy file → (raw_array, None). Returns a tuple to match load_csv's signature."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    logger.info("Loaded %s from %s", data.shape, path.name)
    return data, None


def load_and_preprocess(
    path: str | Path,
    columns: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[pd.DatetimeIndex]]:
    """
    Load a file and return (preprocessed_data, raw_close, dt_index).

    raw_close is kept separately so the environment can use it for
    trade execution prices and indicator precomputation.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        raw, dt_index = load_npy(path)
    else:
        raw, dt_index = load_csv(path, columns=columns)

    processed = preprocess(raw, columns=columns)
    raw_close = raw[:, 3].astype(np.float64)
    return processed, raw_close, dt_index


def load_symbol_files(
    data_dir:  str | Path,
    obs_cfg:   Optional[dict]       = None,
    symbols:   Optional[list[str]]  = None,
    columns:   Optional[list[str]]  = None,
) -> dict[str, dict]:
    """
    Load, preprocess, and build indicator arrays for all symbol files.

    Returns a dict mapping symbol → {
        'processed':  float32 array (n, features),
        'raw_close':  float64 array (n,),
        'obs_arrays': dict of precomputed indicator arrays,
        'dt_index':   DatetimeIndex or None,
    }
    """
    data_dir = Path(data_dir)
    results: dict[str, dict] = {}

    files: list[Path] = []
    for pat in ["*.csv", "*.npy"]:
        files.extend(sorted(data_dir.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    for fpath in files:
        symbol = fpath.stem.upper()
        if symbols is not None and symbol not in [s.upper() for s in symbols]:
            continue
        try:
            processed, raw_close, dt_index = load_and_preprocess(fpath, columns)
            # Reconstruct raw array for indicator computation
            if fpath.suffix.lower() == ".npy":
                raw, _ = load_npy(fpath)
            else:
                raw, _ = load_csv(fpath, columns)

            obs_arrays = build_obs_arrays(raw, obs_cfg or {}, dt_index)
            results[symbol] = {
                "processed":  processed,
                "raw_close":  raw_close,
                "obs_arrays": obs_arrays,
                "dt_index":   dt_index,
            }
            logger.info("Loaded %s → %d bars, obs_arrays: %s",
                        symbol, len(processed), list(obs_arrays.keys()))
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath.name, exc)

    if not results:
        raise ValueError(f"No symbol files loaded from {data_dir} (requested: {symbols})")
    return results
