"""
Preprocessor for forex trading data.

Provides feature scaling, normalization, and transformation
for OHLCV data before feeding to the RL environment.
"""

from typing import Optional

import numpy as np
import pandas as pd


class PriceScaler:
    """
    Scaler for price data using log returns or relative pricing.
    
    Converts absolute prices to relative/normalized values
    to make the model robust across different price levels.
    """
    
    def __init__(self, method: str = "log_return"):
        """
        Initialize the scaler.
        
        Args:
            method: Scaling method ('log_return', 'relative', 'zscore')
        """
        self.method = method
        self.reference_price: Optional[float] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, data: np.ndarray) -> "PriceScaler":
        """
        Fit the scaler to data.
        
        Args:
            data: OHLCV data array of shape (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        if self.method == "zscore":
            # Calculate mean and std for price columns (OHLC)
            self.mean = np.mean(data[:, :4], axis=0)
            self.std = np.std(data[:, :4], axis=0) + 1e-8
        else:
            # Use first close price as reference
            self.reference_price = float(data[0, 3])  # Close is column 3
        
        self._fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: OHLCV data array
            
        Returns:
            Transformed data
        """
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        data = np.asarray(data, dtype=np.float64)
        result = data.copy()
        
        if self.method == "log_return":
            # Convert prices to log returns relative to reference
            for i in range(4):  # OHLC columns
                result[:, i] = np.log(data[:, i] / self.reference_price)
        
        elif self.method == "relative":
            # Convert prices to relative (ratio to reference)
            for i in range(4):  # OHLC columns
                result[:, i] = data[:, i] / self.reference_price - 1.0
        
        elif self.method == "zscore":
            # Standardize price columns
            result[:, :4] = (data[:, :4] - self.mean) / self.std
        
        return result.astype(np.float32)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform back to original price scale.
        
        Args:
            data: Transformed data
            
        Returns:
            Data in original price scale
        """
        if not self._fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        data = np.asarray(data, dtype=np.float64)
        result = data.copy()
        
        if self.method == "log_return":
            for i in range(4):
                result[:, i] = self.reference_price * np.exp(data[:, i])
        
        elif self.method == "relative":
            for i in range(4):
                result[:, i] = self.reference_price * (data[:, i] + 1.0)
        
        elif self.method == "zscore":
            result[:, :4] = data[:, :4] * self.std + self.mean
        
        return result.astype(np.float32)


class VolumeScaler:
    """
    Scaler for volume data using log or min-max scaling.
    """
    
    def __init__(self, method: str = "log"):
        """
        Initialize the volume scaler.
        
        Args:
            method: Scaling method ('log', 'minmax', 'zscore')
        """
        self.method = method
        self.min_vol: Optional[float] = None
        self.max_vol: Optional[float] = None
        self.mean_vol: Optional[float] = None
        self.std_vol: Optional[float] = None
        self._fitted = False
    
    def fit(self, volume: np.ndarray) -> "VolumeScaler":
        """Fit the scaler to volume data."""
        volume = np.asarray(volume).flatten()
        
        if self.method == "log":
            pass  # No fitting needed for log transform
        
        elif self.method == "minmax":
            self.min_vol = float(np.min(volume))
            self.max_vol = float(np.max(volume))
        
        elif self.method == "zscore":
            self.mean_vol = float(np.mean(volume))
            self.std_vol = float(np.std(volume)) + 1e-8
        
        self._fitted = True
        return self
    
    def transform(self, volume: np.ndarray) -> np.ndarray:
        """Transform volume data."""
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        volume = np.asarray(volume).flatten()
        
        if self.method == "log":
            # Log transform with small epsilon to handle zero volume
            result = np.log1p(volume)
        
        elif self.method == "minmax":
            range_vol = self.max_vol - self.min_vol + 1e-8
            result = (volume - self.min_vol) / range_vol
        
        elif self.method == "zscore":
            result = (volume - self.mean_vol) / self.std_vol
        
        return result.astype(np.float32)
    
    def fit_transform(self, volume: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(volume)
        return self.transform(volume)


class FeatureProcessor:
    """
    Complete feature processing pipeline for OHLCV data.
    
    Combines price scaling, volume scaling, and optional
    technical indicator calculation.
    """
    
    def __init__(
        self,
        price_method: str = "log_return",
        volume_method: str = "log",
        add_technical_indicators: bool = False,
    ):
        """
        Initialize the feature processor.
        
        Args:
            price_method: Price scaling method
            volume_method: Volume scaling method
            add_technical_indicators: Whether to add technical indicators
        """
        self.price_method = price_method
        self.volume_method = volume_method
        self.add_technical_indicators = add_technical_indicators
        
        self.price_scaler = PriceScaler(method=price_method)
        self.volume_scaler = VolumeScaler(method=volume_method)
        self._fitted = False
    
    def fit(self, data: np.ndarray) -> "FeatureProcessor":
        """
        Fit the processor to data.
        
        Args:
            data: OHLCV data array of shape (n_samples, 5) or (n_samples, 6)
                  Columns: [open, high, low, close, volume, (spread)]
        """
        data = np.asarray(data)
        
        # Fit price scaler on OHLC columns
        self.price_scaler.fit(data[:, :4])
        
        # Fit volume scaler on volume column
        self.volume_scaler.fit(data[:, 4])
        
        self._fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted processors.
        
        Args:
            data: OHLCV data array
            
        Returns:
            Transformed data with same shape
        """
        if not self._fitted:
            raise ValueError("Processor must be fitted before transform")
        
        data = np.asarray(data, dtype=np.float64)
        n_samples = data.shape[0]
        
        # Transform prices (OHLC)
        prices = self.price_scaler.transform(data[:, :4])

        # Transform volume (flatten for scaler, then reshape)
        volume = self.volume_scaler.transform(data[:, 4]).reshape(-1, 1)

        # Handle optional spread column
        if data.shape[1] >= 6:
            spread = data[:, 5:6].astype(np.float32)
            result = np.hstack([prices, volume, spread])
        else:
            result = np.hstack([prices, volume])
        
        # Add technical indicators if requested
        if self.add_technical_indicators:
            indicators = self._calculate_technical_indicators(data)
            result = np.hstack([result, indicators])
        
        return result
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def _calculate_technical_indicators(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate technical indicators from raw OHLCV data.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Array of technical indicators
        """
        indicators = []
        
        # Price range (high - low)
        price_range = (data[:, 1] - data[:, 2]) / data[:, 3]
        indicators.append(price_range)
        
        # Price momentum (close - open) / open
        momentum = (data[:, 3] - data[:, 0]) / data[:, 0]
        indicators.append(momentum)
        
        # Upper shadow (high - max(open, close))
        upper_shadow = data[:, 1] - np.maximum(data[:, 0], data[:, 3])
        upper_shadow = upper_shadow / data[:, 3]
        indicators.append(upper_shadow)
        
        # Lower shadow (min(open, close) - low)
        lower_shadow = np.minimum(data[:, 0], data[:, 3]) - data[:, 2]
        lower_shadow = lower_shadow / data[:, 3]
        indicators.append(lower_shadow)
        
        return np.column_stack(indicators).astype(np.float32)
    
    def get_feature_names(self, original_names: Optional[list[str]] = None) -> list[str]:
        """
        Get names for transformed features.
        
        Args:
            original_names: Original feature names
            
        Returns:
            List of transformed feature names
        """
        if original_names is None:
            original_names = ["open", "high", "low", "close", "volume"]
        
        names = []
        
        # Transformed price features
        for name in original_names[:4]:
            names.append(f"{name}_{self.price_method}")
        
        # Transformed volume
        names.append(f"volume_{self.volume_method}")
        
        # Spread if present
        if len(original_names) > 5:
            names.append(original_names[5])
        
        # Technical indicators
        if self.add_technical_indicators:
            names.extend(["price_range", "momentum", "upper_shadow", "lower_shadow"])
        
        return names


def create_ohlcv_dataframe(
    data: np.ndarray,
    columns: Optional[list[str]] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from OHLCV numpy array.
    
    Args:
        data: OHLCV data array
        columns: Column names (default: ['open', 'high', 'low', 'close', 'volume'])
        index: Datetime index for the DataFrame
        
    Returns:
        pandas DataFrame with OHLCV data
    """
    if columns is None:
        columns = ["open", "high", "low", "close", "volume"]
    
    return pd.DataFrame(data, columns=columns, index=index)


def normalize_ohlcv(data: np.ndarray, method: str = "log_return") -> np.ndarray:
    """
    Convenience function to normalize OHLCV data.
    
    Args:
        data: OHLCV data array
        method: Normalization method
        
    Returns:
        Normalized data
    """
    processor = FeatureProcessor(price_method=method, volume_method="log")
    return processor.fit_transform(data)
