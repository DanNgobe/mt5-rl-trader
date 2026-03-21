"""
Unit tests for the preprocessor module.
"""

import numpy as np
import pytest

from env.preprocessor import (
    FeatureProcessor,
    PriceScaler,
    VolumeScaler,
    normalize_ohlcv,
)


class TestPriceScaler:
    """Tests for PriceScaler class."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
        ])
    
    def test_log_return_transform(self):
        """Test log return transformation."""
        scaler = PriceScaler(method="log_return")
        transformed = scaler.fit_transform(self.sample_data[:, :4])
        
        # First row should be close to 0 (reference price)
        assert np.allclose(transformed[0, :], 0.0, atol=1e-6)
        
        # Subsequent rows should be positive (prices increased)
        assert np.all(transformed[1:, :] > 0)
    
    def test_relative_transform(self):
        """Test relative transformation."""
        scaler = PriceScaler(method="relative")
        transformed = scaler.fit_transform(self.sample_data[:, :4])
        
        # First row should be close to 0
        assert np.allclose(transformed[0, :], 0.0, atol=1e-6)
    
    def test_zscore_transform(self):
        """Test z-score transformation."""
        scaler = PriceScaler(method="zscore")
        transformed = scaler.fit_transform(self.sample_data[:, :4])
        
        # Mean should be close to 0
        assert np.allclose(np.mean(transformed, axis=0), 0.0, atol=1e-6)
    
    def test_inverse_transform_log_return(self):
        """Test inverse transform for log return method."""
        scaler = PriceScaler(method="log_return")
        transformed = scaler.fit_transform(self.sample_data[:, :4])
        inverse = scaler.inverse_transform(transformed)
        
        # Should recover original prices (approximately)
        assert np.allclose(inverse, self.sample_data[:, :4], rtol=1e-5)
    
    def test_inverse_transform_relative(self):
        """Test inverse transform for relative method."""
        scaler = PriceScaler(method="relative")
        transformed = scaler.fit_transform(self.sample_data[:, :4])
        inverse = scaler.inverse_transform(transformed)
        
        assert np.allclose(inverse, self.sample_data[:, :4], rtol=1e-5)
    
    def test_fit_transform_separate(self):
        """Test that fit + transform gives same result as fit_transform."""
        scaler1 = PriceScaler(method="log_return")
        result1 = scaler1.fit_transform(self.sample_data[:, :4])
        
        scaler2 = PriceScaler(method="log_return")
        scaler2.fit(self.sample_data[:, :4])
        result2 = scaler2.transform(self.sample_data[:, :4])
        
        assert np.array_equal(result1, result2)
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        scaler = PriceScaler(method="log_return")
        
        with pytest.raises(ValueError):
            scaler.transform(self.sample_data[:, :4])


class TestVolumeScaler:
    """Tests for VolumeScaler class."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_volume = np.array([1000, 1500, 2000, 2500, 3000])
    
    def test_log_transform(self):
        """Test log transformation."""
        scaler = VolumeScaler(method="log")
        transformed = scaler.fit_transform(self.sample_volume)
        
        # Log transform should be monotonic
        assert np.all(np.diff(transformed) > 0)
        
        # Values should be log1p transformed
        expected = np.log1p(self.sample_volume)
        assert np.allclose(transformed, expected)
    
    def test_minmax_transform(self):
        """Test min-max transformation."""
        scaler = VolumeScaler(method="minmax")
        transformed = scaler.fit_transform(self.sample_volume)
        
        # Min should be 0, max should be 1
        assert np.isclose(np.min(transformed), 0.0)
        assert np.isclose(np.max(transformed), 1.0)
    
    def test_zscore_transform(self):
        """Test z-score transformation."""
        scaler = VolumeScaler(method="zscore")
        transformed = scaler.fit_transform(self.sample_volume)
        
        # Mean should be 0, std should be 1
        assert np.isclose(np.mean(transformed), 0.0, atol=1e-6)
        assert np.isclose(np.std(transformed), 1.0, atol=1e-6)


class TestFeatureProcessor:
    """Tests for FeatureProcessor class."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
        ])
    
    def test_basic_transform(self):
        """Test basic transformation without technical indicators."""
        processor = FeatureProcessor(
            price_method="log_return",
            volume_method="log",
            add_technical_indicators=False,
        )
        transformed = processor.fit_transform(self.sample_data)
        
        # Should have 5 columns (4 prices + 1 volume)
        assert transformed.shape == (5, 5)
    
    def test_transform_with_technical_indicators(self):
        """Test transformation with technical indicators."""
        processor = FeatureProcessor(
            price_method="log_return",
            volume_method="log",
            add_technical_indicators=True,
        )
        transformed = processor.fit_transform(self.sample_data)
        
        # Should have 9 columns (4 prices + 1 volume + 4 indicators)
        assert transformed.shape == (5, 9)
    
    def test_transform_with_spread_column(self):
        """Test transformation with spread column."""
        data_with_spread = np.hstack([
            self.sample_data,
            np.array([[0.0001]] * 5)
        ])
        
        processor = FeatureProcessor(
            price_method="log_return",
            volume_method="log",
            add_technical_indicators=False,
        )
        transformed = processor.fit_transform(data_with_spread)
        
        # Should have 6 columns (4 prices + 1 volume + 1 spread)
        assert transformed.shape == (5, 6)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        processor = FeatureProcessor(
            price_method="log_return",
            volume_method="log",
            add_technical_indicators=True,
        )
        processor.fit(self.sample_data)
        
        names = processor.get_feature_names()
        
        expected_names = [
            "open_log_return",
            "high_log_return",
            "low_log_return",
            "close_log_return",
            "volume_log",
            "price_range",
            "momentum",
            "upper_shadow",
            "lower_shadow",
        ]
        
        assert names == expected_names
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        processor = FeatureProcessor()
        
        with pytest.raises(ValueError):
            processor.transform(self.sample_data)


class TestNormalizeOhlcv:
    """Tests for normalize_ohlcv convenience function."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
        ])
    
    def test_normalize_default(self):
        """Test normalize with default method."""
        result = normalize_ohlcv(self.sample_data)
        
        assert result.shape == self.sample_data.shape
        # First row prices should be close to 0
        assert np.allclose(result[0, :4], 0.0, atol=1e-6)
    
    def test_normalize_relative(self):
        """Test normalize with relative method."""
        result = normalize_ohlcv(self.sample_data, method="relative")
        
        assert result.shape == self.sample_data.shape


class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""
    
    def setup_method(self):
        """Set up test data with known values."""
        # Simple data for easy verification
        self.sample_data = np.array([
            [1.0, 1.2, 0.8, 1.0, 1000],  # Large range candle
            [1.0, 1.1, 0.9, 1.05, 1100],  # Smaller range
            [1.05, 1.15, 1.0, 1.1, 1200],  # Bullish
        ])
    
    def test_price_range(self):
        """Test price range calculation."""
        processor = FeatureProcessor(add_technical_indicators=True)
        transformed = processor.fit_transform(self.sample_data)
        
        # Price range = (high - low) / close
        expected_range_0 = (1.2 - 0.8) / 1.0  # 0.4
        assert np.isclose(transformed[0, -4], expected_range_0, rtol=0.1)
    
    def test_momentum(self):
        """Test momentum calculation."""
        processor = FeatureProcessor(add_technical_indicators=True)
        transformed = processor.fit_transform(self.sample_data)
        
        # Momentum = (close - open) / open
        expected_momentum_0 = (1.0 - 1.0) / 1.0  # 0.0
        assert np.isclose(transformed[0, -3], expected_momentum_0, rtol=0.1)
        
        expected_momentum_2 = (1.1 - 1.05) / 1.05  # ~0.0476
        assert np.isclose(transformed[2, -3], expected_momentum_2, rtol=0.1)
