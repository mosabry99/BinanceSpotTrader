#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Indicators Module

This module provides comprehensive technical analysis indicators for cryptocurrency trading.
It includes implementations of common indicators such as RSI, MACD, Bollinger Bands,
EMA crossovers, volume analysis, and more. Each indicator includes signal generation
logic and proper documentation.

The module is designed to work with pandas DataFrames containing OHLCV data
(Open, High, Low, Close, Volume) and can be used for both real-time analysis
and backtesting.

Key features:
- Standard technical indicators with customizable parameters
- Signal generation logic for trading decisions
- Multi-layer confirmation systems
- Visualization helpers for charting
- Comprehensive documentation and examples
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import matplotlib.pyplot as plt
from functools import wraps
import logging
import warnings

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Suppress common warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")

# Signal types
SIGNAL_BUY = 1
SIGNAL_SELL = -1
SIGNAL_NEUTRAL = 0
SIGNAL_STRONG_BUY = 2
SIGNAL_STRONG_SELL = -2

# Signal strength mapping
SIGNAL_MAPPING = {
    SIGNAL_STRONG_BUY: "Strong Buy",
    SIGNAL_BUY: "Buy",
    SIGNAL_NEUTRAL: "Neutral",
    SIGNAL_SELL: "Sell",
    SIGNAL_STRONG_SELL: "Strong Sell"
}

# Helper decorator for handling NaN values
def handle_nan_values(func):
    """Decorator to handle NaN values in indicator calculations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return result.fillna(method='ffill').fillna(method='bfill')
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            if isinstance(args[0], pd.DataFrame):
                return args[0].copy()
            return None
    return wrapper


class TechnicalIndicators:
    """
    Technical Indicators class for cryptocurrency trading analysis.
    
    This class provides methods to calculate various technical indicators
    and generate trading signals based on those indicators.
    """
    
    def __init__(self, use_talib: bool = False):
        """
        Initialize the TechnicalIndicators class.
        
        Args:
            use_talib: Whether to use TA-Lib for calculations when available (default: False)
        """
        self.use_talib = use_talib
        
        # Try to import TA-Lib if requested
        if use_talib:
            try:
                import talib
                self.talib = talib
                logger.info("Using TA-Lib for indicator calculations")
            except ImportError:
                logger.warning("TA-Lib requested but not available. Using pandas implementations.")
                self.use_talib = False
                self.talib = None
    
    @handle_nan_values
    def add_moving_averages(
        self, 
        df: pd.DataFrame,
        periods: List[int] = [9, 20, 50, 100, 200],
        ma_type: str = 'sma',
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add moving averages to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for moving averages
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            column: Column to calculate moving average on
            
        Returns:
            DataFrame with added moving averages
        """
        df = df.copy()
        
        for period in periods:
            if ma_type.lower() == 'sma':
                df[f'sma_{period}'] = df[column].rolling(window=period).mean()
            elif ma_type.lower() == 'ema':
                df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
            elif ma_type.lower() == 'wma':
                weights = np.arange(1, period + 1)
                df[f'wma_{period}'] = df[column].rolling(period).apply(
                    lambda x: np.sum(weights * x) / weights.sum(), raw=True
                )
        
        return df
    
    @handle_nan_values
    def add_rsi(
        self, 
        df: pd.DataFrame, 
        period: int = 14,
        column: str = 'close',
        overbought: float = 70.0,
        oversold: float = 30.0
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            column: Column to calculate RSI on
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            DataFrame with added RSI
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            df['rsi'] = self.talib.RSI(df[column].values, timeperiod=period)
        else:
            # Calculate RSI using pandas
            delta = df[column].diff()
            
            gain = delta.copy()
            gain[gain < 0] = 0
            
            loss = delta.copy()
            loss[loss > 0] = 0
            loss = abs(loss)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # Add RSI signal columns
        df['rsi_overbought'] = df['rsi'] > overbought
        df['rsi_oversold'] = df['rsi'] < oversold
        
        # Generate RSI signals
        df['rsi_signal'] = SIGNAL_NEUTRAL
        df.loc[df['rsi_oversold'], 'rsi_signal'] = SIGNAL_BUY
        df.loc[df['rsi_overbought'], 'rsi_signal'] = SIGNAL_SELL
        
        # Strong signals when RSI is extremely overbought/oversold
        df.loc[df['rsi'] < oversold - 10, 'rsi_signal'] = SIGNAL_STRONG_BUY
        df.loc[df['rsi'] > overbought + 10, 'rsi_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_macd(
        self, 
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            column: Column to calculate MACD on
            
        Returns:
            DataFrame with added MACD
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            macd, signal, hist = self.talib.MACD(
                df[column].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
        else:
            # Calculate MACD using pandas
            ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
            
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Generate MACD signals
        df['macd_crossover'] = SIGNAL_NEUTRAL
        
        # Detect crossovers
        df.loc[(df['macd'] > df['macd_signal']) & 
               (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 
               'macd_crossover'] = SIGNAL_BUY
        
        df.loc[(df['macd'] < df['macd_signal']) & 
               (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 
               'macd_crossover'] = SIGNAL_SELL
        
        # Strong signals when histogram is large
        hist_threshold = df['macd_hist'].abs().mean() * 2
        
        df.loc[(df['macd_crossover'] == SIGNAL_BUY) & 
               (df['macd_hist'] > hist_threshold), 
               'macd_crossover'] = SIGNAL_STRONG_BUY
        
        df.loc[(df['macd_crossover'] == SIGNAL_SELL) & 
               (df['macd_hist'] < -hist_threshold), 
               'macd_crossover'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_bollinger_bands(
        self, 
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            std_dev: Standard deviation multiplier
            column: Column to calculate Bollinger Bands on
            
        Returns:
            DataFrame with added Bollinger Bands
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            upper, middle, lower = self.talib.BBANDS(
                df[column].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0  # Simple Moving Average
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        else:
            # Calculate Bollinger Bands using pandas
            df['bb_middle'] = df[column].rolling(window=period).mean()
            rolling_std = df[column].rolling(window=period).std()
            
            df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent_b'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Generate Bollinger Band signals
        df['bb_signal'] = SIGNAL_NEUTRAL
        
        # Price crossing below lower band (potential buy)
        df.loc[(df[column] < df['bb_lower']) & 
               (df[column].shift(1) >= df['bb_lower'].shift(1)), 
               'bb_signal'] = SIGNAL_BUY
        
        # Price crossing above upper band (potential sell)
        df.loc[(df[column] > df['bb_upper']) & 
               (df[column].shift(1) <= df['bb_upper'].shift(1)), 
               'bb_signal'] = SIGNAL_SELL
        
        # Strong signals when price is far outside the bands
        df.loc[(df[column] < df['bb_lower'] * 0.99), 'bb_signal'] = SIGNAL_STRONG_BUY
        df.loc[(df[column] > df['bb_upper'] * 1.01), 'bb_signal'] = SIGNAL_STRONG_SELL
        
        # Detect potential volatility expansion (bandwidth increasing)
        df['bb_bandwidth_increasing'] = df['bb_bandwidth'] > df['bb_bandwidth'].shift(1)
        
        return df
    
    @handle_nan_values
    def add_ema_crossover(
        self, 
        df: pd.DataFrame,
        fast_period: int = 9,
        slow_period: int = 21,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Add EMA Crossover signals to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            column: Column to calculate EMAs on
            
        Returns:
            DataFrame with added EMA crossover signals
        """
        df = df.copy()
        
        # Calculate EMAs
        df[f'ema_{fast_period}'] = df[column].ewm(span=fast_period, adjust=False).mean()
        df[f'ema_{slow_period}'] = df[column].ewm(span=slow_period, adjust=False).mean()
        
        # Generate EMA crossover signals
        df['ema_crossover'] = SIGNAL_NEUTRAL
        
        # Detect crossovers
        df.loc[(df[f'ema_{fast_period}'] > df[f'ema_{slow_period}']) & 
               (df[f'ema_{fast_period}'].shift(1) <= df[f'ema_{slow_period}'].shift(1)), 
               'ema_crossover'] = SIGNAL_BUY
        
        df.loc[(df[f'ema_{fast_period}'] < df[f'ema_{slow_period}']) & 
               (df[f'ema_{fast_period}'].shift(1) >= df[f'ema_{slow_period}'].shift(1)), 
               'ema_crossover'] = SIGNAL_SELL
        
        # Strong signals when crossover happens with significant distance
        threshold = df[column].rolling(window=slow_period).std() * 0.5
        
        df.loc[(df['ema_crossover'] == SIGNAL_BUY) & 
               (df[f'ema_{fast_period}'] - df[f'ema_{slow_period}'] > threshold), 
               'ema_crossover'] = SIGNAL_STRONG_BUY
        
        df.loc[(df['ema_crossover'] == SIGNAL_SELL) & 
               (df[f'ema_{slow_period}'] - df[f'ema_{fast_period}'] > threshold), 
               'ema_crossover'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_stochastic(
        self, 
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        slowing: int = 3,
        overbought: float = 80.0,
        oversold: float = 20.0
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            slowing: Slowing period
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            DataFrame with added Stochastic Oscillator
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            slowk, slowd = self.talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=k_period,
                slowk_period=slowing,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
        else:
            # Calculate Stochastic Oscillator using pandas
            # Find the lowest low and highest high for the period
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            # Calculate %K
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            
            # Apply slowing if needed
            if slowing > 1:
                df['stoch_k'] = df['stoch_k'].rolling(window=slowing).mean()
            
            # Calculate %D (moving average of %K)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Generate Stochastic signals
        df['stoch_signal'] = SIGNAL_NEUTRAL
        
        # Detect crossovers
        df.loc[(df['stoch_k'] > df['stoch_d']) & 
               (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)), 
               'stoch_signal'] = SIGNAL_BUY
        
        df.loc[(df['stoch_k'] < df['stoch_d']) & 
               (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)), 
               'stoch_signal'] = SIGNAL_SELL
        
        # Strengthen signals in oversold/overbought conditions
        df.loc[(df['stoch_signal'] == SIGNAL_BUY) & 
               (df['stoch_k'] < oversold), 
               'stoch_signal'] = SIGNAL_STRONG_BUY
        
        df.loc[(df['stoch_signal'] == SIGNAL_SELL) & 
               (df['stoch_k'] > overbought), 
               'stoch_signal'] = SIGNAL_STRONG_SELL
        
        # Add overbought/oversold indicators
        df['stoch_overbought'] = df['stoch_k'] > overbought
        df['stoch_oversold'] = df['stoch_k'] < oversold
        
        return df
    
    @handle_nan_values
    def add_atr(
        self, 
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range (ATR) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            DataFrame with added ATR
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            df['atr'] = self.talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=period
            )
        else:
            # Calculate ATR using pandas
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            # True Range is the greatest of the three
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # ATR is the moving average of the True Range
            df['atr'] = true_range.rolling(window=period).mean()
        
        # Add ATR percentage (ATR relative to price)
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Add ATR-based volatility classification
        median_atr_pct = df['atr_percent'].median()
        df['volatility'] = 'normal'
        df.loc[df['atr_percent'] > median_atr_pct * 1.5, 'volatility'] = 'high'
        df.loc[df['atr_percent'] < median_atr_pct * 0.5, 'volatility'] = 'low'
        
        return df
    
    @handle_nan_values
    def add_ichimoku(
        self, 
        df: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26
    ) -> pd.DataFrame:
        """
        Add Ichimoku Cloud to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            conversion_period: Tenkan-sen (Conversion Line) period
            base_period: Kijun-sen (Base Line) period
            span_b_period: Senkou Span B period
            displacement: Displacement period (Chikou Span)
            
        Returns:
            DataFrame with added Ichimoku Cloud
        """
        df = df.copy()
        
        # Calculate Tenkan-sen (Conversion Line)
        high_tenkan = df['high'].rolling(window=conversion_period).max()
        low_tenkan = df['low'].rolling(window=conversion_period).min()
        df['ichimoku_conversion'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line)
        high_kijun = df['high'].rolling(window=base_period).max()
        low_kijun = df['low'].rolling(window=base_period).min()
        df['ichimoku_base'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        high_senkou = df['high'].rolling(window=span_b_period).max()
        low_senkou = df['low'].rolling(window=span_b_period).min()
        df['ichimoku_span_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        df['ichimoku_lagging'] = df['close'].shift(-displacement)
        
        # Generate Ichimoku signals
        df['ichimoku_signal'] = SIGNAL_NEUTRAL
        
        # Conversion Line crosses above Base Line (bullish)
        df.loc[(df['ichimoku_conversion'] > df['ichimoku_base']) & 
               (df['ichimoku_conversion'].shift(1) <= df['ichimoku_base'].shift(1)), 
               'ichimoku_signal'] = SIGNAL_BUY
        
        # Conversion Line crosses below Base Line (bearish)
        df.loc[(df['ichimoku_conversion'] < df['ichimoku_base']) & 
               (df['ichimoku_conversion'].shift(1) >= df['ichimoku_base'].shift(1)), 
               'ichimoku_signal'] = SIGNAL_SELL
        
        # Price crosses above the cloud (strong bullish)
        df.loc[(df['close'] > df['ichimoku_span_a']) & 
               (df['close'] > df['ichimoku_span_b']) & 
               ((df['close'].shift(1) <= df['ichimoku_span_a'].shift(1)) | 
                (df['close'].shift(1) <= df['ichimoku_span_b'].shift(1))), 
               'ichimoku_signal'] = SIGNAL_STRONG_BUY
        
        # Price crosses below the cloud (strong bearish)
        df.loc[(df['close'] < df['ichimoku_span_a']) & 
               (df['close'] < df['ichimoku_span_b']) & 
               ((df['close'].shift(1) >= df['ichimoku_span_a'].shift(1)) | 
                (df['close'].shift(1) >= df['ichimoku_span_b'].shift(1))), 
               'ichimoku_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_volume_profile(
        self, 
        df: pd.DataFrame,
        period: int = 14,
        bins: int = 10
    ) -> pd.DataFrame:
        """
        Add Volume Profile analysis to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for volume analysis
            bins: Number of price bins for volume distribution
            
        Returns:
            DataFrame with added Volume Profile metrics
        """
        df = df.copy()
        
        # Calculate relative volume (compared to moving average)
        df['volume_sma'] = df['volume'].rolling(window=period).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # Calculate on-balance volume (OBV)
        df['obv'] = 0
        df.loc[df['close'] > df['close'].shift(1), 'obv'] = df['volume']
        df.loc[df['close'] < df['close'].shift(1), 'obv'] = -df['volume']
        df['obv'] = df['obv'].cumsum()
        
        # Calculate price-volume trend
        df['price_volume_trend'] = (df['close'] - df['close'].shift(1)) * df['volume'] / 1000000
        df['price_volume_trend'] = df['price_volume_trend'].cumsum()
        
        # Calculate volume-weighted average price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Generate volume signals
        df['volume_signal'] = SIGNAL_NEUTRAL
        
        # High volume breakouts
        df.loc[(df['close'] > df['close'].shift(1)) & 
               (df['relative_volume'] > 1.5), 
               'volume_signal'] = SIGNAL_BUY
        
        df.loc[(df['close'] < df['close'].shift(1)) & 
               (df['relative_volume'] > 1.5), 
               'volume_signal'] = SIGNAL_SELL
        
        # Very high volume breakouts (strong signals)
        df.loc[(df['close'] > df['close'].shift(1)) & 
               (df['relative_volume'] > 2.5), 
               'volume_signal'] = SIGNAL_STRONG_BUY
        
        df.loc[(df['close'] < df['close'].shift(1)) & 
               (df['relative_volume'] > 2.5), 
               'volume_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_fibonacci_levels(
        self, 
        df: pd.DataFrame,
        period: int = 100
    ) -> pd.DataFrame:
        """
        Add Fibonacci retracement levels to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: Lookback period for high/low calculation
            
        Returns:
            DataFrame with added Fibonacci levels
        """
        df = df.copy()
        
        # Calculate rolling high and low for the period
        df['period_high'] = df['high'].rolling(window=period).max()
        df['period_low'] = df['low'].rolling(window=period).min()
        
        # Calculate the range
        df['fib_range'] = df['period_high'] - df['period_low']
        
        # Calculate Fibonacci levels
        df['fib_0.0'] = df['period_low']  # 0% level
        df['fib_0.236'] = df['period_low'] + 0.236 * df['fib_range']  # 23.6% level
        df['fib_0.382'] = df['period_low'] + 0.382 * df['fib_range']  # 38.2% level
        df['fib_0.5'] = df['period_low'] + 0.5 * df['fib_range']  # 50% level
        df['fib_0.618'] = df['period_low'] + 0.618 * df['fib_range']  # 61.8% level
        df['fib_0.786'] = df['period_low'] + 0.786 * df['fib_range']  # 78.6% level
        df['fib_1.0'] = df['period_high']  # 100% level
        
        # Generate Fibonacci signals
        df['fib_signal'] = SIGNAL_NEUTRAL
        
        # Bullish signals when price bounces off key Fibonacci levels during uptrend
        uptrend = df['close'] > df['close'].rolling(window=20).mean()
        
        for level in [0.236, 0.382, 0.5, 0.618]:
            level_col = f'fib_{level}'
            
            # Price bounces off the level (gets within 0.5% and then moves away)
            close_to_level = (df['close'] / df[level_col] - 1).abs() < 0.005
            bounce_up = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
            
            df.loc[uptrend & close_to_level & bounce_up, 'fib_signal'] = SIGNAL_BUY
            
            # Stronger signal for 0.618 level (golden ratio)
            if level == 0.618:
                df.loc[uptrend & close_to_level & bounce_up, 'fib_signal'] = SIGNAL_STRONG_BUY
        
        # Bearish signals when price fails at key Fibonacci levels during downtrend
        downtrend = df['close'] < df['close'].rolling(window=20).mean()
        
        for level in [0.618, 0.786, 1.0]:
            level_col = f'fib_{level}'
            
            # Price fails at the level (gets within 0.5% and then moves away)
            close_to_level = (df['close'] / df[level_col] - 1).abs() < 0.005
            bounce_down = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
            
            df.loc[downtrend & close_to_level & bounce_down, 'fib_signal'] = SIGNAL_SELL
            
            # Stronger signal for 0.618 level (golden ratio)
            if level == 0.618:
                df.loc[downtrend & close_to_level & bounce_down, 'fib_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_parabolic_sar(
        self, 
        df: pd.DataFrame,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> pd.DataFrame:
        """
        Add Parabolic SAR to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor
            
        Returns:
            DataFrame with added Parabolic SAR
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            df['sar'] = self.talib.SAR(
                df['high'].values,
                df['low'].values,
                acceleration=acceleration,
                maximum=maximum
            )
        else:
            # Parabolic SAR is complex to implement from scratch
            # This is a simplified version
            df['sar'] = np.nan
            
            # Initialize variables
            trend = 1  # 1 for uptrend, -1 for downtrend
            sar = df['low'].iloc[0]
            extreme_point = df['high'].iloc[0]
            acc_factor = acceleration
            
            # Calculate SAR for each candle
            for i in range(1, len(df)):
                # Calculate SAR value
                sar = sar + acc_factor * (extreme_point - sar)
                
                if trend == 1:  # Uptrend
                    # SAR can't be higher than the previous two lows
                    sar = min(sar, df['low'].iloc[max(0, i-2):i].min())
                    
                    # Check for trend reversal
                    if sar > df['low'].iloc[i]:
                        trend = -1
                        sar = df['high'].iloc[i-1]
                        extreme_point = df['low'].iloc[i]
                        acc_factor = acceleration
                    else:
                        # Update extreme point and acceleration factor
                        if df['high'].iloc[i] > extreme_point:
                            extreme_point = df['high'].iloc[i]
                            acc_factor = min(acc_factor + acceleration, maximum)
                else:  # Downtrend
                    # SAR can't be lower than the previous two highs
                    sar = max(sar, df['high'].iloc[max(0, i-2):i].max())
                    
                    # Check for trend reversal
                    if sar < df['high'].iloc[i]:
                        trend = 1
                        sar = df['low'].iloc[i-1]
                        extreme_point = df['high'].iloc[i]
                        acc_factor = acceleration
                    else:
                        # Update extreme point and acceleration factor
                        if df['low'].iloc[i] < extreme_point:
                            extreme_point = df['low'].iloc[i]
                            acc_factor = min(acc_factor + acceleration, maximum)
                
                df.iloc[i, df.columns.get_loc('sar')] = sar
        
        # Generate Parabolic SAR signals
        df['sar_signal'] = SIGNAL_NEUTRAL
        
        # Detect trend changes
        df.loc[(df['close'] > df['sar']) & (df['close'].shift(1) <= df['sar'].shift(1)), 'sar_signal'] = SIGNAL_BUY
        df.loc[(df['close'] < df['sar']) & (df['close'].shift(1) >= df['sar'].shift(1)), 'sar_signal'] = SIGNAL_SELL
        
        # Strong signals when price is far from SAR
        threshold = df['atr'].rolling(window=14).mean() if 'atr' in df.columns else df['close'].rolling(window=14).std()
        
        df.loc[(df['sar_signal'] == SIGNAL_BUY) & 
               (df['close'] - df['sar'] > threshold), 
               'sar_signal'] = SIGNAL_STRONG_BUY
        
        df.loc[(df['sar_signal'] == SIGNAL_SELL) & 
               (df['sar'] - df['close'] > threshold), 
               'sar_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_adx(
        self, 
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            period: ADX period
            
        Returns:
            DataFrame with added ADX
        """
        df = df.copy()
        
        if self.use_talib and self.talib:
            df['adx'] = self.talib.ADX(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=period
            )
            df['plus_di'] = self.talib.PLUS_DI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=period
            )
            df['minus_di'] = self.talib.MINUS_DI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=period
            )
        else:
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Calculate +DM and -DM
            plus_dm = df['high'] - df['high'].shift()
            minus_dm = df['low'].shift() - df['low']
            
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            # Calculate smoothed values
            tr_period = true_range.rolling(window=period).sum()
            plus_dm_period = plus_dm.rolling(window=period).sum()
            minus_dm_period = minus_dm.rolling(window=period).sum()
            
            # Calculate +DI and -DI
            df['plus_di'] = 100 * (plus_dm_period / tr_period)
            df['minus_di'] = 100 * (minus_dm_period / tr_period)
            
            # Calculate DX
            dx = 100 * ((df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di']).abs())
            
            # Calculate ADX
            df['adx'] = dx.rolling(window=period).mean()
        
        # Generate ADX signals
        df['adx_signal'] = SIGNAL_NEUTRAL
        
        # Strong trend when ADX is high
        strong_trend = df['adx'] > 25
        
        # Crossovers during strong trend
        df.loc[strong_trend & 
               (df['plus_di'] > df['minus_di']) & 
               (df['plus_di'].shift(1) <= df['minus_di'].shift(1)), 
               'adx_signal'] = SIGNAL_BUY
        
        df.loc[strong_trend & 
               (df['plus_di'] < df['minus_di']) & 
               (df['plus_di'].shift(1) >= df['minus_di'].shift(1)), 
               'adx_signal'] = SIGNAL_SELL
        
        # Very strong trend signals
        very_strong_trend = df['adx'] > 40
        
        df.loc[very_strong_trend & 
               (df['adx_signal'] == SIGNAL_BUY), 
               'adx_signal'] = SIGNAL_STRONG_BUY
        
        df.loc[very_strong_trend & 
               (df['adx_signal'] == SIGNAL_SELL), 
               'adx_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    @handle_nan_values
    def add_support_resistance(
        self, 
        df: pd.DataFrame,
        window: int = 10,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Add Support and Resistance levels to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window to look for local extrema
            threshold: Minimum percentage change to consider a level
            
        Returns:
            DataFrame with added Support and Resistance levels
        """
        df = df.copy()
        
        # Initialize columns
        df['is_support'] = False
        df['is_resistance'] = False
        df['support_level'] = np.nan
        df['resistance_level'] = np.nan
        
        # Function to identify local minima and maxima
        def is_local_min(window_prices):
            center_idx = len(window_prices) // 2
            center_val = window_prices[center_idx]
            return all(center_val <= price for price in window_prices)
        
        def is_local_max(window_prices):
            center_idx = len(window_prices) // 2
            center_val = window_prices[center_idx]
            return all(center_val >= price for price in window_prices)
        
        # Identify support and resistance levels
        for i in range(window, len(df) - window):
            low_window = df['low'].iloc[i-window:i+window+1].values
            high_window = df['high'].iloc[i-window:i+window+1].values
            
            # Check for support (local minimum)
            if is_local_min(low_window):
                df.iloc[i, df.columns.get_loc('is_support')] = True
                df.iloc[i, df.columns.get_loc('support_level')] = df['low'].iloc[i]
            
            # Check for resistance (local maximum)
            if is_local_max(high_window):
                df.iloc[i, df.columns.get_loc('is_resistance')] = True
                df.iloc[i, df.columns.get_loc('resistance_level')] = df['high'].iloc[i]
        
        # Forward fill support and resistance levels
        df['support_level'] = df['support_level'].fillna(method='ffill')
        df['resistance_level'] = df['resistance_level'].fillna(method='ffill')
        
        # Generate signals based on price approaching support or resistance
        df['sr_signal'] = SIGNAL_NEUTRAL
        
        # Calculate distance to nearest support and resistance
        df['distance_to_support'] = (df['close'] / df['support_level'] - 1).abs()
        df['distance_to_resistance'] = (df['close'] / df['resistance_level'] - 1).abs()
        
        # Signals when price approaches support or resistance
        approaching_support = (df['distance_to_support'] < threshold) & (df['close'] > df['support_level'])
        approaching_resistance = (df['distance_to_resistance'] < threshold) & (df['close'] < df['resistance_level'])
        
        # Generate signals
        df.loc[approaching_support, 'sr_signal'] = SIGNAL_BUY
        df.loc[approaching_resistance, 'sr_signal'] = SIGNAL_SELL
        
        # Strong signals when price bounces off support or resistance
        bounce_off_support = approaching_support & (df['close'] > df['close'].shift(1))
        bounce_off_resistance = approaching_resistance & (df['close'] < df['close'].shift(1))
        
        df.loc[bounce_off_support, 'sr_signal'] = SIGNAL_STRONG_BUY
        df.loc[bounce_off_resistance, 'sr_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    def generate_combined_signal(
        self, 
        df: pd.DataFrame,
        indicators: Dict[str, float] = None,
        threshold: float = 0.7,
        confirmation_needed: int = 2
    ) -> pd.DataFrame:
        """
        Generate combined trading signals from multiple indicators.
        
        Args:
            df: DataFrame with indicator signals
            indicators: Dictionary of indicators and their weights
            threshold: Signal threshold (0-1)
            confirmation_needed: Number of indicators needed for confirmation
            
        Returns:
            DataFrame with combined signals
        """
        df = df.copy()
        
        # Default weights if not provided
        if indicators is None:
            indicators = {
                'rsi_signal': 1.0,
                'macd_crossover': 1.0,
                'bb_signal': 0.8,
                'ema_crossover': 1.0,
                'stoch_signal': 0.7,
                'volume_signal': 0.6,
                'adx_signal': 0.8,
                'sar_signal': 0.7,
                'sr_signal': 0.9,
                'ichimoku_signal': 0.6,
                'fib_signal': 0.5
            }
        
        # Filter to only include indicators that exist in the DataFrame
        valid_indicators = {k: v for k, v in indicators.items() if k in df.columns}
        
        if not valid_indicators:
            logger.warning("No valid indicators found in DataFrame for signal generation")
            df['combined_signal'] = SIGNAL_NEUTRAL
            df['signal_strength'] = 0.0
            return df
        
        # Normalize weights
        total_weight = sum(valid_indicators.values())
        normalized_weights = {k: v / total_weight for k, v in valid_indicators.items()}
        
        # Calculate weighted signal
        df['weighted_signal'] = 0.0
        for indicator, weight in normalized_weights.items():
            df['weighted_signal'] += df[indicator] * weight
        
        # Calculate signal strength (absolute value, normalized to 0-1)
        max_possible = max(abs(SIGNAL_STRONG_BUY), abs(SIGNAL_STRONG_SELL))
        df['signal_strength'] = df['weighted_signal'].abs() / max_possible
        
        # Count confirming indicators
        df['buy_indicators'] = 0
        df['sell_indicators'] = 0
        
        for indicator in valid_indicators.keys():
            df.loc[df[indicator] > 0, 'buy_indicators'] += 1
            df.loc[df[indicator] < 0, 'sell_indicators'] += 1
        
        # Generate combined signal based on weighted signal and confirmation count
        df['combined_signal'] = SIGNAL_NEUTRAL
        
        # Buy signals
        df.loc[(df['weighted_signal'] > 0) & 
               (df['signal_strength'] >= threshold) & 
               (df['buy_indicators'] >= confirmation_needed), 
               'combined_signal'] = SIGNAL_BUY
        
        # Strong buy signals
        df.loc[(df['weighted_signal'] > 0) & 
               (df['signal_strength'] >= threshold * 1.3) & 
               (df['buy_indicators'] >= confirmation_needed + 1), 
               'combined_signal'] = SIGNAL_STRONG_BUY
        
        # Sell signals
        df.loc[(df['weighted_signal'] < 0) & 
               (df['signal_strength'] >= threshold) & 
               (df['sell_indicators'] >= confirmation_needed), 
               'combined_signal'] = SIGNAL_SELL
        
        # Strong sell signals
        df.loc[(df['weighted_signal'] < 0) & 
               (df['signal_strength'] >= threshold * 1.3) & 
               (df['sell_indicators'] >= confirmation_needed + 1), 
               'combined_signal'] = SIGNAL_STRONG_SELL
        
        return df
    
    def apply_all_indicators(
        self, 
        df: pd.DataFrame,
        include: List[str] = None,
        exclude: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply all or selected indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            include: List of indicators to include (None for all)
            exclude: List of indicators to exclude (None for none)
            
        Returns:
            DataFrame with all indicators applied
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Define all available indicators
        all_indicators = {
            'moving_averages': self.add_moving_averages,
            'rsi': self.add_rsi,
            'macd': self.add_macd,
            'bollinger_bands': self.add_bollinger_bands,
            'ema_crossover': self.add_ema_crossover,
            'stochastic': self.add_stochastic,
            'atr': self.add_atr,
            'volume_profile': self.add_volume_profile,
            'adx': self.add_adx,
            'parabolic_sar': self.add_parabolic_sar,
            'support_resistance': self.add_support_resistance,
            'fibonacci': self.add_fibonacci_levels,
            'ichimoku': self.add_ichimoku
        }
        
        # Filter indicators based on include/exclude lists
        if include is not None:
            indicators_to_apply = {k: v for k, v in all_indicators.items() if k in include}
        else:
            indicators_to_apply = all_indicators.copy()
        
        if exclude is not None:
            indicators_to_apply = {k: v for k, v in indicators_to_apply.items() if k not in exclude}
        
        # Apply selected indicators
        for name, func in indicators_to_apply.items():
            logger.info(f"Applying indicator: {name}")
            try:
                df = func(df)
            except Exception as e:
                logger.error(f"Error applying {name}: {e}")
        
        # Generate combined signal
        try:
            df = self.generate_combined_signal(df)
        except Exception as e:
            logger.error(f"Error generating combined signal: {e}")
        
        return df
    
    def plot_indicators(
        self, 
        df: pd.DataFrame,
        indicators: List[str] = None,
        title: str = "Technical Indicators",
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot technical indicators for visualization.
        
        Args:
            df: DataFrame with indicators
            indicators: List of indicators to plot (None for default selection)
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        # Default indicators to plot
        if indicators is None:
            indicators = ['close', 'ema_9', 'ema_21', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        
        # Check which indicators are available
        available = [ind for ind in indicators if ind in df.columns]
        
        if not available:
            logger.warning("No requested indicators available in DataFrame")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No indicators available to plot", 
                    horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Determine how many subplots we need
        price_indicators = ['close', 'open', 'high', 'low', 'ema', 'sma', 'wma', 
                           'bb_upper', 'bb_middle', 'bb_lower', 'vwap', 
                           'ichimoku', 'sar', 'support_level', 'resistance_level']
        
        oscillator_indicators = ['rsi', 'stoch', 'macd', 'adx', 'obv', 'volume']
        
        # Check what types of indicators we have
        has_price = any(any(pi in ind for pi in price_indicators) for ind in available)
        has_oscillators = any(any(oi in ind for oi in oscillator_indicators) for ind in available)
        has_volume = 'volume' in available
        
        # Create subplot structure
        n_plots = sum([has_price, has_oscillators, has_volume])
        if n_plots == 0:
            n_plots = 1
        
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, 
                               gridspec_kw={'height_ratios': [3 if has_price else 1] + 
                                           [1 for _ in range(n_plots-1)]})
        
        if n_plots == 1:
            axs = [axs]
        
        plot_idx = 0
        
        # Plot price and price-related indicators
        if has_price:
            ax = axs[plot_idx]
            
            # Plot price
            if 'close' in df.columns:
                ax.plot(df.index, df['close'], label='Close', color='black', alpha=0.75)
            
            # Plot EMAs
            ema_cols = [col for col in df.columns if 'ema_' in col]
            for col in ema_cols:
                if col in available:
                    ax.plot(df.index, df[col], label=col.replace('_', ' ').title(), alpha=0.75)
            
            # Plot Bollinger Bands
            if all(x in df.columns for x in ['bb_upper', 'bb_lower']):
                ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                               color='gray', alpha=0.15, label='Bollinger Bands')
                ax.plot(df.index, df['bb_upper'], color='gray', linestyle='--', alpha=0.5)
                ax.plot(df.index, df['bb_lower'], color='gray', linestyle='--', alpha=0.5)
            
            # Plot support and resistance levels
            if 'support_level' in df.columns:
                supports = df[df['is_support']]['support_level']
                ax.scatter(supports.index, supports, marker='^', color='green', 
                          s=100, label='Support')
            
            if 'resistance_level' in df.columns:
                resistances = df[df['is_resistance']]['resistance_level']
                ax.scatter(resistances.index, resistances, marker='v', color='red', 
                          s=100, label='Resistance')
            
            # Plot buy/sell signals if available
            if 'combined_signal' in df.columns:
                buys = df[df['combined_signal'] >= SIGNAL_BUY]
                sells = df[df['combined_signal'] <= SIGNAL_SELL]
                
                ax.scatter(buys.index, buys['close'] * 0.99, marker='^', color='green', 
                          s=100, label='Buy Signal')
                ax.scatter(sells.index, sells['close'] * 1.01, marker='v', color='red', 
                          s=100, label='Sell Signal')
            
            ax.set_title(f"{title} - Price Chart")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot oscillators
        if has_oscillators:
            ax = axs[plot_idx]
            
            # Plot RSI
            if 'rsi' in df.columns:
                ax.plot(df.index, df['rsi'], label='RSI', color='purple')
                ax.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                ax.axhline(y=30, color='g', linestyle='--', alpha=0.3)
                ax.set_ylim(0, 100)
            
            # Plot MACD
            if all(x in df.columns for x in ['macd', 'macd_signal']):
                ax.plot(df.index, df['macd'], label='MACD', color='blue')
                ax.plot(df.index, df['macd_signal'], label='Signal', color='red')
                
                if 'macd_hist' in df.columns:
                    ax.bar(df.index, df['macd_hist'], label='Histogram', 
                          color=df['macd_hist'].apply(lambda x: 'green' if x > 0 else 'red'),
                          alpha=0.3)
            
            # Plot Stochastic
            if all(x in df.columns for x in ['stoch_k', 'stoch_d']):
                ax.plot(df.index, df['stoch_k'], label='Stoch %K', color='blue')
                ax.plot(df.index, df['stoch_d'], label='Stoch %D', color='red')
                ax.axhline(y=80, color='r', linestyle='--', alpha=0.3)
                ax.axhline(y=20, color='g', linestyle='--', alpha=0.3)
                ax.set_ylim(0, 100)
            
            ax.set_title("Oscillators")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot volume
        if has_volume:
            ax = axs[plot_idx]
            
            # Plot volume bars
            if 'volume' in df.columns:
                ax.bar(df.index, df['volume'], label='Volume', 
                      color=df['close'].diff().apply(lambda x: 'green' if x > 0 else 'red'),
                      alpha=0.5)
                
                if 'volume_sma' in df.columns:
                    ax.plot(df.index, df['volume_sma'], label='Volume SMA', 
                           color='blue', linestyle='--')
            
            ax.set_title("Volume")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        return fig

    def get_signal_statistics(self, df: pd.DataFrame, signal_column: str = 'combined_signal') -> Dict:
        """
        Calculate statistics about generated signals.
        
        Args:
            df: DataFrame with signals
            signal_column: Column containing signals
            
        Returns:
            Dictionary with signal statistics
        """
        if signal_column not in df.columns:
            return {"error": f"Signal column '{signal_column}' not found in DataFrame"}
        
        # Count signals by type
        signal_counts = df[signal_column].value_counts().to_dict()
        
        # Map numeric signals to text
        signal_counts_text = {SIGNAL_MAPPING.get(k, f"Unknown ({k})"): v 
                             for k, v in signal_counts.items()}
        
        # Calculate signal transitions
        transitions = {}
        prev_signal = None
        
        for signal in df[signal_column]:
            if prev_signal is not None:
                if prev_signal != signal:
                    key = f"{SIGNAL_MAPPING.get(prev_signal, 'Unknown')} → {SIGNAL_MAPPING.get(signal, 'Unknown')}"
                    transitions[key] = transitions.get(key, 0) + 1
            prev_signal = signal
        
        # Calculate average signal duration
        durations = []
        current_signal = None
        current_start = None
        
        for i, row in enumerate(df.itertuples()):
            signal = getattr(row, signal_column)
            
            if current_signal is None:
                current_signal = signal
                current_start = i
            elif signal != current_signal:
                durations.append(i - current_start)
                current_signal = signal
                current_start = i
        
        # Add the last duration
        if current_start is not None and current_start < len(df):
            durations.append(len(df) - current_start)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "signal_counts": signal_counts_text,
            "transitions": transitions,
            "average_duration": avg_duration,
            "total_signals": len(df) - df[signal_column].isna().sum(),
            "signal_ratio": {
                "buy": (df[signal_column] > 0).sum() / len(df) if len(df) > 0 else 0,
                "neutral": (df[signal_column] == 0).sum() / len(df) if len(df) > 0 else 0,
                "sell": (df[signal_column] < 0).sum() / len(df) if len(df) > 0 else 0
            }
        }


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all indicators added
    """
    indicators = TechnicalIndicators()
    return indicators.apply_all_indicators(df)


def get_signal_name(signal_value: int) -> str:
    """
    Get the text representation of a signal value.
    
    Args:
        signal_value: Numeric signal value
        
    Returns:
        Text representation of the signal
    """
    return SIGNAL_MAPPING.get(signal_value, f"Unknown ({signal_value})")


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    print("Downloading sample data...")
    data = yf.download("BTC-USD", period="60d", interval="1h")
    
    # Create indicators instance
    indicators = TechnicalIndicators()
    
    # Apply indicators
    print("Applying technical indicators...")
    df_with_indicators = indicators.apply_all_indicators(data)
    
    # Print sample of the data
    print("\nSample of data with indicators:")
    print(df_with_indicators.tail())
    
    # Get signal statistics
    print("\nSignal statistics:")
    stats = indicators.get_signal_statistics(df_with_indicators)
    print(f"Total signals: {stats['total_signals']}")
    print(f"Signal counts: {stats['signal_counts']}")
    print(f"Signal ratio: Buy {stats['signal_ratio']['buy']:.2%}, "
          f"Neutral {stats['signal_ratio']['neutral']:.2%}, "
          f"Sell {stats['signal_ratio']['sell']:.2%}")
    
    # Plot indicators
    print("\nPlotting indicators...")
    fig = indicators.plot_indicators(df_with_indicators)
    plt.show()
