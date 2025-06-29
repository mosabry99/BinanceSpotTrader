"""
Technical Indicators Module for Neural Pulse Scalper

This module provides a comprehensive set of technical indicators and market analysis
functions used by the Neural Pulse Scalper strategy. It includes optimized implementations
of various indicators with proper error handling and type hints.

Indicators included:
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- Bollinger Bands
- VWAP (Volume Weighted Average Price)
- Volume Analysis
- MACD (Moving Average Convergence Divergence)
- Candlestick Pattern Detection
- Support/Resistance Levels
- Volatility Calculations
- Market Structure Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from enum import Enum
import logging
from dataclasses import dataclass
import talib
from talib import abstract
import pandas_ta as pta
from scipy.signal import argrelextrema
from scipy.stats import linregress

# Setup logging
logger = logging.getLogger(__name__)


class CandlePattern(Enum):
    """Enumeration of supported candlestick patterns."""
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    SPINNING_TOP = "spinning_top"
    MARUBOZU = "marubozu"
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    THREE_BAR_REVERSAL = "three_bar_reversal"
    PIN_BAR = "pin_bar"


class MarketStructure(Enum):
    """Enumeration of market structure states."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    REVERSAL_POTENTIAL = "reversal_potential"


@dataclass
class SupportResistanceLevel:
    """Dataclass to store support/resistance level information."""
    price: float
    strength: int  # Number of touches
    type: str  # 'support' or 'resistance'
    timestamp: pd.Timestamp
    confirmed: bool = False


class TechnicalIndicators:
    """
    A comprehensive technical indicators class that provides methods for calculating
    various technical indicators and market analysis functions.
    """
    
    def __init__(self, use_talib: bool = True):
        """
        Initialize the TechnicalIndicators class.
        
        Args:
            use_talib: Whether to use talib for calculations (faster) or pandas_ta (more indicators)
        """
        self.use_talib = use_talib
        logger.info(f"Initialized TechnicalIndicators with use_talib={use_talib}")
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe has the required columns for indicator calculations.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            
        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"DataFrame missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if df.empty:
            error_msg = "DataFrame is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return True
    
    def add_all_indicators(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Add all indicators to the dataframe based on the configuration.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            config: Configuration dictionary with indicator settings
            
        Returns:
            pd.DataFrame: DataFrame with all indicators added
        """
        try:
            self.validate_dataframe(df)
            
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Add EMA indicators
            if 'ema' in config:
                for period in config['ema'].get('periods', [5, 8, 13, 20, 50, 200]):
                    df = self.add_ema(df, period)
            
            # Add RSI
            if 'rsi' in config:
                period = config['rsi'].get('period', 14)
                df = self.add_rsi(df, period)
            
            # Add Bollinger Bands
            if 'bollinger_bands' in config:
                period = config['bollinger_bands'].get('period', 20)
                std_dev = config['bollinger_bands'].get('std_dev', 2.0)
                df = self.add_bollinger_bands(df, period, std_dev)
            
            # Add VWAP
            if 'vwap' in config:
                df = self.add_vwap(df)
            
            # Add MACD
            if 'macd' in config:
                fast_period = config['macd'].get('fast_period', 12)
                slow_period = config['macd'].get('slow_period', 26)
                signal_period = config['macd'].get('signal_period', 9)
                df = self.add_macd(df, fast_period, slow_period, signal_period)
            
            # Add Volume indicators
            df = self.add_volume_indicators(df)
            
            # Add Volatility indicators
            df = self.add_volatility_indicators(df)
            
            # Add Market Structure analysis
            df = self.add_market_structure(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding all indicators: {str(e)}")
            raise
    
    def add_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Add Exponential Moving Average to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            period: EMA period
            
        Returns:
            pd.DataFrame: DataFrame with EMA added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            column_name = f'ema_{period}'
            
            if self.use_talib:
                df[column_name] = talib.EMA(df['close'].values, timeperiod=period)
            else:
                df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
                
            return df
            
        except Exception as e:
            logger.error(f"Error calculating EMA({period}): {str(e)}")
            raise
    
    def add_ema_crossover_signals(self, df: pd.DataFrame, fast_period: int = 5, slow_period: int = 20) -> pd.DataFrame:
        """
        Add EMA crossover signals to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            pd.DataFrame: DataFrame with EMA crossover signals added
        """
        try:
            df = df.copy()
            
            # Ensure EMAs are calculated
            fast_col = f'ema_{fast_period}'
            slow_col = f'ema_{slow_period}'
            
            if fast_col not in df.columns:
                df = self.add_ema(df, fast_period)
            
            if slow_col not in df.columns:
                df = self.add_ema(df, slow_period)
            
            # Calculate the difference between fast and slow EMAs
            df['ema_diff'] = df[fast_col] - df[slow_col]
            
            # Generate crossover signals
            df['ema_crossover'] = 0  # 0 = no signal
            df.loc[df['ema_diff'] > 0, 'ema_crossover'] = 1  # 1 = bullish (fast above slow)
            df.loc[df['ema_diff'] < 0, 'ema_crossover'] = -1  # -1 = bearish (fast below slow)
            
            # Detect actual crossover points
            df['ema_cross_signal'] = 0
            df.loc[(df['ema_diff'] > 0) & (df['ema_diff'].shift(1) <= 0), 'ema_cross_signal'] = 1  # Bullish crossover
            df.loc[(df['ema_diff'] < 0) & (df['ema_diff'].shift(1) >= 0), 'ema_cross_signal'] = -1  # Bearish crossover
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating EMA crossover signals: {str(e)}")
            raise
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            pd.DataFrame: DataFrame with RSI added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            column_name = f'rsi_{period}'
            
            if self.use_talib:
                df[column_name] = talib.RSI(df['close'].values, timeperiod=period)
            else:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).fillna(0)
                loss = -delta.where(delta < 0, 0).fillna(0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                df[column_name] = 100 - (100 / (1 + rs))
                
            # Add RSI zones for easier signal generation
            df['rsi_zone'] = 'neutral'
            df.loc[df[column_name] > 70, 'rsi_zone'] = 'overbought'
            df.loc[df[column_name] < 30, 'rsi_zone'] = 'oversold'
            df.loc[(df[column_name] >= 40) & (df[column_name] <= 60), 'rsi_zone'] = 'middle'
                
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RSI({period}): {str(e)}")
            raise
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            period: Bollinger Bands period
            std_dev: Number of standard deviations
            
        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            if self.use_talib:
                upper, middle, lower = talib.BBANDS(
                    df['close'].values,
                    timeperiod=period,
                    nbdevup=std_dev,
                    nbdevdn=std_dev,
                    matype=0  # Simple Moving Average
                )
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
            else:
                # Calculate middle band (SMA)
                df['bb_middle'] = df['close'].rolling(window=period).mean()
                
                # Calculate standard deviation
                rolling_std = df['close'].rolling(window=period).std()
                
                # Calculate upper and lower bands
                df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
                df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
            
            # Calculate bandwidth and %B for additional signals
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Add touch signals
            df['bb_touch_upper'] = (df['high'] >= df['bb_upper']).astype(int)
            df['bb_touch_lower'] = (df['low'] <= df['bb_lower']).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands({period}, {std_dev}): {str(e)}")
            raise
    
    def add_vwap(self, df: pd.DataFrame, anchor: str = 'day') -> pd.DataFrame:
        """
        Add Volume Weighted Average Price to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            anchor: VWAP anchor point ('day', 'week', 'month')
            
        Returns:
            pd.DataFrame: DataFrame with VWAP added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # Ensure we have datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex, VWAP calculation may be inaccurate")
                
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Create anchor points
            if anchor == 'day':
                df['anchor'] = df.index.date
            elif anchor == 'week':
                df['anchor'] = df.index.isocalendar().week
            elif anchor == 'month':
                df['anchor'] = df.index.month
            else:
                df['anchor'] = df.index.date  # Default to day
            
            # Group by anchor and calculate VWAP
            df['vwap'] = 0.0
            
            # Calculate cumulative values within each anchor group
            for group_name, group_data in df.groupby('anchor'):
                cumulative_tp_vol = (group_data['typical_price'] * group_data['volume']).cumsum()
                cumulative_vol = group_data['volume'].cumsum()
                
                # Update VWAP for this group
                df.loc[group_data.index, 'vwap'] = cumulative_tp_vol / cumulative_vol
            
            # Calculate distance from VWAP
            df['vwap_distance'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
            df['vwap_cross'] = np.sign(df['close'] - df['vwap'])
            
            # Add VWAP cross signals
            df['vwap_cross_signal'] = 0
            df.loc[(df['vwap_cross'] > 0) & (df['vwap_cross'].shift(1) <= 0), 'vwap_cross_signal'] = 1  # Bullish cross
            df.loc[(df['vwap_cross'] < 0) & (df['vwap_cross'].shift(1) >= 0), 'vwap_cross_signal'] = -1  # Bearish cross
            
            # Clean up
            df.drop(['typical_price', 'anchor'], axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            raise
    
    def add_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            pd.DataFrame: DataFrame with MACD added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            if self.use_talib:
                macd, signal, hist = talib.MACD(
                    df['close'].values,
                    fastperiod=fast_period,
                    slowperiod=slow_period,
                    signalperiod=signal_period
                )
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_hist'] = hist
            else:
                # Calculate fast and slow EMAs
                ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
                ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
                
                # Calculate MACD line
                df['macd'] = ema_fast - ema_slow
                
                # Calculate signal line
                df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                
                # Calculate histogram
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Add MACD crossover signals
            df['macd_cross'] = 0
            df.loc[(df['macd'] > df['macd_signal']), 'macd_cross'] = 1  # Bullish
            df.loc[(df['macd'] < df['macd_signal']), 'macd_cross'] = -1  # Bearish
            
            # Add MACD crossover signals (actual crossover points)
            df['macd_cross_signal'] = 0
            df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_cross_signal'] = 1  # Bullish crossover
            df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_cross_signal'] = -1  # Bearish crossover
            
            # Add MACD histogram direction change signals
            df['macd_hist_direction'] = np.sign(df['macd_hist'] - df['macd_hist'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise
    
    def add_volume_indicators(self, df: pd.DataFrame, volume_period: int = 20) -> pd.DataFrame:
        """
        Add various volume indicators to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            volume_period: Period for volume indicators
            
        Returns:
            pd.DataFrame: DataFrame with volume indicators added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # Volume Moving Average
            df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
            
            # Volume Ratio (current volume / average volume)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Volume Spike detection (volume > X times average)
            df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
            
            # On-Balance Volume (OBV)
            if self.use_talib:
                df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            else:
                df['obv'] = 0
                df.loc[1:, 'obv'] = np.where(
                    df['close'] > df['close'].shift(1),
                    df['obv'].shift(1) + df['volume'],
                    np.where(
                        df['close'] < df['close'].shift(1),
                        df['obv'].shift(1) - df['volume'],
                        df['obv'].shift(1)
                    )
                )
            
            # Chaikin Money Flow (CMF)
            if self.use_talib:
                df['cmf'] = talib.ADOSC(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    df['volume'].values,
                    fastperiod=3,
                    slowperiod=10
                )
            else:
                mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
                df['cmf'] = mfv.rolling(window=volume_period).sum() / df['volume'].rolling(window=volume_period).sum()
            
            # Volume Weighted MACD
            # Use volume as weight for MACD calculation
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                df['vol_weighted_macd'] = df['macd'] * df['volume_ratio']
            
            # Volume Price Trend (VPT)
            df['vpt'] = (df['volume'] * (df['close'].pct_change())).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            raise
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with volatility indicators added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # True Range and Average True Range (ATR)
            if self.use_talib:
                df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            else:
                # Calculate True Range
                df['tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                # Calculate ATR
                df['atr_14'] = df['tr'].rolling(window=14).mean()
            
            # ATR percentage (ATR as percentage of price)
            df['atr_pct'] = (df['atr_14'] / df['close']) * 100
            
            # Historical Volatility (close-to-close)
            df['returns'] = df['close'].pct_change()
            df['hv_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100  # Annualized
            
            # Bollinger Bands Width (if not already calculated)
            if 'bb_width' not in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Keltner Channels
            if self.use_talib:
                df['kc_middle'] = talib.SMA(df['close'].values, timeperiod=20)
            else:
                df['kc_middle'] = df['close'].rolling(window=20).mean()
            
            df['kc_upper'] = df['kc_middle'] + (df['atr_14'] * 2)
            df['kc_lower'] = df['kc_middle'] - (df['atr_14'] * 2)
            
            # Volatility over different timeframes
            for period in [5, 30]:
                df[f'volatility_{period}m'] = df['returns'].rolling(window=period).std() * 100
            
            # Volatility Ratio (short-term vs long-term volatility)
            df['volatility_ratio'] = df['volatility_5m'] / df['volatility_30m']
            
            # Detect volatility expansion/contraction
            df['volatility_change'] = df['volatility_5m'].pct_change()
            df['volatility_regime'] = 'normal'
            df.loc[df['volatility_change'] > 0.5, 'volatility_regime'] = 'expansion'
            df.loc[df['volatility_change'] < -0.3, 'volatility_regime'] = 'contraction'
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            raise
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect various candlestick patterns in the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with candlestick patterns added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # Initialize pattern columns
            for pattern in CandlePattern:
                df[f'pattern_{pattern.value}'] = 0
            
            # Calculate candle properties
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['candle_range'] = df['high'] - df['low']
            df['body_pct'] = df['body_size'] / df['candle_range']
            df['is_bullish'] = (df['close'] > df['open']).astype(int)
            df['is_bearish'] = (df['close'] < df['open']).astype(int)
            
            # Detect Doji
            doji_condition = (df['body_size'] / df['candle_range'] < 0.1) & (df['candle_range'] > 0)
            df.loc[doji_condition, 'pattern_doji'] = 1
            
            # Detect Hammer and Hanging Man
            hammer_condition = (
                (df['lower_shadow'] > 2 * df['body_size']) &
                (df['upper_shadow'] < 0.2 * df['body_size']) &
                (df['body_size'] > 0)
            )
            df.loc[hammer_condition & (df['is_bullish'] == 1), 'pattern_hammer'] = 1
            df.loc[hammer_condition & (df['is_bearish'] == 1), 'pattern_hanging_man'] = 1
            
            # Detect Inverted Hammer and Shooting Star
            inv_hammer_condition = (
                (df['upper_shadow'] > 2 * df['body_size']) &
                (df['lower_shadow'] < 0.2 * df['body_size']) &
                (df['body_size'] > 0)
            )
            df.loc[inv_hammer_condition & (df['is_bullish'] == 1), 'pattern_inverted_hammer'] = 1
            df.loc[inv_hammer_condition & (df['is_bearish'] == 1), 'pattern_shooting_star'] = 1
            
            # Detect Engulfing patterns
            bullish_engulfing = (
                (df['is_bullish'] == 1) &
                (df['is_bearish'].shift(1) == 1) &
                (df['open'] < df['close'].shift(1)) &
                (df['close'] > df['open'].shift(1))
            )
            df.loc[bullish_engulfing, 'pattern_engulfing_bullish'] = 1
            
            bearish_engulfing = (
                (df['is_bearish'] == 1) &
                (df['is_bullish'].shift(1) == 1) &
                (df['open'] > df['close'].shift(1)) &
                (df['close'] < df['open'].shift(1))
            )
            df.loc[bearish_engulfing, 'pattern_engulfing_bearish'] = 1
            
            # Detect Marubozu (strong trend candles)
            marubozu_condition = (
                (df['body_pct'] > 0.9) &
                (df['upper_shadow'] < 0.05 * df['candle_range']) &
                (df['lower_shadow'] < 0.05 * df['candle_range'])
            )
            df.loc[marubozu_condition, 'pattern_marubozu'] = 1
            
            # Detect Three Bar Reversal
            # Downtrend followed by reversal
            three_bar_bullish = (
                (df['close'].shift(2) < df['open'].shift(2)) &  # First bar bearish
                (df['close'].shift(1) < df['open'].shift(1)) &  # Second bar bearish
                (df['close'] > df['open']) &                    # Third bar bullish
                (df['close'] > df['open'].shift(1))             # Close above previous open
            )
            df.loc[three_bar_bullish, 'pattern_three_bar_reversal'] = 1
            
            # Uptrend followed by reversal
            three_bar_bearish = (
                (df['close'].shift(2) > df['open'].shift(2)) &  # First bar bullish
                (df['close'].shift(1) > df['open'].shift(1)) &  # Second bar bullish
                (df['close'] < df['open']) &                    # Third bar bearish
                (df['close'] < df['open'].shift(1))             # Close below previous open
            )
            df.loc[three_bar_bearish, 'pattern_three_bar_reversal'] = -1
            
            # Detect Pin Bar (price rejection)
            pin_bar_bullish = (
                (df['lower_shadow'] > 2 * df['body_size']) &
                (df['lower_shadow'] > df['upper_shadow'] * 3) &
                (df['body_size'] > 0)
            )
            df.loc[pin_bar_bullish, 'pattern_pin_bar'] = 1
            
            pin_bar_bearish = (
                (df['upper_shadow'] > 2 * df['body_size']) &
                (df['upper_shadow'] > df['lower_shadow'] * 3) &
                (df['body_size'] > 0)
            )
            df.loc[pin_bar_bearish, 'pattern_pin_bar'] = -1
            
            # If using talib, add more pattern recognition
            if self.use_talib:
                # Morning Star
                df['pattern_morning_star'] = talib.CDLMORNINGSTAR(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                ) / 100
                
                # Evening Star
                df['pattern_evening_star'] = talib.CDLEVENINGSTAR(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                ) / 100
                
                # Three White Soldiers
                df['pattern_three_white_soldiers'] = talib.CDL3WHITESOLDIERS(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                ) / 100
                
                # Three Black Crows
                df['pattern_three_black_crows'] = talib.CDL3BLACKCROWS(
                    df['open'].values, df['high'].values, 
                    df['low'].values, df['close'].values
                ) / 100
            
            # Aggregate all patterns into a signal column
            df['pattern_signal'] = 0
            
            # Bullish patterns
            bullish_patterns = [
                'pattern_hammer', 'pattern_inverted_hammer', 
                'pattern_engulfing_bullish', 'pattern_morning_star',
                'pattern_three_white_soldiers', 'pattern_piercing_line'
            ]
            
            # Bearish patterns
            bearish_patterns = [
                'pattern_hanging_man', 'pattern_shooting_star',
                'pattern_engulfing_bearish', 'pattern_evening_star',
                'pattern_three_black_crows', 'pattern_dark_cloud_cover'
            ]
            
            # Calculate pattern signals
            for pattern in bullish_patterns:
                if pattern in df.columns:
                    df.loc[df[pattern] > 0, 'pattern_signal'] = 1
            
            for pattern in bearish_patterns:
                if pattern in df.columns:
                    df.loc[df[pattern] > 0, 'pattern_signal'] = -1
            
            # Handle patterns that can be both bullish and bearish
            if 'pattern_pin_bar' in df.columns:
                df.loc[df['pattern_pin_bar'] > 0, 'pattern_signal'] = 1
                df.loc[df['pattern_pin_bar'] < 0, 'pattern_signal'] = -1
            
            if 'pattern_three_bar_reversal' in df.columns:
                df.loc[df['pattern_three_bar_reversal'] > 0, 'pattern_signal'] = 1
                df.loc[df['pattern_three_bar_reversal'] < 0, 'pattern_signal'] = -1
            
            return df
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            raise
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20, strength_threshold: int = 2) -> pd.DataFrame:
        """
        Find support and resistance levels in the dataframe.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            window: Window size for finding local extrema
            strength_threshold: Minimum number of touches to consider a level valid
            
        Returns:
            pd.DataFrame: DataFrame with support and resistance levels added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # Find local maxima and minima
            df['is_local_max'] = 0
            df['is_local_min'] = 0
            
            # Use scipy's argrelextrema to find local extrema
            max_idx = argrelextrema(df['high'].values, np.greater, order=window)[0]
            min_idx = argrelextrema(df['low'].values, np.less, order=window)[0]
            
            df.loc[max_idx, 'is_local_max'] = 1
            df.loc[min_idx, 'is_local_min'] = 1
            
            # Initialize support and resistance columns
            df['support'] = np.nan
            df['resistance'] = np.nan
            
            # Collect all potential levels
            support_levels = df.loc[df['is_local_min'] == 1, 'low'].tolist()
            resistance_levels = df.loc[df['is_local_max'] == 1, 'high'].tolist()
            
            # Group nearby levels (within 0.2% of each other)
            def group_levels(levels, threshold=0.002):
                if not levels:
                    return []
                
                # Sort levels
                sorted_levels = sorted(levels)
                grouped = []
                current_group = [sorted_levels[0]]
                
                for i in range(1, len(sorted_levels)):
                    if (sorted_levels[i] - current_group[0]) / current_group[0] <= threshold:
                        current_group.append(sorted_levels[i])
                    else:
                        # Add average of current group
                        grouped.append(sum(current_group) / len(current_group))
                        current_group = [sorted_levels[i]]
                
                # Add the last group
                if current_group:
                    grouped.append(sum(current_group) / len(current_group))
                
                return grouped
            
            # Group the levels
            grouped_supports = group_levels(support_levels)
            grouped_resistances = group_levels(resistance_levels)
            
            # Count touches for each level
            def count_touches(price_series, level, threshold=0.002):
                # Count how many times price comes within threshold% of level
                touches = sum((abs(price_series - level) / level) <= threshold)
                return touches
            
            # Filter levels by strength
            strong_supports = [level for level in grouped_supports 
                              if count_touches(df['low'], level) >= strength_threshold]
            
            strong_resistances = [level for level in grouped_resistances 
                                 if count_touches(df['high'], level) >= strength_threshold]
            
            # Add levels to dataframe
            # For each candle, find the nearest strong support and resistance
            for i in range(len(df)):
                if strong_supports:
                    # Find nearest support below current price
                    valid_supports = [s for s in strong_supports if s < df.iloc[i]['close']]
                    if valid_supports:
                        df.loc[df.index[i], 'support'] = max(valid_supports)
                
                if strong_resistances:
                    # Find nearest resistance above current price
                    valid_resistances = [r for r in strong_resistances if r > df.iloc[i]['close']]
                    if valid_resistances:
                        df.loc[df.index[i], 'resistance'] = min(valid_resistances)
            
            # Calculate distance to nearest support/resistance as percentage
            df['support_distance_pct'] = np.where(
                df['support'].notna(),
                (df['close'] - df['support']) / df['close'] * 100,
                np.nan
            )
            
            df['resistance_distance_pct'] = np.where(
                df['resistance'].notna(),
                (df['resistance'] - df['close']) / df['close'] * 100,
                np.nan
            )
            
            # Identify if price is testing support or resistance
            df['at_support'] = np.where(
                df['support'].notna() & (df['support_distance_pct'].abs() < 0.2),
                1, 0
            )
            
            df['at_resistance'] = np.where(
                df['resistance'].notna() & (df['resistance_distance_pct'].abs() < 0.2),
                1, 0
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error finding support/resistance levels: {str(e)}")
            raise
    
    def add_market_structure(self, df: pd.DataFrame, swing_period: int = 10) -> pd.DataFrame:
        """
        Analyze market structure and identify trends, ranges, and potential reversals.
        
        Args:
            df: Pandas DataFrame with OHLCV data
            swing_period: Period for identifying swing highs and lows
            
        Returns:
            pd.DataFrame: DataFrame with market structure analysis added
        """
        try:
            self.validate_dataframe(df)
            df = df.copy()
            
            # Find swing highs and lows
            df['swing_high'] = 0
            df['swing_low'] = 0
            
            for i in range(swing_period, len(df) - swing_period):
                # Check if this is a swing high
                if all(df.iloc[i]['high'] > df.iloc[i-j]['high'] for j in range(1, swing_period+1)) and \
                   all(df.iloc[i]['high'] > df.iloc[i+j]['high'] for j in range(1, swing_period+1)):
                    df.iloc[i, df.columns.get_loc('swing_high')] = 1
                
                # Check if this is a swing low
                if all(df.iloc[i]['low'] < df.iloc[i-j]['low'] for j in range(1, swing_period+1)) and \
                   all(df.iloc[i]['low'] < df.iloc[i+j]['low'] for j in range(1, swing_period+1)):
                    df.iloc[i, df.columns.get_loc('swing_low')] = 1
            
            # Initialize market structure columns
            df['market_structure'] = None
            df['trend_direction'] = 0
            df['in_range'] = 0
            df['breakout_direction'] = 0
            
            # Analyze higher highs (HH), higher lows (HL), lower highs (LH), lower lows (LL)
            # Get swing high and low points
            swing_high_points = df[df['swing_high'] == 1].copy()
            swing_low_points = df[df['swing_low'] == 1].copy()
            
            if len(swing_high_points) >= 2 and len(swing_low_points) >= 2:
                # Check for higher highs and higher lows (uptrend)
                higher_highs = swing_high_points['high'].is_monotonic_increasing
                higher_lows = swing_low_points['low'].is_monotonic_increasing
                
                # Check for lower highs and lower lows (downtrend)
                lower_highs = swing_high_points['high'].is_monotonic_decreasing
                lower_lows = swing_low_points['low'].is_monotonic_decreasing
                
                # Determine trend
                if higher_highs and higher_lows:
                    df['market_structure'] = MarketStructure.UPTREND.value
                    df['trend_direction'] = 1
                elif lower_highs and lower_lows:
                    df['market_structure'] = MarketStructure.DOWNTREND.value
                    df['trend_direction'] = -1
                else:
                    # Check for range
                    recent_swing_highs = swing_high_points.iloc[-3:]['high'] if len(swing_high_points) >= 3 else swing_high_points['high']
                    recent_swing_lows = swing_low_points.iloc[-3:]['low'] if len(swing_low_points) >= 3 else swing_low_points['low']
                    
                    high_range = recent_swing_highs.max() - recent_swing_highs.min()
                    low_range = recent_swing_lows.max() - recent_swing_lows.min()
                    
                    if high_range / recent_swing_highs.mean() < 0.03 and low_range / recent_swing_lows.mean() < 0.03:
                        df['market_structure'] = MarketStructure.RANGING.value
                        df['in_range'] = 1
            
            # Detect potential breakouts
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                # Breakout above Bollinger Band
                breakout_up = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
                df.loc[breakout_up, 'market_structure'] = MarketStructure.BREAKOUT.value
                df.loc[breakout_up, 'breakout_direction'] = 1
                
                # Breakdown below Bollinger Band
                breakout_down = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
                df.loc[breakout_down, 'market_structure'] = MarketStructure.BREAKDOWN.value
                df.loc[breakout_down, 'breakout_direction'] = -1
            
            # Detect potential reversals
            if 'rsi_14' in df.columns:
                # Bullish divergence (price makes lower low but RSI makes higher low)
                if len(swing_low_points) >= 2:
                    for i in range(1, len(swing_low_points)):
                        curr_idx = swing_low_points.index[i]
                        prev_idx = swing_low_points.index[i-1]
                        
                        if swing_low_points.loc[curr_idx, 'low'] < swing_low_points.loc[prev_idx, 'low'] and \
                           df.loc[curr_idx, 'rsi_14'] > df.loc[prev_idx, 'rsi_14']:
                            df.loc[curr_idx:, 'market_structure'] = MarketStructure.REVERSAL_POTENTIAL.value
                
                # Bearish divergence (price makes higher high but RSI makes lower high)
                if len(swing_high_points) >= 2:
                    for i in range(1, len(swing_high_points)):
                        curr_idx = swing_high_points.index[i]
                        prev_idx = swing_high_points.index[i-1]
                        
                        if swing_high_points.loc[curr_idx, 'high'] > swing_high_points.loc[prev_idx, 'high'] and \
                           df.loc[curr_idx, 'rsi_14'] < df.loc[prev_idx, 'rsi_14']:
                            df.loc[curr_idx:, 'market_structure'] = MarketStructure.REVERSAL_POTENTIAL.value
            
            # Fill NaN values with previous state
            df['market_structure'] = df['market_structure'].fillna(method='ffill')
            
            # Default to ranging if still NaN
            df['market_structure'] = df['market_structure'].fillna(MarketStructure.RANGING.value)
            
            # Add market structure change signals
            df['structure_changed'] = df['market_structure'] != df['market_structure'].shift(1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            raise
    
    def generate_signals(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals based on the indicators and configuration.
        
        Args:
            df: Pandas DataFrame with indicators
            config: Configuration dictionary with strategy settings
            
        Returns:
            pd.DataFrame: DataFrame with trading signals added
        """
        try:
            df = df.copy()
            
            # Initialize signal columns
            df['entry_signal'] = 0  # 1 for buy, -1 for sell
            df['exit_signal'] = 0   # 1 for exit buy, -1 for exit sell
            df['signal_strength'] = 0.0  # 0.0 to 1.0
            
            # Get strategy configuration
            require_all_conditions = config['strategy']['entry'].get('require_all_conditions', True)
            
            # Create individual condition columns
            conditions = {}
            
            # EMA Crossover condition
            if config['strategy']['entry']['ema_crossover']['enabled']:
                fast_period = config['strategy']['entry']['ema_crossover']['fast_period']
                slow_period = config['strategy']['entry']['ema_crossover']['slow_period']
                
                # Ensure we have the necessary columns
                if f'ema_{fast_period}' not in df.columns or f'ema_{slow_period}' not in df.columns:
                    df = self.add_ema_crossover_signals(df, fast_period, slow_period)
                
                conditions['ema_condition'] = df['ema_cross_signal']
            
            # RSI condition
            if config['strategy']['entry']['rsi']['enabled']:
                period = config['strategy']['entry']['rsi']['period']
                lower_bound = config['strategy']['entry']['rsi']['lower_bound']
                upper_bound = config['strategy']['entry']['rsi']['upper_bound']
                
                rsi_col = f'rsi_{period}'
                if rsi_col not in df.columns:
                    df = self.add_rsi(df, period)
                
                # Buy when RSI is between lower and upper bounds
                conditions['rsi_condition'] = 0
                df.loc[(df[rsi_col] >= lower_bound) & (df[rsi_col] <= upper_bound), 'rsi_condition'] = 1
            
            # Bollinger Bands condition
            if config['strategy']['entry']['bollinger_bands']['enabled'] and config['strategy']['entry']['bollinger_bands']['use_for_entry']:
                if 'bb_touch_lower' not in df.columns or 'bb_touch_upper' not in df.columns:
                    period = config['strategy']['entry']['bollinger_bands']['period']
                    std_dev = config['strategy']['entry']['bollinger_bands']['std_dev']
                    df = self.add_bollinger_bands(df, period, std_dev)
                
                conditions['bb_condition'] = 0
                df.loc[df['bb_touch_lower'] == 1, 'bb_condition'] = 1  # Buy signal
                df.loc[df['bb_touch_upper'] == 1, 'bb_condition'] = -1  # Sell signal
            
            # VWAP condition
            if config['strategy']['entry']['vwap']['enabled']:
                if 'vwap_cross_signal' not in df.columns:
                    df = self.add_vwap(df)
                
                conditions['vwap_condition'] = df['vwap_cross_signal']
            
            # Volume condition
            if config['strategy']['entry']['volume']['enabled']:
                spike_multiplier = config['strategy']['entry']['volume']['spike_multiplier']
                
                if 'volume_ratio' not in df.columns:
                    df = self.add_volume_indicators(df)
                
                conditions['volume_condition'] = 0
                df.loc[df['volume_ratio'] >= spike_multiplier, 'volume_condition'] = 1
            
            # Candlestick Patterns condition
            if config['strategy']['entry']['candlestick_patterns']['enabled']:
                if 'pattern_signal' not in df.columns:
                    df = self.detect_candlestick_patterns(df)
                
                conditions['pattern_condition'] = df['pattern_signal']
            
            # MACD condition (optional)
            if config['strategy']['entry']['macd']['enabled']:
                if 'macd_cross_signal' not in df.columns:
                    fast_period = config['strategy']['entry']['macd']['fast_period']
                    slow_period = config['strategy']['entry']['macd']['slow_period']
                    signal_period = config['strategy']['entry']['macd']['signal_period']
                    df = self.add_macd(df, fast_period, slow_period, signal_period)
                
                conditions['macd_condition'] = df['macd_cross_signal']
            
            # Generate entry signals based on conditions
            if require_all_conditions:
                # All conditions must be met
                df['entry_signal'] = 0
                
                # For buy signals (all conditions must be positive or zero)
                buy_conditions = all(
                    (df[col] >= 0).all() for col in conditions.values() 
                    if isinstance(col, pd.Series)
                )
                
                # For sell signals (all conditions must be negative or zero)
                sell_conditions = all(
                    (df[col] <= 0).all() for col in conditions.values() 
                    if isinstance(col, pd.Series)
                )
                
                # Count how many conditions are met
                df['conditions_met_count'] = 0
                df['total_conditions'] = len(conditions)
                
                for condition in conditions.values():
                    if isinstance(condition, pd.Series):
                        # For buy signals
                        df.loc[condition > 0, 'conditions_met_count'] += 1
                        # For sell signals
                        df.loc[condition < 0, 'conditions_met_count'] -= 1
                
                # Set signal strength based on percentage of conditions met
                df['signal_strength'] = abs(df['conditions_met_count']) / df['total_conditions']
                
                # Set entry signal if all conditions are met
                df.loc[df['conditions_met_count'] >= df['total_conditions'], 'entry_signal'] = 1
                df.loc[df['conditions_met_count'] <= -df['total_conditions'], 'entry_signal'] = -1
                
            else:
                # Weighted approach - count positive and negative signals
                df['buy_score'] = 0
                df['sell_score'] = 0
                
                for condition in conditions.values():
                    if isinstance(condition, pd.Series):
                        df.loc[condition > 0, 'buy_score'] += 1
                        df.loc[condition < 0, 'sell_score'] += 1
                
                # Calculate signal strength as percentage of conditions met
                total_conditions = len(conditions)
                df['signal_strength'] = np.maximum(df['buy_score'], df['sell_score']) / total_conditions
                
                # Generate signals if score exceeds threshold (e.g., 50% of conditions)
                threshold = total_conditions / 2
                df.loc[df['buy_score'] > threshold, 'entry_signal'] = 1
                df.loc[df['sell_score'] > threshold, 'entry_signal'] = -1
            
            # Generate exit signals
            # Time-based exit
            if config['strategy']['exit']['time_stop']['enabled']:
                max_duration = config['strategy']['exit']['time_stop']['max_trade_duration_minutes']
                # This would need to be implemented in the trading logic since it depends on entry time
            
            # Take profit and stop loss exits would be implemented in the trading logic
            
            # VWAP-based exit
            if config['strategy']['exit']['vwap_exit']['enabled']:
                threshold = config['strategy']['exit']['vwap_exit']['threshold']
                
                if 'vwap_distance' not in df.columns:
                    df = self.add_vwap(df)
                
                # Exit long if price breaks below VWAP by threshold
                df.loc[df['vwap_distance'] < -threshold, 'exit_signal'] = 1
                
                # Exit short if price breaks above VWAP by threshold
                df.loc[df['vwap_distance'] > threshold, 'exit_signal'] = -1
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise


def calculate_volatility(df: pd.DataFrame, period: int = 30, use_atr: bool = False) -> float:
    """
    Calculate volatility over a given period.
    
    Args:
        df: Pandas DataFrame with OHLCV data
        period: Period to calculate volatility over
        use_atr: Whether to use ATR for volatility calculation
        
    Returns:
        float: Volatility as a percentage
    """
    try:
        if df.empty or len(df) < period:
            return 0.0
        
        if use_atr:
            # Use ATR for volatility
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            volatility = (atr[-1] / df['close'].iloc[-1]) * 100
        else:
            # Use standard deviation of returns
            returns = df['close'].pct_change().dropna()
            if len(returns) < 2:
                return 0.0
            volatility = returns.tail(period).std() * 100
        
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0.0


def check_market_conditions(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if market conditions meet the pre-filter criteria.
    
    Args:
        df: Pandas DataFrame with OHLCV data
        config: Configuration dictionary with pre-filter settings
        
    Returns:
        Dict: Dictionary with market condition checks and results
    """
    try:
        results = {
            'passed': False,
            'volume_check': False,
            'volatility_check': False,
            'trend_check': False,
            'price_check': False,
            'details': {}
        }
        
        # Check 24h volume
        min_volume = config['strategy']['pre_filter']['min_24h_volume_usd']
        current_volume = df['volume'].sum() * df['close'].iloc[-1]
        results['volume_check'] = current_volume >= min_volume
        results['details']['volume'] = {
            'current': current_volume,
            'required': min_volume
        }
        
        # Check volatility
        min_volatility = config['strategy']['pre_filter']['min_volatility_pct']
        current_volatility = calculate_volatility(df, period=30)
        results['volatility_check'] = current_volatility >= min_volatility
        results['details']['volatility'] = {
            'current': current_volatility,
            'required': min_volatility
        }
        
        # Check price range
        min_price = config['strategy']['pre_filter']['min_price']
        max_price = config['strategy']['pre_filter']['max_price']
        current_price = df['close'].iloc[-1]
        results['price_check'] = min_price <= current_price <= max_price
        results['details']['price'] = {
            'current': current_price,
            'min_required': min_price,
            'max_required': max_price
        }
        
        # Check if all required conditions are met
        results['passed'] = results['volume_check'] and results['volatility_check'] and results['price_check']
        
        return results
        
    except Exception as e:
        logger.error(f"Error checking market conditions: {str(e)}")
        return {'passed': False, 'error': str(e)}


def is_consolidated(df: pd.DataFrame, percentage: float = 2.0, period: int = 20) -> bool:
    """
    Check if the market is consolidated (low volatility range).
    
    Args:
        df: Pandas DataFrame with OHLCV data
        percentage: Maximum percentage range for consolidation
        period: Period to check for consolidation
        
    Returns:
        bool: True if market is consolidated, False otherwise
    """
    try:
        if len(df) < period:
            return False
        
        # Get the last 'period' candles
        recent_df = df.tail(period)
        
        # Calculate the high and low of the range
        high = recent_df['high'].max()
        low = recent_df['low'].min()
        
        # Calculate the percentage range
        price_range = ((high - low) / low) * 100
        
        # Check if the range is less than the specified percentage
        return price_range <= percentage
        
    except Exception as e:
        logger.error(f"Error checking consolidation: {str(e)}")
        return False


def get_trend_direction(df: pd.DataFrame, method: str = 'ema', period: int = 50) -> int:
    """
    Determine the trend direction using various methods.
    
    Args:
        df: Pandas DataFrame with OHLCV data
        method: Method to determine trend ('ema', 'sma', 'linear_regression')
        period: Period to use for trend calculation
        
    Returns:
        int: 1 for uptrend, -1 for downtrend, 0 for no clear trend
    """
    try:
        if len(df) < period:
            return 0
        
        if method == 'ema':
            # Use EMA slope
            ema = df['close'].ewm(span=period, adjust=False).mean()
            slope = (ema.iloc[-1] - ema.iloc[-20]) / 20
            
            if slope > 0:
                return 1
            elif slope < 0:
                return -1
            else:
                return 0
                
        elif method == 'sma':
            # Use SMA slope
            sma = df['close'].rolling(window=period).mean()
            slope = (sma.iloc[-1] - sma.iloc[-20]) / 20
            
            if slope > 0:
                return 1
            elif slope < 0:
                return -1
            else:
                return 0
                
        elif method == 'linear_regression':
            # Use linear regression
            y = df['close'].tail(period).values
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            
            if slope > 0:
                return 1
            elif slope < 0:
                return -1
            else:
                return 0
                
        else:
            logger.warning(f"Unknown trend method: {method}, defaulting to EMA")
            return get_trend_direction(df, method='ema', period=period)
            
    except Exception as e:
        logger.error(f"Error determining trend direction: {str(e)}")
        return 0
