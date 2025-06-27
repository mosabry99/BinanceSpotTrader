#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Module

This module defines trading strategies that combine technical indicators,
risk management, position sizing, and entry/exit logic. It provides a flexible
framework for creating and backtesting trading strategies with multiple
confirmation layers and signal validation.

Key features:
- Base Strategy class for extension
- Multi-indicator strategy implementation
- Risk management integration
- Dynamic position sizing
- Entry and exit rules with confirmation
- Signal validation to reduce false positives
- Strategy performance metrics
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json

# Import local modules
import config
from technical_indicators import (
    TechnicalIndicators, SIGNAL_BUY, SIGNAL_SELL, 
    SIGNAL_NEUTRAL, SIGNAL_STRONG_BUY, SIGNAL_STRONG_SELL,
    get_signal_name
)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO if not config.DEBUG_MODE else logging.DEBUG)


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategy implementations must follow.
    It provides common functionality for strategy initialization, execution,
    and performance tracking.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            parameters: Dictionary of strategy parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        
        # Performance tracking
        self.trades = []
        self.current_position = None
        self.performance_metrics = {}
        
        logger.info(f"Initialized strategy: {self.name}")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added strategy signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        capital: float,
        risk_level: str,
        current_price: float,
        stop_loss_price: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on available capital and risk parameters.
        
        Args:
            capital: Available capital
            risk_level: Risk level (low, medium, high)
            current_price: Current asset price
            stop_loss_price: Stop loss price if available
            
        Returns:
            Position size (quantity to buy/sell)
        """
        pass
    
    def validate_signal(
        self, 
        signal: int,
        df: pd.DataFrame,
        index: int,
        min_confirmation: int = 2
    ) -> bool:
        """
        Validate a trading signal to reduce false positives.
        
        Args:
            signal: Signal value to validate
            df: DataFrame with indicator data
            index: Current index in the DataFrame
            min_confirmation: Minimum number of confirming indicators
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Basic validation - check if we have enough data
        if index < 5 or index >= len(df):
            return False
        
        # Get current row
        row = df.iloc[index]
        
        # Count confirming indicators for buy/sell
        confirming_indicators = 0
        
        # Check various indicator signals
        signal_columns = [col for col in df.columns if col.endswith('_signal')]
        
        for col in signal_columns:
            # Skip the combined signal column
            if col == 'combined_signal':
                continue
                
            # Count indicators that agree with the signal direction
            if (signal > 0 and row[col] > 0) or (signal < 0 and row[col] < 0):
                confirming_indicators += 1
        
        # Check if we have enough confirmation
        return confirming_indicators >= min_confirmation
    
    def calculate_stop_loss(
        self, 
        entry_price: float,
        signal: int,
        df: pd.DataFrame,
        index: int,
        risk_level: str = 'medium'
    ) -> float:
        """
        Calculate stop loss price based on strategy and risk level.
        
        Args:
            entry_price: Entry price
            signal: Signal type (buy/sell)
            df: DataFrame with indicator data
            index: Current index in the DataFrame
            risk_level: Risk level (low, medium, high)
            
        Returns:
            Stop loss price
        """
        # Get risk parameters based on risk level
        risk_params = config.RISK_LEVELS.get(risk_level, config.RISK_LEVELS['medium'])
        stop_loss_pct = risk_params['stop_loss_pct'] / 100
        
        # For ATR-based stop loss, use ATR if available
        if 'atr' in df.columns:
            atr_multiplier = {
                'low': 1.5,
                'medium': 2.0,
                'high': 3.0
            }.get(risk_level, 2.0)
            
            atr_value = df['atr'].iloc[index]
            
            if signal > 0:  # Buy signal
                return entry_price * (1 - stop_loss_pct)
            else:  # Sell signal
                return entry_price * (1 + stop_loss_pct)
        
        # Default percentage-based stop loss
        if signal > 0:  # Buy signal
            return entry_price * (1 - stop_loss_pct)
        else:  # Sell signal
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(
        self, 
        entry_price: float,
        signal: int,
        df: pd.DataFrame,
        index: int,
        risk_level: str = 'medium'
    ) -> float:
        """
        Calculate take profit price based on strategy and risk level.
        
        Args:
            entry_price: Entry price
            signal: Signal type (buy/sell)
            df: DataFrame with indicator data
            index: Current index in the DataFrame
            risk_level: Risk level (low, medium, high)
            
        Returns:
            Take profit price
        """
        # Get risk parameters based on risk level
        risk_params = config.RISK_LEVELS.get(risk_level, config.RISK_LEVELS['medium'])
        take_profit_pct = risk_params['take_profit_pct'] / 100
        
        # For ATR-based take profit, use ATR if available
        if 'atr' in df.columns:
            atr_multiplier = {
                'low': 2.0,
                'medium': 3.0,
                'high': 5.0
            }.get(risk_level, 3.0)
            
            atr_value = df['atr'].iloc[index]
            
            if signal > 0:  # Buy signal
                return entry_price * (1 + take_profit_pct)
            else:  # Sell signal
                return entry_price * (1 - take_profit_pct)
        
        # Default percentage-based take profit
        if signal > 0:  # Buy signal
            return entry_price * (1 + take_profit_pct)
        else:  # Sell signal
            return entry_price * (1 - take_profit_pct)
    
    def should_exit_position(
        self, 
        position: Dict,
        current_price: float,
        df: pd.DataFrame,
        index: int
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.
        
        Args:
            position: Current position information
            current_price: Current asset price
            df: DataFrame with indicator data
            index: Current index in the DataFrame
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if not position:
            return False, ""
        
        # Get current row
        row = df.iloc[index]
        
        # Check stop loss
        if position['direction'] == 'long' and current_price <= position['stop_loss']:
            return True, "stop_loss"
        elif position['direction'] == 'short' and current_price >= position['stop_loss']:
            return True, "stop_loss"
        
        # Check take profit
        if position['direction'] == 'long' and current_price >= position['take_profit']:
            return True, "take_profit"
        elif position['direction'] == 'short' and current_price <= position['take_profit']:
            return True, "take_profit"
        
        # Check trailing stop if activated
        if position.get('trailing_stop_active', False):
            if position['direction'] == 'long' and current_price <= position['trailing_stop']:
                return True, "trailing_stop"
            elif position['direction'] == 'short' and current_price >= position['trailing_stop']:
                return True, "trailing_stop"
        
        # Check for opposite signal
        if (position['direction'] == 'long' and row.get('combined_signal', 0) <= SIGNAL_SELL) or \
           (position['direction'] == 'short' and row.get('combined_signal', 0) >= SIGNAL_BUY):
            return True, "opposite_signal"
        
        # Check time-based exit (if position held for too long)
        if 'max_holding_time' in position and position.get('entry_time'):
            entry_time = position['entry_time']
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            
            current_time = df.index[index]
            holding_time = current_time - entry_time
            
            if holding_time.total_seconds() > position['max_holding_time']:
                return True, "time_exit"
        
        # No exit conditions met
        return False, ""
    
    def update_trailing_stop(
        self, 
        position: Dict,
        current_price: float,
        risk_level: str = 'medium'
    ) -> Dict:
        """
        Update trailing stop for an open position.
        
        Args:
            position: Current position information
            current_price: Current asset price
            risk_level: Risk level (low, medium, high)
            
        Returns:
            Updated position dictionary
        """
        if not position:
            return position
        
        # Get risk parameters
        risk_params = config.RISK_LEVELS.get(risk_level, config.RISK_LEVELS['medium'])
        activation_pct = risk_params['trailing_stop_activation_pct'] / 100
        distance_pct = risk_params['trailing_stop_distance_pct'] / 100
        
        # Check if trailing stop should be activated
        if not position.get('trailing_stop_active', False):
            if position['direction'] == 'long':
                profit_pct = (current_price / position['entry_price']) - 1
                if profit_pct >= activation_pct:
                    position['trailing_stop_active'] = True
                    position['trailing_stop'] = current_price * (1 - distance_pct)
            else:  # short
                profit_pct = 1 - (current_price / position['entry_price'])
                if profit_pct >= activation_pct:
                    position['trailing_stop_active'] = True
                    position['trailing_stop'] = current_price * (1 + distance_pct)
        
        # Update trailing stop if already active
        elif position['trailing_stop_active']:
            if position['direction'] == 'long':
                new_stop = current_price * (1 - distance_pct)
                if new_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_stop
            else:  # short
                new_stop = current_price * (1 + distance_pct)
                if new_stop < position['trailing_stop']:
                    position['trailing_stop'] = new_stop
        
        return position
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics based on completed trades.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_profit": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_loss": 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['profit_pct'] > 0]
        losing_trades = [t for t in self.trades if t['profit_pct'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = sum(t['profit'] for t in winning_trades)
        total_loss = abs(sum(t['profit'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_profit = sum(t['profit'] for t in self.trades) / total_trades if total_trades > 0 else 0
        average_profit_pct = sum(t['profit_pct'] for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Calculate drawdown
        equity_curve = [0]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade['profit'])
        
        max_equity = 0
        max_drawdown = 0
        
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = max_equity - equity
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['profit_pct'] for t in self.trades]
        mean_return = sum(returns) / len(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 1
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Calculate total profit/loss
        profit_loss = sum(t['profit'] for t in self.trades)
        profit_loss_pct = sum(t['profit_pct'] for t in self.trades)
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_profit": average_profit,
            "average_profit_pct": average_profit_pct,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown / max_equity if max_equity > 0 else 0,
            "sharpe_ratio": sharpe_ratio,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
            "average_holding_time": sum(t.get('holding_time_seconds', 0) for t in self.trades) / total_trades if total_trades > 0 else 0,
            "long_trades": len([t for t in self.trades if t['direction'] == 'long']),
            "short_trades": len([t for t in self.trades if t['direction'] == 'short']),
        }
    
    def to_json(self) -> str:
        """
        Convert strategy to JSON string.
        
        Returns:
            JSON string representation of the strategy
        """
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics,
            "trades_count": len(self.trades)
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Strategy':
        """
        Create strategy from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Strategy instance
        """
        data = json.loads(json_str)
        strategy = cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"]
        )
        strategy.performance_metrics = data.get("performance_metrics", {})
        return strategy


class MultiIndicatorStrategy(Strategy):
    """
    Multi-Indicator Strategy that combines multiple technical indicators
    with confirmation layers, trend analysis, and adaptive position sizing.
    
    This strategy uses a weighted combination of indicators with multiple
    confirmation layers to generate trading signals. It includes risk management,
    dynamic position sizing, and customizable parameters.
    """
    
    def __init__(
        self,
        name: str = "Multi-Indicator Strategy",
        description: str = "A strategy combining multiple technical indicators with confirmation layers",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Multi-Indicator Strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            parameters: Dictionary of strategy parameters
        """
        # Default parameters
        default_params = {
            "indicators": {
                "rsi": {"enabled": True, "weight": 1.0},
                "macd": {"enabled": True, "weight": 1.0},
                "bollinger_bands": {"enabled": True, "weight": 0.8},
                "ema_crossover": {"enabled": True, "weight": 1.0},
                "stochastic": {"enabled": True, "weight": 0.7},
                "volume_profile": {"enabled": True, "weight": 0.8},
                "adx": {"enabled": True, "weight": 0.8},
                "parabolic_sar": {"enabled": True, "weight": 0.7},
                "support_resistance": {"enabled": True, "weight": 0.9},
                "fibonacci": {"enabled": False, "weight": 0.6},
                "ichimoku": {"enabled": False, "weight": 0.6}
            },
            "signal_threshold": 0.7,
            "confirmation_needed": 2,
            "trend_confirmation": True,
            "use_market_sentiment": True,
            "exit_on_opposite_signal": True,
            "use_trailing_stop": True,
            "enable_dynamic_sizing": True,
            "max_holding_time_hours": 72,  # 3 days max holding time
        }
        
        # Override defaults with provided parameters
        if parameters:
            for key, value in parameters.items():
                if key == "indicators" and isinstance(value, dict):
                    for ind_key, ind_value in value.items():
                        if ind_key in default_params["indicators"]:
                            default_params["indicators"][ind_key].update(ind_value)
                else:
                    default_params[key] = value
        
        super().__init__(name, description, default_params)
        
        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        
        logger.info(f"Initialized {self.name} with {len([i for i, v in self.parameters['indicators'].items() if v['enabled']])} active indicators")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the multi-indicator strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added strategy signals
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return df
        
        # Get enabled indicators
        enabled_indicators = [
            ind for ind, config in self.parameters['indicators'].items() 
            if config['enabled']
        ]
        
        # Apply all enabled indicators
        df = self.indicators.apply_all_indicators(df, include=enabled_indicators)
        
        # Get indicator weights for signal generation
        indicator_weights = {
            f"{ind}_signal": config["weight"]
            for ind, config in self.parameters['indicators'].items()
            if config['enabled'] and f"{ind}_signal" in df.columns
        }
        
        # Generate combined signal
        df = self.indicators.generate_combined_signal(
            df,
            indicators=indicator_weights,
            threshold=self.parameters['signal_threshold'],
            confirmation_needed=self.parameters['confirmation_needed']
        )
        
        # Add trend confirmation if enabled
        if self.parameters['trend_confirmation']:
            # Add trend based on EMA direction
            if 'ema_50' not in df.columns:
                df = self.indicators.add_moving_averages(
                    df, periods=[50], ma_type='ema', column='close'
                )
            
            df['trend'] = np.where(
                df['close'] > df['ema_50'],
                'uptrend',
                'downtrend'
            )
            
            # Only allow buy signals in uptrend and sell signals in downtrend
            if 'trend' in df.columns:
                df['strategy_signal'] = df['combined_signal']
                
                # Filter out buy signals in downtrend
                df.loc[(df['trend'] == 'downtrend') & 
                       (df['strategy_signal'] > 0), 
                       'strategy_signal'] = SIGNAL_NEUTRAL
                
                # Filter out sell signals in uptrend
                df.loc[(df['trend'] == 'uptrend') & 
                       (df['strategy_signal'] < 0), 
                       'strategy_signal'] = SIGNAL_NEUTRAL
            else:
                df['strategy_signal'] = df['combined_signal']
        else:
            df['strategy_signal'] = df['combined_signal']
        
        # Add market sentiment if enabled
        if self.parameters['use_market_sentiment']:
            # Use volume profile as a simple proxy for market sentiment
            if 'volume_signal' in df.columns:
                # Adjust strategy signal based on volume confirmation
                df.loc[(df['strategy_signal'] > 0) & 
                       (df['volume_signal'] > 0), 
                       'strategy_signal'] = df['strategy_signal'] + 0.5
                
                df.loc[(df['strategy_signal'] < 0) & 
                       (df['volume_signal'] < 0), 
                       'strategy_signal'] = df['strategy_signal'] - 0.5
                
                # Reduce signal strength if volume doesn't confirm
                df.loc[(df['strategy_signal'] > 0) & 
                       (df['volume_signal'] < 0), 
                       'strategy_signal'] = df['strategy_signal'] * 0.5
                
                df.loc[(df['strategy_signal'] < 0) & 
                       (df['volume_signal'] > 0), 
                       'strategy_signal'] = df['strategy_signal'] * 0.5
        
        # Validate signals
        for i in range(len(df)):
            signal = df['strategy_signal'].iloc[i]
            
            if signal != SIGNAL_NEUTRAL:
                if not self.validate_signal(
                    signal, 
                    df, 
                    i, 
                    min_confirmation=self.parameters['confirmation_needed']
                ):
                    df.iloc[i, df.columns.get_loc('strategy_signal')] = SIGNAL_NEUTRAL
        
        # Add entry/exit points
        df['entry_long'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        df['entry_short'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_long'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_short'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        
        return df
    
    def calculate_position_size(
        self, 
        capital: float,
        risk_level: str,
        current_price: float,
        stop_loss_price: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on available capital and risk parameters.
        
        Args:
            capital: Available capital
            risk_level: Risk level (low, medium, high)
            current_price: Current asset price
            stop_loss_price: Stop loss price if available
            
        Returns:
            Position size (quantity to buy/sell)
        """
        # Get risk parameters
        risk_params = config.RISK_LEVELS.get(risk_level, config.RISK_LEVELS['medium'])
        max_position_size_pct = risk_params['max_position_size_pct'] / 100
        
        # Calculate base position size as percentage of capital
        base_position_size = capital * max_position_size_pct
        
        # If dynamic sizing is enabled, adjust based on stop loss
        if self.parameters['enable_dynamic_sizing'] and stop_loss_price is not None:
            # Calculate risk per unit
            if current_price > stop_loss_price:  # Long position
                risk_per_unit = current_price - stop_loss_price
            else:  # Short position
                risk_per_unit = stop_loss_price - current_price
            
            # Calculate maximum risk amount (1% of capital by default)
            max_risk_amount = capital * 0.01
            
            # Calculate position size based on risk
            if risk_per_unit > 0:
                risk_based_size = max_risk_amount / risk_per_unit
                # Use the smaller of the two position sizes
                position_size = min(base_position_size / current_price, risk_based_size)
            else:
                position_size = base_position_size / current_price
        else:
            # Simple position sizing
            position_size = base_position_size / current_price
        
        return position_size
    
    def backtest(
        self, 
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
        risk_level: str = 'medium',
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest the strategy on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            initial_capital: Initial capital for backtesting
            risk_level: Risk level (low, medium, high)
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            
        Returns:
            Tuple of (DataFrame with backtest results, performance metrics)
        """
        # Generate signals
        df = self.generate_signals(df)
        
        # Initialize backtest variables
        capital = initial_capital
        position = None
        self.trades = []
        
        # Add columns for backtest results
        df['capital'] = initial_capital
        df['position'] = None
        df['position_size'] = 0.0
        df['position_value'] = 0.0
        df['equity'] = initial_capital
        
        # Convert commission and slippage to decimals
        commission = commission_pct / 100
        slippage = slippage_pct / 100
        
        # Run backtest
        for i in range(1, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            current_signal = df['strategy_signal'].iloc[i]
            
            # Copy previous values
            df.iloc[i, df.columns.get_loc('capital')] = capital
            df.iloc[i, df.columns.get_loc('position')] = position['direction'] if position else None
            df.iloc[i, df.columns.get_loc('position_size')] = position['size'] if position else 0.0
            
            # Check if we need to exit an existing position
            if position:
                # Update position value
                position_value = position['size'] * current_price
                df.iloc[i, df.columns.get_loc('position_value')] = position_value
                
                # Update trailing stop if enabled
                if self.parameters['use_trailing_stop']:
                    position = self.update_trailing_stop(position, current_price, risk_level)
                
                # Check exit conditions
                should_exit, exit_reason = self.should_exit_position(
                    position, current_price, df, i
                )
                
                if should_exit:
                    # Calculate exit details
                    exit_price = current_price
                    
                    # Apply slippage
                    if position['direction'] == 'long':
                        exit_price *= (1 - slippage)
                    else:  # short
                        exit_price *= (1 + slippage)
                    
                    # Calculate profit/loss
                    if position['direction'] == 'long':
                        profit = position['size'] * (exit_price - position['entry_price'])
                        profit_pct = (exit_price / position['entry_price']) - 1
                    else:  # short
                        profit = position['size'] * (position['entry_price'] - exit_price)
                        profit_pct = 1 - (exit_price / position['entry_price'])
                    
                    # Apply commission
                    commission_cost = position['size'] * exit_price * commission
                    profit -= commission_cost
                    
                    # Update capital
                    capital += position['size'] * exit_price + profit - commission_cost
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'size': position['size'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'exit_reason': exit_reason,
                        'holding_time_seconds': (current_time - pd.to_datetime(position['entry_time'])).total_seconds()
                    }
                    self.trades.append(trade)
                    
                    # Clear position
                    position = None
                    df.iloc[i, df.columns.get_loc('position')] = None
                    df.iloc[i, df.columns.get_loc('position_size')] = 0.0
                    df.iloc[i, df.columns.get_loc('position_value')] = 0.0
                    df.iloc[i, df.columns.get_loc('capital')] = capital
            
            # Check if we should enter a new position
            if position is None:
                # Check entry conditions
                entry_long = df['entry_long'].iloc[i]
                entry_short = df['entry_short'].iloc[i]
                
                if entry_long or entry_short:
                    # Validate the signal
                    signal = SIGNAL_BUY if entry_long else SIGNAL_SELL
                    
                    if self.validate_signal(signal, df, i, min_confirmation=self.parameters['confirmation_needed']):
                        # Calculate position size
                        stop_loss_price = self.calculate_stop_loss(
                            current_price, signal, df, i, risk_level
                        )
                        
                        position_size = self.calculate_position_size(
                            capital, risk_level, current_price, stop_loss_price
                        )
                        
                        # Apply slippage to entry price
                        entry_price = current_price
                        if signal > 0:  # Buy
                            entry_price *= (1 + slippage)
                        else:  # Sell
                            entry_price *= (1 - slippage)
                        
                        # Calculate entry cost with commission
                        entry_cost = position_size * entry_price
                        commission_cost = entry_cost * commission
                        
                        # Check if we have enough capital
                        if entry_cost + commission_cost <= capital:
                            # Create position
                            position = {
                                'direction': 'long' if signal > 0 else 'short',
                                'size': position_size,
                                'entry_price': entry_price,
                                'entry_time': current_time,
                                'stop_loss': stop_loss_price,
                                'take_profit': self.calculate_take_profit(
                                    entry_price, signal, df, i, risk_level
                                ),
                                'trailing_stop_active': False,
                                'max_holding_time': self.parameters['max_holding_time_hours'] * 3600
                            }
                            
                            # Update capital
                            capital -= entry_cost + commission_cost
                            
                            # Update DataFrame
                            df.iloc[i, df.columns.get_loc('position')] = position['direction']
                            df.iloc[i, df.columns.get_loc('position_size')] = position['size']
                            df.iloc[i, df.columns.get_loc('position_value')] = entry_cost
                            df.iloc[i, df.columns.get_loc('capital')] = capital
            
            # Calculate equity (capital + position value)
            if position:
                position_value = position['size'] * current_price
                equity = capital + position_value
            else:
                equity = capital
            
            df.iloc[i, df.columns.get_loc('equity')] = equity
        
        # Calculate performance metrics
        self.performance_metrics = self.calculate_performance_metrics()
        
        return df, self.performance_metrics
    
    def optimize_parameters(
        self, 
        df: pd.DataFrame,
        parameter_ranges: Dict[str, List],
        initial_capital: float = 10000.0,
        risk_level: str = 'medium',
        optimization_metric: str = 'profit_loss',
        n_iterations: int = 10
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            df: DataFrame with OHLCV data
            parameter_ranges: Dictionary of parameters and their ranges
            initial_capital: Initial capital for backtesting
            risk_level: Risk level (low, medium, high)
            optimization_metric: Metric to optimize for
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimized parameters and results
        """
        import itertools
        from tqdm import tqdm
        
        # Store original parameters
        original_params = self.parameters.copy()
        
        # Generate parameter combinations
        param_keys = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Limit the number of combinations to avoid excessive computation
        combinations = list(itertools.product(*param_values))
        if len(combinations) > n_iterations:
            import random
            combinations = random.sample(combinations, n_iterations)
        
        # Track best parameters and results
        best_metric_value = float('-inf')
        best_params = None
        best_metrics = None
        all_results = []
        
        # Run optimization
        for params_tuple in tqdm(combinations, desc="Optimizing parameters"):
            # Update parameters
            params = self.parameters.copy()
            for i, key in enumerate(param_keys):
                # Handle nested parameters
                if '.' in key:
                    main_key, sub_key = key.split('.', 1)
                    if main_key in params and isinstance(params[main_key], dict):
                        params[main_key][sub_key] = params_tuple[i]
                else:
                    params[key] = params_tuple[i]
            
            # Update strategy parameters
            self.parameters = params
            
            # Run backtest
            _, metrics = self.backtest(
                df.copy(),
                initial_capital=initial_capital,
                risk_level=risk_level
            )
            
            # Track results
            result = {
                'parameters': params.copy(),
                'metrics': metrics
            }
            all_results.append(result)
            
            # Check if this is the best result
            metric_value = metrics.get(optimization_metric, float('-inf'))
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params.copy()
                best_metrics = metrics.copy()
        
        # Restore original parameters
        self.parameters = original_params
        
        # Return optimization results
        return {
            'best_parameters': best_params,
            'best_metrics': best_metrics,
            'best_metric_value': best_metric_value,
            'optimization_metric': optimization_metric,
            'all_results': all_results
        }


class TrendFollowingStrategy(MultiIndicatorStrategy):
    """
    Trend Following Strategy that focuses on identifying and following
    established market trends.
    
    This strategy uses trend indicators like moving averages, ADX, and
    Parabolic SAR to identify trending markets and enter positions in
    the direction of the trend.
    """
    
    def __init__(
        self,
        name: str = "Trend Following Strategy",
        description: str = "A strategy focused on identifying and following established market trends",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Trend Following Strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            parameters: Dictionary of strategy parameters
        """
        # Default parameters for trend following
        default_params = {
            "indicators": {
                "rsi": {"enabled": True, "weight": 0.7},
                "macd": {"enabled": True, "weight": 1.0},
                "bollinger_bands": {"enabled": True, "weight": 0.6},
                "ema_crossover": {"enabled": True, "weight": 1.0},
                "adx": {"enabled": True, "weight": 1.0},
                "parabolic_sar": {"enabled": True, "weight": 0.8},
                "ichimoku": {"enabled": True, "weight": 0.9},
                "volume_profile": {"enabled": True, "weight": 0.7},
                "stochastic": {"enabled": False, "weight": 0.5},
                "support_resistance": {"enabled": False, "weight": 0.6},
                "fibonacci": {"enabled": False, "weight": 0.5}
            },
            "signal_threshold": 0.7,
            "confirmation_needed": 3,
            "trend_confirmation": True,
            "use_market_sentiment": True,
            "exit_on_opposite_signal": True,
            "use_trailing_stop": True,
            "enable_dynamic_sizing": True,
            "max_holding_time_hours": 120,  # 5 days max holding time for trend following
            "min_adx_value": 25,  # Minimum ADX value to confirm trend
            "trend_ema_period": 50,  # EMA period for trend determination
        }
        
        # Override defaults with provided parameters
        if parameters:
            for key, value in parameters.items():
                if key == "indicators" and isinstance(value, dict):
                    for ind_key, ind_value in value.items():
                        if ind_key in default_params["indicators"]:
                            default_params["indicators"][ind_key].update(ind_value)
                else:
                    default_params[key] = value
        
        super().__init__(name, description, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on trend following strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added strategy signals
        """
        # Apply standard multi-indicator strategy
        df = super().generate_signals(df)
        
        # Add additional trend filters
        if 'adx' in df.columns:
            # Only trade when ADX indicates a strong trend
            min_adx = self.parameters.get('min_adx_value', 25)
            df.loc[df['adx'] < min_adx, 'strategy_signal'] = SIGNAL_NEUTRAL
        
        # Use longer-term EMA for trend direction
        trend_period = self.parameters.get('trend_ema_period', 50)
        ema_col = f'ema_{trend_period}'
        
        if ema_col not in df.columns:
            df = self.indicators.add_moving_averages(
                df, periods=[trend_period], ma_type='ema', column='close'
            )
        
        # Only allow trades in the direction of the longer-term trend
        if ema_col in df.columns:
            df.loc[(df['close'] < df[ema_col]) & 
                   (df['strategy_signal'] > 0), 
                   'strategy_signal'] = SIGNAL_NEUTRAL
            
            df.loc[(df['close'] > df[ema_col]) & 
                   (df['strategy_signal'] < 0), 
                   'strategy_signal'] = SIGNAL_NEUTRAL
        
        # Update entry/exit signals
        df['entry_long'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        df['entry_short'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_long'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_short'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        
        return df


class MeanReversionStrategy(MultiIndicatorStrategy):
    """
    Mean Reversion Strategy that focuses on identifying overbought and oversold
    conditions and trading counter to the trend.
    
    This strategy uses oscillators like RSI, Stochastic, and Bollinger Bands
    to identify potential reversal points in the market.
    """
    
    def __init__(
        self,
        name: str = "Mean Reversion Strategy",
        description: str = "A strategy focused on trading overbought and oversold conditions",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Mean Reversion Strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            parameters: Dictionary of strategy parameters
        """
        # Default parameters for mean reversion
        default_params = {
            "indicators": {
                "rsi": {"enabled": True, "weight": 1.0},
                "stochastic": {"enabled": True, "weight": 1.0},
                "bollinger_bands": {"enabled": True, "weight": 1.0},
                "macd": {"enabled": True, "weight": 0.7},
                "ema_crossover": {"enabled": False, "weight": 0.6},
                "adx": {"enabled": True, "weight": 0.8},
                "parabolic_sar": {"enabled": False, "weight": 0.5},
                "support_resistance": {"enabled": True, "weight": 0.9},
                "volume_profile": {"enabled": True, "weight": 0.8},
                "fibonacci": {"enabled": True, "weight": 0.7},
                "ichimoku": {"enabled": False, "weight": 0.5}
            },
            "signal_threshold": 0.7,
            "confirmation_needed": 2,
            "trend_confirmation": False,  # We're trading counter-trend
            "use_market_sentiment": True,
            "exit_on_opposite_signal": True,
            "use_trailing_stop": False,  # Mean reversion often has clear targets
            "enable_dynamic_sizing": True,
            "max_holding_time_hours": 48,  # 2 days max holding time for mean reversion
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_threshold": 0.05,  # How close to the bands to trigger
            "volatility_filter": True,  # Only trade in certain volatility conditions
        }
        
        # Override defaults with provided parameters
        if parameters:
            for key, value in parameters.items():
                if key == "indicators" and isinstance(value, dict):
                    for ind_key, ind_value in value.items():
                        if ind_key in default_params["indicators"]:
                            default_params["indicators"][ind_key].update(ind_value)
                else:
                    default_params[key] = value
        
        super().__init__(name, description, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added strategy signals
        """
        # Apply standard multi-indicator strategy
        df = super().generate_signals(df)
        
        # Add specific mean reversion filters
        
        # RSI filter
        if 'rsi' in df.columns:
            rsi_oversold = self.parameters.get('rsi_oversold', 30)
            rsi_overbought = self.parameters.get('rsi_overbought', 70)
            
            # Strengthen buy signals when RSI is oversold
            df.loc[(df['strategy_signal'] > 0) & 
                   (df['rsi'] < rsi_oversold), 
                   'strategy_signal'] = SIGNAL_STRONG_BUY
            
            # Strengthen sell signals when RSI is overbought
            df.loc[(df['strategy_signal'] < 0) & 
                   (df['rsi'] > rsi_overbought), 
                   'strategy_signal'] = SIGNAL_STRONG_SELL
            
            # Filter out signals that don't align with RSI
            df.loc[(df['strategy_signal'] > 0) & 
                   (df['rsi'] > 50), 
                   'strategy_signal'] = SIGNAL_NEUTRAL
            
            df.loc[(df['strategy_signal'] < 0) & 
                   (df['rsi'] < 50), 
                   'strategy_signal'] = SIGNAL_NEUTRAL
        
        # Bollinger Bands filter
        if all(x in df.columns for x in ['bb_upper', 'bb_lower', 'bb_percent_b']):
            bb_threshold = self.parameters.get('bb_threshold', 0.05)
            
            # Strengthen buy signals when price is near lower band
            df.loc[(df['strategy_signal'] > 0) & 
                   (df['bb_percent_b'] < bb_threshold), 
                   'strategy_signal'] = SIGNAL_STRONG_BUY
            
            # Strengthen sell signals when price is near upper band
            df.loc[(df['strategy_signal'] < 0) & 
                   (df['bb_percent_b'] > 1 - bb_threshold), 
                   'strategy_signal'] = SIGNAL_STRONG_SELL
        
        # Volatility filter
        if self.parameters.get('volatility_filter', True) and 'atr_percent' in df.columns:
            # Calculate median ATR percentage
            median_atr_pct = df['atr_percent'].median()
            
            # Only trade when volatility is appropriate for mean reversion
            # Too low volatility = not enough movement to profit
            # Too high volatility = too risky for mean reversion
            df.loc[df['atr_percent'] < median_atr_pct * 0.5, 'strategy_signal'] = SIGNAL_NEUTRAL
            df.loc[df['atr_percent'] > median_atr_pct * 2.0, 'strategy_signal'] = SIGNAL_NEUTRAL
        
        # Update entry/exit signals
        df['entry_long'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        df['entry_short'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_long'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_short'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        
        return df


class BreakoutStrategy(MultiIndicatorStrategy):
    """
    Breakout Strategy that focuses on identifying and trading price breakouts
    from consolidation patterns, support/resistance levels, or chart patterns.
    
    This strategy uses volatility indicators, volume analysis, and support/resistance
    to identify potential breakout opportunities.
    """
    
    def __init__(
        self,
        name: str = "Breakout Strategy",
        description: str = "A strategy focused on trading price breakouts with volume confirmation",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Breakout Strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            parameters: Dictionary of strategy parameters
        """
        # Default parameters for breakout strategy
        default_params = {
            "indicators": {
                "bollinger_bands": {"enabled": True, "weight": 1.0},
                "volume_profile": {"enabled": True, "weight": 1.0},
                "support_resistance": {"enabled": True, "weight": 1.0},
                "adx": {"enabled": True, "weight": 0.8},
                "atr": {"enabled": True, "weight": 0.8},
                "rsi": {"enabled": True, "weight": 0.7},
                "macd": {"enabled": True, "weight": 0.7},
                "ema_crossover": {"enabled": True, "weight": 0.6},
                "stochastic": {"enabled": False, "weight": 0.5},
                "parabolic_sar": {"enabled": False, "weight": 0.5},
                "fibonacci": {"enabled": False, "weight": 0.5},
                "ichimoku": {"enabled": False, "weight": 0.5}
            },
            "signal_threshold": 0.7,
            "confirmation_needed": 2,
            "trend_confirmation": True,
            "use_market_sentiment": True,
            "exit_on_opposite_signal": True,
            "use_trailing_stop": True,
            "enable_dynamic_sizing": True,
            "max_holding_time_hours": 72,  # 3 days max holding time
            "min_volume_increase": 1.5,  # Minimum volume increase for breakout confirmation
            "consolidation_periods": 20,  # Periods to look for consolidation
            "volatility_expansion_threshold": 1.3,  # Volatility expansion threshold
        }
        
        # Override defaults with provided parameters
        if parameters:
            for key, value in parameters.items():
                if key == "indicators" and isinstance(value, dict):
                    for ind_key, ind_value in value.items():
                        if ind_key in default_params["indicators"]:
                            default_params["indicators"][ind_key].update(ind_value)
                else:
                    default_params[key] = value
        
        super().__init__(name, description, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on breakout strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added strategy signals
        """
        # Apply standard multi-indicator strategy
        df = super().generate_signals(df)
        
        # Add specific breakout filters and signals
        
        # Detect consolidation periods (low volatility)
        if 'atr_percent' in df.columns:
            # Calculate rolling standard deviation of ATR
            df['atr_std'] = df['atr_percent'].rolling(
                window=self.parameters.get('consolidation_periods', 20)
            ).std()
            
            # Identify consolidation (low volatility)
            df['is_consolidating'] = df['atr_std'] < df['atr_std'].rolling(
                window=self.parameters.get('consolidation_periods', 20) * 2
            ).mean() * 0.7
            
            # Identify volatility expansion (potential breakout)
            df['volatility_expansion'] = (
                df['atr_percent'] > df['atr_percent'].shift(1) * 
                self.parameters.get('volatility_expansion_threshold', 1.3)
            )
            
            # Strengthen signals on volatility expansion after consolidation
            consolidation_breakout = (
                df['is_consolidating'].shift(1) & 
                df['volatility_expansion']
            )
            
            df.loc[consolidation_breakout & 
                   (df['strategy_signal'] > 0), 
                   'strategy_signal'] = SIGNAL_STRONG_BUY
            
            df.loc[consolidation_breakout & 
                   (df['strategy_signal'] < 0), 
                   'strategy_signal'] = SIGNAL_STRONG_SELL
        
        # Volume confirmation for breakouts
        if 'volume' in df.columns and 'relative_volume' in df.columns:
            min_vol_increase = self.parameters.get('min_volume_increase', 1.5)
            
            # Only confirm breakout signals with increased volume
            df.loc[(df['strategy_signal'] != SIGNAL_NEUTRAL) & 
                   (df['relative_volume'] < min_vol_increase), 
                   'strategy_signal'] = SIGNAL_NEUTRAL
            
            # Strengthen signals with very high volume
            df.loc[(df['strategy_signal'] > 0) & 
                   (df['relative_volume'] > min_vol_increase * 1.5), 
                   'strategy_signal'] = SIGNAL_STRONG_BUY
            
            df.loc[(df['strategy_signal'] < 0) & 
                   (df['relative_volume'] > min_vol_increase * 1.5), 
                   'strategy_signal'] = SIGNAL_STRONG_SELL
        
        # Support/Resistance breakouts
        if all(x in df.columns for x in ['support_level', 'resistance_level']):
            # Detect price breaking above resistance
            resistance_breakout = (
                df['close'] > df['resistance_level']) & (
                df['close'].shift(1) <= df['resistance_level'].shift(1)
            )
            
            # Detect price breaking below support
            support_breakdown = (
                df['close'] < df['support_level']) & (
                df['close'].shift(1) >= df['support_level'].shift(1)
            )
            
            # Generate breakout signals
            df.loc[resistance_breakout, 'strategy_signal'] = SIGNAL_STRONG_BUY
            df.loc[support_breakdown, 'strategy_signal'] = SIGNAL_STRONG_SELL
        
        # Update entry/exit signals
        df['entry_long'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        df['entry_short'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_long'] = (df['strategy_signal'] <= SIGNAL_SELL) & (df['strategy_signal'].shift(1) > SIGNAL_SELL)
        df['exit_short'] = (df['strategy_signal'] >= SIGNAL_BUY) & (df['strategy_signal'].shift(1) < SIGNAL_BUY)
        
        return df


def get_strategy_by_name(
    strategy_name: str,
    parameters: Dict[str, Any] = None
) -> Strategy:
    """
    Factory function to get a strategy instance by name.
    
    Args:
        strategy_name: Name of the strategy
        parameters: Dictionary of strategy parameters
        
    Returns:
        Strategy instance
    """
    strategies = {
        "multi_indicator": MultiIndicatorStrategy,
        "trend_following": TrendFollowingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy
    }
    
    strategy_class = strategies.get(strategy_name.lower().replace(" ", "_"), MultiIndicatorStrategy)
    return strategy_class(parameters=parameters)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from data_fetcher import get_data_fetcher
    
    # Get data
    fetcher = get_data_fetcher()
    df = fetcher.get_historical_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=500
    )
    
    # Create strategy
    strategy = MultiIndicatorStrategy()
    
    # Run backtest
    results, metrics = strategy.backtest(df)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print sample of results
    print("\nBacktest Results Sample:")
    print(results[['close', 'strategy_signal', 'position', 'position_value', 'equity']].tail())
    
    # Print trades
    print(f"\nTotal Trades: {len(strategy.trades)}")
    if strategy.trades:
        print("\nLast 3 trades:")
        for trade in strategy.trades[-3:]:
            print(f"Direction: {trade['direction']}, "
                  f"Entry: {trade['entry_price']:.2f}, "
                  f"Exit: {trade['exit_price']:.2f}, "
                  f"Profit: {trade['profit']:.2f} ({trade['profit_pct']:.2%}), "
                  f"Reason: {trade['exit_reason']}")
