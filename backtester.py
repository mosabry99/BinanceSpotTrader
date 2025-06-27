#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting Engine

This module provides a comprehensive backtesting framework for cryptocurrency trading strategies.
It supports multiple strategies, realistic trading simulation with proper portfolio management,
commission/slippage modeling, and detailed performance analytics and visualization.

Key features:
- Portfolio-level backtesting across multiple assets
- Realistic order execution with slippage and commissions
- Detailed performance metrics and risk analysis
- Monte Carlo simulations for robustness testing
- Optimization framework for parameter tuning
- Visualization of equity curves, drawdowns, and trade distributions
- Trade journal and detailed reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
import json
import os
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass, field, asdict

# Import local modules
import config
from data_fetcher import DataFetcher, get_data_fetcher
from technical_indicators import TechnicalIndicators
from strategy import Strategy, MultiIndicatorStrategy, get_strategy_by_name

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO if not config.DEBUG_MODE else logging.DEBUG)


@dataclass
class TradeResult:
    """Class for storing individual trade results"""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    direction: str = 'long'  # 'long' or 'short'
    size: float = 0.0
    symbol: str = ''
    profit: float = 0.0
    profit_pct: float = 0.0
    exit_reason: str = ''
    holding_time_seconds: float = 0.0
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ''
    trade_id: str = ''
    
    def to_dict(self) -> Dict:
        """Convert trade result to dictionary"""
        return asdict(self)


@dataclass
class PositionInfo:
    """Class for storing active position information"""
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_stop_active: bool = False
    max_holding_time: Optional[float] = None
    strategy_name: str = ''
    trade_id: str = ''
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return asdict(self)


@dataclass
class BacktestResults:
    """Class for storing backtest results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: List[TradeResult] = field(default_factory=list)
    equity_curve: Dict[datetime, float] = field(default_factory=dict)
    drawdowns: Dict[datetime, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert backtest results to dictionary"""
        result = {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "trades_count": len(self.trades),
            "performance_metrics": self.performance_metrics,
            "parameters": self.parameters,
            "symbols": self.symbols
        }
        
        # Convert trades to list of dicts
        result["trades"] = [
            {**t.to_dict(), 
             "entry_time": t.entry_time.isoformat(),
             "exit_time": t.exit_time.isoformat() if t.exit_time else None}
            for t in self.trades
        ]
        
        # Convert equity curve and drawdowns
        result["equity_curve"] = {
            dt.isoformat(): value for dt, value in self.equity_curve.items()
        }
        result["drawdowns"] = {
            dt.isoformat(): value for dt, value in self.drawdowns.items()
        }
        
        return result
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save backtest results to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BacktestResults':
        """Load backtest results from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class Backtester:
    """
    Comprehensive backtesting engine for cryptocurrency trading strategies.
    
    This class provides a framework for backtesting trading strategies with
    realistic order execution, portfolio management, and detailed performance
    analysis.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
        risk_level: str = 'medium',
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize the Backtester.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
            risk_level: Risk level (low, medium, high)
            data_dir: Directory for storing data
            results_dir: Directory for storing results
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100  # Convert to decimal
        self.slippage_pct = slippage_pct / 100  # Convert to decimal
        self.risk_level = risk_level
        
        # Set up directories
        self.data_dir = data_dir or config.DATA_DIR
        self.results_dir = results_dir or config.RESULTS_DIR
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize data fetcher
        self.data_fetcher = get_data_fetcher()
        
        # Initialize performance tracking
        self.reset()
        
        logger.info(f"Initialized Backtester with {initial_capital} initial capital")
    
    def reset(self) -> None:
        """Reset the backtester state"""
        self.capital = self.initial_capital
        self.positions = {}  # symbol -> PositionInfo
        self.trades = []
        self.equity_curve = {}
        self.drawdowns = {}
        self.trade_history = []
        self.performance_metrics = {}
        self.max_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_returns = []
    
    def fetch_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe string
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of candles
            
        Returns:
            Dictionary of DataFrames with historical data
        """
        data = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol} ({timeframe})")
            
            try:
                df = self.data_fetcher.get_historical_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=start_date,
                    end_time=end_date,
                    limit=limit
                )
                
                if df is not None and not df.empty:
                    data[symbol] = df
                    logger.info(f"Fetched {len(df)} candles for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def prepare_data(
        self,
        data: Dict[str, pd.DataFrame],
        indicators: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data by adding technical indicators.
        
        Args:
            data: Dictionary of DataFrames with historical data
            indicators: List of indicators to add
            
        Returns:
            Dictionary of DataFrames with added indicators
        """
        indicator_calculator = TechnicalIndicators()
        prepared_data = {}
        
        for symbol, df in data.items():
            logger.info(f"Preparing data for {symbol}")
            
            try:
                prepared_df = indicator_calculator.apply_all_indicators(
                    df, include=indicators
                )
                prepared_data[symbol] = prepared_df
            
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {e}")
                prepared_data[symbol] = df
        
        return prepared_data
    
    def _apply_slippage(
        self,
        price: float,
        direction: str,
        is_entry: bool
    ) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Original price
            direction: Trade direction ('long' or 'short')
            is_entry: Whether this is an entry or exit
            
        Returns:
            Price with slippage applied
        """
        if direction == 'long':
            # Buy at higher price, sell at lower price
            return price * (1 + self.slippage_pct) if is_entry else price * (1 - self.slippage_pct)
        else:
            # Short at lower price, cover at higher price
            return price * (1 - self.slippage_pct) if is_entry else price * (1 + self.slippage_pct)
    
    def _calculate_commission(self, price: float, size: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            price: Trade price
            size: Trade size
            
        Returns:
            Commission amount
        """
        return price * size * self.commission_pct
    
    def _generate_trade_id(self) -> str:
        """Generate a unique trade ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _update_equity(self, timestamp: datetime) -> None:
        """
        Update equity curve and drawdown calculations.
        
        Args:
            timestamp: Current timestamp
        """
        # Calculate total position value
        position_value = 0.0
        for position in self.positions.values():
            # This is a simplification - in reality we'd need the current price
            # But for this update we'll use the last known price
            position_value += position.size * position.entry_price
        
        # Calculate total equity
        equity = self.capital + position_value
        
        # Update equity curve
        self.equity_curve[timestamp] = equity
        
        # Update max equity and drawdown
        if equity > self.max_equity:
            self.max_equity = equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.max_equity - equity) / self.max_equity
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Update drawdowns
        self.drawdowns[timestamp] = self.current_drawdown
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
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
                "profit_loss": 0.0,
                "return_pct": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "win_streak": 0,
                "loss_streak": 0,
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "profit_to_max_drawdown": 0.0,
                "recovery_factor": 0.0,
                "expectancy": 0.0,
                "avg_holding_time": 0.0
            }
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = sum(t.profit for t in winning_trades)
        total_loss = abs(sum(t.profit for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        profit_loss = sum(t.profit for t in self.trades)
        return_pct = (profit_loss / self.initial_capital) * 100
        
        # Time-based metrics
        if self.equity_curve:
            start_date = min(self.equity_curve.keys())
            end_date = max(self.equity_curve.keys())
            days = (end_date - start_date).days
            
            if days > 0:
                # Annualized return
                annualized_return = ((1 + return_pct / 100) ** (365 / days) - 1) * 100
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Calculate daily returns
        if len(self.equity_curve) > 1:
            equity_series = pd.Series({k: v for k, v in self.equity_curve.items()})
            equity_series = equity_series.resample('D').last().fillna(method='ffill')
            daily_returns = equity_series.pct_change().dropna()
            
            # Calculate volatility
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
            
            # Calculate Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * (252 ** 0.5) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Calculate Calmar ratio
            calmar_ratio = annualized_return / (self.max_drawdown * 100) if self.max_drawdown > 0 else 0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0
        
        # Trade statistics
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        largest_win = max([t.profit for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.profit for t in losing_trades]) if losing_trades else 0
        
        # Calculate win/loss streaks
        current_streak = 0
        current_streak_type = None
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in self.trades:
            is_win = trade.profit > 0
            
            if current_streak_type is None:
                current_streak_type = is_win
                current_streak = 1
            elif current_streak_type == is_win:
                current_streak += 1
            else:
                if current_streak_type:  # Win streak
                    max_win_streak = max(max_win_streak, current_streak)
                else:  # Loss streak
                    max_loss_streak = max(max_loss_streak, current_streak)
                
                current_streak_type = is_win
                current_streak = 1
        
        # Update max streaks with final streak
        if current_streak_type is not None:
            if current_streak_type:  # Win streak
                max_win_streak = max(max_win_streak, current_streak)
            else:  # Loss streak
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Calculate profit to max drawdown ratio
        profit_to_max_drawdown = profit_loss / (self.max_drawdown * self.initial_capital) if self.max_drawdown > 0 else float('inf')
        
        # Calculate recovery factor
        recovery_factor = profit_loss / (self.max_drawdown * self.initial_capital) if self.max_drawdown > 0 else float('inf')
        
        # Calculate expectancy
        win_avg = avg_win / self.initial_capital if self.initial_capital > 0 else 0
        loss_avg = avg_loss / self.initial_capital if self.initial_capital > 0 else 0
        expectancy = (win_rate * win_avg) - ((1 - win_rate) * loss_avg)
        
        # Calculate average holding time
        avg_holding_time = sum(t.holding_time_seconds for t in self.trades) / total_trades if total_trades > 0 else 0
        avg_holding_time_hours = avg_holding_time / 3600  # Convert to hours
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_profit": profit_loss / total_trades if total_trades > 0 else 0,
            "max_drawdown": self.max_drawdown * 100,  # Convert to percentage
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "profit_loss": profit_loss,
            "return_pct": return_pct,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "profit_to_max_drawdown": profit_to_max_drawdown,
            "recovery_factor": recovery_factor,
            "expectancy": expectancy,
            "avg_holding_time_hours": avg_holding_time_hours,
            "total_commission": sum(t.entry_commission + t.exit_commission for t in self.trades)
        }
    
    def backtest_strategy(
        self,
        strategy: Strategy,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> BacktestResults:
        """
        Backtest a strategy on historical data.
        
        Args:
            strategy: Strategy instance
            data: Dictionary of DataFrames with historical data and indicators
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResults with backtest results
        """
        # Reset backtester state
        self.reset()
        
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate signals for each symbol
        signals = {}
        for symbol, df in data.items():
            # Filter data by date range if specified
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            
            # Skip if no data
            if df.empty:
                logger.warning(f"No data for {symbol} in specified date range")
                continue
            
            # Generate signals
            signals[symbol] = strategy.generate_signals(df)
        
        # Find common date range across all symbols
        common_dates = None
        for df in signals.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates &= set(df.index)
        
        if not common_dates:
            logger.error("No common dates found across all symbols")
            return BacktestResults(
                strategy_name=strategy.name,
                start_date=start_date or datetime.now(),
                end_date=end_date or datetime.now(),
                initial_capital=self.initial_capital,
                final_capital=self.capital,
                symbols=list(data.keys())
            )
        
        # Sort common dates
        common_dates = sorted(common_dates)
        
        # Run backtest through common dates
        for date in common_dates:
            # Process each symbol
            for symbol, df in signals.items():
                # Skip if date not in dataframe
                if date not in df.index:
                    continue
                
                # Get current row
                row = df.loc[date]
                current_price = row['close']
                
                # Check for exit signals first
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # Check exit conditions
                    should_exit, exit_reason = strategy.should_exit_position(
                        position.to_dict(),
                        current_price,
                        df,
                        df.index.get_loc(date)
                    )
                    
                    if should_exit:
                        # Apply slippage to exit price
                        exit_price = self._apply_slippage(
                            current_price,
                            position.direction,
                            is_entry=False
                        )
                        
                        # Calculate commission
                        exit_commission = self._calculate_commission(exit_price, position.size)
                        
                        # Calculate profit/loss
                        if position.direction == 'long':
                            profit = position.size * (exit_price - position.entry_price)
                            profit_pct = (exit_price / position.entry_price) - 1
                        else:  # short
                            profit = position.size * (position.entry_price - exit_price)
                            profit_pct = 1 - (exit_price / position.entry_price)
                        
                        # Adjust profit for commission
                        profit -= exit_commission
                        
                        # Update capital
                        self.capital += position.size * exit_price - exit_commission
                        
                        # Record trade
                        trade = TradeResult(
                            entry_time=position.entry_time,
                            entry_price=position.entry_price,
                            exit_time=date,
                            exit_price=exit_price,
                            direction=position.direction,
                            size=position.size,
                            symbol=symbol,
                            profit=profit,
                            profit_pct=profit_pct,
                            exit_reason=exit_reason,
                            holding_time_seconds=(date - position.entry_time).total_seconds(),
                            entry_commission=position.size * position.entry_price * self.commission_pct,
                            exit_commission=exit_commission,
                            stop_loss=position.stop_loss,
                            take_profit=position.take_profit,
                            strategy_name=strategy.name,
                            trade_id=position.trade_id
                        )
                        self.trades.append(trade)
                        
                        # Remove position
                        del self.positions[symbol]
                    else:
                        # Update trailing stop if enabled
                        if strategy.parameters.get('use_trailing_stop', False):
                            updated_position = strategy.update_trailing_stop(
                                position.to_dict(),
                                current_price,
                                self.risk_level
                            )
                            
                            # Update position with new trailing stop
                            if updated_position and 'trailing_stop' in updated_position:
                                self.positions[symbol].trailing_stop = updated_position['trailing_stop']
                                self.positions[symbol].trailing_stop_active = updated_position.get('trailing_stop_active', False)
                
                # Check for entry signals
                entry_long = row.get('entry_long', False)
                entry_short = row.get('entry_short', False)
                
                if symbol not in self.positions and (entry_long or entry_short):
                    # Validate signal
                    signal = 1 if entry_long else -1  # 1 for buy, -1 for sell
                    
                    if strategy.validate_signal(
                        signal,
                        df,
                        df.index.get_loc(date),
                        min_confirmation=strategy.parameters.get('confirmation_needed', 2)
                    ):
                        # Calculate stop loss
                        stop_loss = strategy.calculate_stop_loss(
                            current_price,
                            signal,
                            df,
                            df.index.get_loc(date),
                            self.risk_level
                        )
                        
                        # Calculate position size
                        position_size = strategy.calculate_position_size(
                            self.capital,
                            self.risk_level,
                            current_price,
                            stop_loss
                        )
                        
                        # Apply slippage to entry price
                        entry_price = self._apply_slippage(
                            current_price,
                            'long' if signal > 0 else 'short',
                            is_entry=True
                        )
                        
                        # Calculate commission
                        entry_commission = self._calculate_commission(entry_price, position_size)
                        
                        # Check if we have enough capital
                        required_capital = position_size * entry_price + entry_commission
                        
                        if required_capital <= self.capital:
                            # Create position
                            direction = 'long' if signal > 0 else 'short'
                            trade_id = self._generate_trade_id()
                            
                            position = PositionInfo(
                                symbol=symbol,
                                direction=direction,
                                size=position_size,
                                entry_price=entry_price,
                                entry_time=date,
                                stop_loss=stop_loss,
                                take_profit=strategy.calculate_take_profit(
                                    entry_price,
                                    signal,
                                    df,
                                    df.index.get_loc(date),
                                    self.risk_level
                                ),
                                trailing_stop=None,
                                trailing_stop_active=False,
                                max_holding_time=strategy.parameters.get('max_holding_time_hours', 72) * 3600,
                                strategy_name=strategy.name,
                                trade_id=trade_id
                            )
                            
                            # Update capital
                            self.capital -= required_capital
                            
                            # Add position
                            self.positions[symbol] = position
            
            # Update equity curve at this timestamp
            self._update_equity(date)
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        # Create backtest results
        results = BacktestResults(
            strategy_name=strategy.name,
            start_date=common_dates[0] if common_dates else (start_date or datetime.now()),
            end_date=common_dates[-1] if common_dates else (end_date or datetime.now()),
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            trades=self.trades,
            equity_curve=self.equity_curve,
            drawdowns=self.drawdowns,
            performance_metrics=self.performance_metrics,
            parameters=strategy.parameters,
            symbols=list(signals.keys())
        )
        
        return results
    
    def optimize_strategy(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        parameter_ranges: Dict[str, List],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        optimization_metric: str = 'profit_loss',
        max_workers: int = None,
        n_iterations: Optional[int] = None
    ) -> Tuple[Dict, List[Dict]]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Dictionary of DataFrames with historical data and indicators
            parameter_ranges: Dictionary of parameters and their ranges
            start_date: Start date for backtest
            end_date: End date for backtest
            optimization_metric: Metric to optimize for
            max_workers: Maximum number of worker processes
            n_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (best parameters, all results)
        """
        import itertools
        
        # Generate parameter combinations
        param_keys = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        
        # Limit the number of combinations if specified
        if n_iterations is not None and len(combinations) > n_iterations:
            import random
            combinations = random.sample(combinations, n_iterations)
        
        logger.info(f"Optimizing strategy with {len(combinations)} parameter combinations")
        
        # Function to run a single backtest
        def run_backtest(params_tuple):
            # Create parameter dictionary
            params = {}
            for i, key in enumerate(param_keys):
                # Handle nested parameters
                if '.' in key:
                    main_key, sub_key = key.split('.', 1)
                    if main_key not in params:
                        params[main_key] = {}
                    params[main_key][sub_key] = params_tuple[i]
                else:
                    params[key] = params_tuple[i]
            
            # Create strategy instance
            strategy = strategy_class(parameters=params)
            
            # Run backtest
            results = self.backtest_strategy(
                strategy,
                data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Return parameters and metrics
            return {
                'parameters': params,
                'metrics': results.performance_metrics
            }
        
        # Run optimization in parallel
        all_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(run_backtest, params): params
                for params in combinations
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_params), total=len(combinations), desc="Optimizing"):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error in optimization: {e}")
        
        # Find best result
        best_result = None
        best_metric_value = float('-inf')
        
        for result in all_results:
            metric_value = result['metrics'].get(optimization_metric, float('-inf'))
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_result = result
        
        if best_result:
            logger.info(f"Optimization complete. Best {optimization_metric}: {best_metric_value}")
            return best_result['parameters'], all_results
        else:
            logger.warning("No valid results found during optimization")
            return {}, all_results
    
    def monte_carlo_analysis(
        self,
        trades: List[TradeResult],
        initial_capital: float,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis on a set of trades.
        
        Args:
            trades: List of trade results
            initial_capital: Initial capital
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for metrics
            
        Returns:
            Dictionary with Monte Carlo analysis results
        """
        if not trades:
            return {
                "error": "No trades provided for Monte Carlo analysis"
            }
        
        # Extract profit percentages from trades
        returns = [trade.profit_pct for trade in trades]
        
        # Generate random sequences of trades
        np.random.seed(42)  # For reproducibility
        
        # Store results for each simulation
        final_capitals = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Shuffle the returns
            shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate equity curve
            equity = [initial_capital]
            for ret in shuffled_returns:
                equity.append(equity[-1] * (1 + ret))
            
            # Calculate drawdown
            max_equity = equity[0]
            max_dd = 0
            
            for eq in equity:
                max_equity = max(max_equity, eq)
                dd = (max_equity - eq) / max_equity
                max_dd = max(max_dd, dd)
            
            # Store results
            final_capitals.append(equity[-1])
            max_drawdowns.append(max_dd)
        
        # Calculate statistics
        final_capitals = np.array(final_capitals)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower_idx = int(alpha / 2 * n_simulations)
        ci_upper_idx = int((1 - alpha / 2) * n_simulations)
        
        sorted_capitals = np.sort(final_capitals)
        sorted_drawdowns = np.sort(max_drawdowns)
        
        return {
            "mean_final_capital": float(np.mean(final_capitals)),
            "median_final_capital": float(np.median(final_capitals)),
            "min_final_capital": float(np.min(final_capitals)),
            "max_final_capital": float(np.max(final_capitals)),
            "std_final_capital": float(np.std(final_capitals)),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "min_max_drawdown": float(np.min(max_drawdowns)),
            "max_max_drawdown": float(np.max(max_drawdowns)),
            "ci_lower_capital": float(sorted_capitals[ci_lower_idx]),
            "ci_upper_capital": float(sorted_capitals[ci_upper_idx]),
            "ci_lower_drawdown": float(sorted_drawdowns[ci_lower_idx]),
            "ci_upper_drawdown": float(sorted_drawdowns[ci_upper_idx]),
            "probability_profit": float(np.mean(final_capitals > initial_capital)),
            "simulations": n_simulations,
            "confidence_level": confidence_level
        }
    
    def plot_equity_curve(
        self,
        results: BacktestResults,
        include_drawdowns: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot equity curve from backtest results.
        
        Args:
            results: BacktestResults instance
            include_drawdowns: Whether to include drawdowns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Convert dictionaries to Series
        equity_series = pd.Series({k: v for k, v in results.equity_curve.items()})
        
        # Plot equity curve
        ax1.plot(equity_series.index, equity_series.values, 'b-', label='Equity')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Add drawdowns if requested
        if include_drawdowns and results.drawdowns:
            ax2 = ax1.twinx()
            drawdown_series = pd.Series({k: v for k, v in results.drawdowns.items()})
            ax2.fill_between(drawdown_series.index, 0, drawdown_series.values * 100, 
                            color='r', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown %', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.invert_yaxis()  # Invert to show drawdowns going down
            ax2.set_ylim(100, 0)  # Set y-axis limits
        
        # Add trades if available
        if results.trades:
            # Plot buy points
            buy_trades = [t for t in results.trades if t.direction == 'long']
            if buy_trades:
                buy_times = [t.entry_time for t in buy_trades]
                buy_prices = [results.equity_curve.get(t.entry_time, None) for t in buy_trades]
                buy_prices = [p for p in buy_prices if p is not None]
                if buy_prices:
                    ax1.scatter(buy_times[:len(buy_prices)], buy_prices, 
                               marker='^', color='g', s=50, label='Buy')
            
            # Plot sell points
            sell_trades = [t for t in results.trades if t.direction == 'short']
            if sell_trades:
                sell_times = [t.entry_time for t in sell_trades]
                sell_prices = [results.equity_curve.get(t.entry_time, None) for t in sell_trades]
                sell_prices = [p for p in sell_prices if p is not None]
                if sell_prices:
                    ax1.scatter(sell_times[:len(sell_prices)], sell_prices, 
                               marker='v', color='r', s=50, label='Sell')
        
        # Add title and legend
        plt.title(f"{results.strategy_name} - Equity Curve")
        fig.tight_layout()
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if include_drawdowns and results.drawdowns:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        return fig
    
    def plot_drawdowns(
        self,
        results: BacktestResults,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot drawdowns from backtest results.
        
        Args:
            results: BacktestResults instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not results.drawdowns:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No drawdown data available", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert dictionary to Series
        drawdown_series = pd.Series({k: v for k, v in results.drawdowns.items()})
        
        # Plot drawdowns
        ax.fill_between(drawdown_series.index, 0, drawdown_series.values * 100, 
                       color='r', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Invert y-axis to show drawdowns going down
        ax.invert_yaxis()
        
        # Add title
        plt.title(f"{results.strategy_name} - Drawdowns")
        fig.tight_layout()
        
        return fig
    
    def plot_monthly_returns(
        self,
        results: BacktestResults,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            results: BacktestResults instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not results.equity_curve:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No equity data available", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Convert equity curve to DataFrame
        equity_df = pd.Series(results.equity_curve).to_frame('equity')
        
        # Calculate daily returns
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Group by year and month
        equity_df['year'] = equity_df.index.year
        equity_df['month'] = equity_df.index.month
        
        # Calculate monthly returns
        monthly_returns = equity_df.groupby(['year', 'month'])['return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Create pivot table
        pivot_table = monthly_returns.pivot(index='year', columns='month', values='return')
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            pivot_table * 100,  # Convert to percentage
            annot=True,
            fmt=".2f",
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Monthly Return %'}
        )
        
        # Set labels
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month names
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Add title
        plt.title(f"{results.strategy_name} - Monthly Returns (%)")
        fig.tight_layout()
        
        return fig
    
    def plot_trade_distribution(
        self,
        results: BacktestResults,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot trade profit distribution.
        
        Args:
            results: BacktestResults instance
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not results.trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No trade data available", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Extract profit percentages
        profits = [trade.profit_pct * 100 for trade in results.trades]  # Convert to percentage
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot histogram
        sns.histplot(profits, bins=50, kde=True, ax=ax1)
        ax1.axvline(0, color='r', linestyle='--')
        ax1.set_xlabel('Profit %')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f"{results.strategy_name} - Trade Profit Distribution")
        
        # Plot cumulative profit
        cumulative_profit = np.cumsum(profits)
        ax2.plot(cumulative_profit)
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('Cumulative Profit %')
        ax2.set_title('Cumulative Profit')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        return fig
    
    def generate_performance_report(
        self,
        results: BacktestResults,
        output_dir: Optional[Union[str, Path]] = None,
        include_plots: bool = True,
        save_trades: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: BacktestResults instance
            output_dir: Directory to save report files
            include_plots: Whether to include plots
            save_trades: Whether to save trades to CSV
            
        Returns:
            Dictionary with report information
        """
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = self.results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Basic report info
        report = {
            "strategy_name": results.strategy_name,
            "start_date": results.start_date.strftime("%Y-%m-%d"),
            "end_date": results.end_date.strftime("%Y-%m-%d"),
            "duration_days": (results.end_date - results.start_date).days,
            "initial_capital": results.initial_capital,
            "final_capital": results.final_capital,
            "profit_loss": results.final_capital - results.initial_capital,
            "return_pct": ((results.final_capital / results.initial_capital) - 1) * 100,
            "symbols": results.symbols,
            "total_trades": len(results.trades),
            "metrics": results.performance_metrics,
            "parameters": results.parameters,
            "files": {}
        }
        
        # Save trades if requested
        if save_trades and results.trades:
            trades_df = pd.DataFrame([t.to_dict() for t in results.trades])
            
            # Convert datetime columns to string
            for col in ['entry_time', 'exit_time']:
                if col in trades_df.columns:
                    trades_df[col] = trades_df[col].apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if x else None
                    )
            
            # Save to CSV
            trades_file = output_dir / f"{results.strategy_name}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            report["files"]["trades"] = str(trades_file)
        
        # Save equity curve
        if results.equity_curve:
            equity_df = pd.DataFrame({
                'equity': results.equity_curve,
                'drawdown': results.drawdowns
            })
            
            equity_file = output_dir / f"{results.strategy_name}_equity.csv"
            equity_df.to_csv(equity_file)
            report["files"]["equity"] = str(equity_file)
        
        # Generate and save plots if requested
        if include_plots:
            # Equity curve
            fig_equity = self.plot_equity_curve(results)
            equity_plot_file = output_dir / f"{results.strategy_name}_equity.png"
            fig_equity.savefig(equity_plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig_equity)
            report["files"]["equity_plot"] = str(equity_plot_file)
            
            # Drawdowns
            fig_drawdowns = self.plot_drawdowns(results)
            drawdowns_plot_file = output_dir / f"{results.strategy_name}_drawdowns.png"
            fig_drawdowns.savefig(drawdowns_plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig_drawdowns)
            report["files"]["drawdowns_plot"] = str(drawdowns_plot_file)
            
            # Monthly returns
            fig_monthly = self.plot_monthly_returns(results)
            monthly_plot_file = output_dir / f"{results.strategy_name}_monthly_returns.png"
            fig_monthly.savefig(monthly_plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig_monthly)
            report["files"]["monthly_returns_plot"] = str(monthly_plot_file)
            
            # Trade distribution
            if results.trades:
                fig_trades = self.plot_trade_distribution(results)
                trades_plot_file = output_dir / f"{results.strategy_name}_trade_distribution.png"
                fig_trades.savefig(trades_plot_file, dpi=300, bbox_inches='tight')
                plt.close(fig_trades)
                report["files"]["trade_distribution_plot"] = str(trades_plot_file)
        
        # Run Monte Carlo analysis if we have trades
        if results.trades:
            mc_results = self.monte_carlo_analysis(
                results.trades,
                results.initial_capital
            )
            report["monte_carlo"] = mc_results
        
        # Save full report as JSON
        report_file = output_dir / f"{results.strategy_name}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save full backtest results
        results_file = output_dir / f"{results.strategy_name}_results.pkl"
        results.save(results_file)
        
        report["files"]["report"] = str(report_file)
        report["files"]["results"] = str(results_file)
        
        return report
    
    def walk_forward_analysis(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        train_size: int = 180,  # days
        test_size: int = 60,    # days
        step_size: int = 30,    # days
        parameter_ranges: Dict[str, List] = None,
        optimization_metric: str = 'profit_loss'
    ) -> List[Dict]:
        """
        Perform walk-forward analysis with parameter optimization.
        
        Args:
            strategy_class: Strategy class
            data: Dictionary of DataFrames with historical data
            train_size: Training period in days
            test_size: Testing period in days
            step_size: Step size in days
            parameter_ranges: Dictionary of parameters and their ranges
            optimization_metric: Metric to optimize for
            
        Returns:
            List of walk-forward analysis results
        """
        # Find earliest and latest dates across all symbols
        min_date = None
        max_date = None
        
        for df in data.values():
            if min_date is None or df.index.min() < min_date:
                min_date = df.index.min()
            if max_date is None or df.index.max() > max_date:
                max_date = df.index.max()
        
        if min_date is None or max_date is None:
            logger.error("No valid dates found in data")
            return []
        
        # Generate walk-forward windows
        windows = []
        current_date = min_date
        
        while current_date + timedelta(days=train_size + test_size) <= max_date:
            train_start = current_date
            train_end = train_start + timedelta(days=train_size)
            test_start = train_end
            test_end = test_start + timedelta(days=test_size)
            
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_date += timedelta(days=step_size)
        
        if not windows:
            logger.error("No valid walk-forward windows generated")
            return []
        
        # Run walk-forward analysis
        results = []
        
        for i, window in enumerate(windows):
            logger.info(f"Walk-forward window {i+1}/{len(windows)}: "
                       f"Train {window['train_start'].date()} to {window['train_end'].date()}, "
                       f"Test {window['test_start'].date()} to {window['test_end'].date()}")
            
            # Optimize parameters on training data
            best_params, _ = self.optimize_strategy(
                strategy_class,
                data,
                parameter_ranges,
                start_date=window['train_start'],
                end_date=window['train_end'],
                optimization_metric=optimization_metric
            )
            
            if not best_params:
                logger.warning(f"No valid parameters found for window {i+1}")
                continue
            
            # Test optimized parameters on test data
            strategy = strategy_class(parameters=best_params)
            
            test_results = self.backtest_strategy(
                strategy,
                data,
                start_date=window['test_start'],
                end_date=window['test_end']
            )
            
            # Store results
            window_result = {
                'window_index': i,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'parameters': best_params,
                'train_metric': None,  # We don't have this without rerunning
                'test_metrics': test_results.performance_metrics,
                'test_return_pct': ((test_results.final_capital / test_results.initial_capital) - 1) * 100
            }
            
            results.append(window_result)
        
        return results


def run_backtest(
    strategy_name: str,
    symbols: List[str],
    timeframe: str = '1h',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0,
    risk_level: str = 'medium',
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    parameters: Dict[str, Any] = None,
    generate_report: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> BacktestResults:
    """
    Convenience function to run a backtest with a single command.
    
    Args:
        strategy_name: Name of the strategy
        symbols: List of trading pair symbols
        timeframe: Timeframe string
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital
        risk_level: Risk level (low, medium, high)
        commission_pct: Commission percentage
        slippage_pct: Slippage percentage
        parameters: Strategy parameters
        generate_report: Whether to generate a report
        output_dir: Directory to save report
        
    Returns:
        BacktestResults with backtest results
    """
    # Initialize backtester
    backtester = Backtester(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        risk_level=risk_level
    )
    
    # Fetch data
    data = backtester.fetch_data(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if not data:
        logger.error("No data fetched")
        return None
    
    # Prepare data with indicators
    prepared_data = backtester.prepare_data(data)
    
    # Create strategy
    strategy = get_strategy_by_name(strategy_name, parameters)
    
    # Run backtest
    results = backtester.backtest_strategy(
        strategy,
        prepared_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate report if requested
    if generate_report:
        backtester.generate_performance_report(
            results,
            output_dir=output_dir
        )
    
    return results


if __name__ == "__main__":
    # Example usage
    symbols = ["BTC/USDT"]
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-06-30"
    
    # Run backtest
    results = run_backtest(
        strategy_name="trend_following",
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        risk_level="medium",
        commission_pct=0.1,
        slippage_pct=0.05,
        generate_report=True
    )
    
    # Print summary
    if results:
        print(f"\nBacktest Summary for {results.strategy_name}")
        print(f"Period: {results.start_date.date()} to {results.end_date.date()}")
        print(f"Initial Capital: ${results.initial_capital:.2f}")
        print(f"Final Capital: ${results.final_capital:.2f}")
        print(f"Total Return: {((results.final_capital / results.initial_capital) - 1) * 100:.2f}%")
        print(f"Total Trades: {len(results.trades)}")
        
        if results.performance_metrics:
            print("\nPerformance Metrics:")
            metrics = results.performance_metrics
            print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
