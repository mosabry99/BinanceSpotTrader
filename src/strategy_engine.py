"""
Neural Pulse Scalper - Strategy Engine

This module contains the core logic for the Neural Pulse Scalper strategy.
It orchestrates the entire trading process, from data ingestion and signal
generation to order execution and risk management.

The StrategyEngine class is the central component, managing the state of
the bot, processing market data in real-time, and making trading decisions
based on the pre-defined strategy rules in the configuration.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from enum import Enum
from src.binance_client import BinanceClient, OrderSide
from src.indicators import TechnicalIndicators

# Configure logging
logger = logging.getLogger(__name__)


class TradeStatus(str, Enum):
    """Enumeration for the status of a trade."""
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"


@dataclass
class Trade:
    """Dataclass to hold all information about a single trade."""
    trade_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    status: TradeStatus = TradeStatus.ACTIVE
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    initial_stop_loss: float = 0.0
    current_stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: Optional[str] = None
    entry_signal_strength: float = 0.0
    trailing_stop_activated: bool = False


class StrategyEngine:
    """
    The main engine that implements the Neural Pulse Scalper strategy.
    It connects to the Binance client, processes market data, generates signals,
    and executes trades based on the configured strategy.
    """

    def __init__(self, config: Dict[str, Any], binance_client: BinanceClient, indicators: TechnicalIndicators):
        """
        Initialize the StrategyEngine.

        Args:
            config: The configuration dictionary.
            binance_client: An instance of the BinanceClient.
            indicators: An instance of the TechnicalIndicators class.
        """
        self.config = config
        self.client = binance_client
        self.indicators = indicators

        self.is_running = False
        self.active_trades: Dict[str, Trade] = {}  # symbol -> Trade
        self.trade_history: List[Trade] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.trading_pairs: List[str] = self.config['trading']['custom_pairs']

        self._lock = threading.RLock()
        self._init_risk_manager_state()

        logger.info("StrategyEngine initialized.")

    def _init_risk_manager_state(self):
        """Initializes the state for the risk management module."""
        self.risk_state = {
            "trades_today": 0,
            "consecutive_losses": 0,
            "daily_drawdown": 0.0,
            "is_paused": False,
            "pause_until": None,
            "last_trade_date": datetime.utcnow().date()
        }
        logger.info("Risk manager state initialized.")

    def run(self):
        """Starts the strategy engine."""
        if self.is_running:
            logger.warning("StrategyEngine is already running.")
            return

        self.is_running = True
        logger.info("Starting StrategyEngine...")

        # Load initial historical data for all pairs
        self._load_all_historical_data()

        # Start websocket streams
        self._start_websocket_streams()

        logger.info("StrategyEngine is now running.")
        try:
            while self.is_running:
                # The main logic is event-driven by websocket messages.
                # This loop can be used for periodic tasks, like checking for time-based exits.
                with self._lock:
                    self._check_time_stops()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping engine.")
            self.stop()
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            self.stop()

    def stop(self):
        """Stops the strategy engine."""
        if not self.is_running:
            logger.warning("StrategyEngine is not running.")
            return

        logger.info("Stopping StrategyEngine...")
        self.is_running = False
        self.client.stop_all_sockets()
        # Optionally, close all open positions
        # self._close_all_positions()
        logger.info("StrategyEngine stopped.")

    def _load_all_historical_data(self):
        """Loads historical data for all configured trading pairs."""
        logger.info("Loading initial historical data for all pairs...")
        for symbol in self.trading_pairs:
            try:
                self._load_historical_data(symbol, self.config['strategy']['timeframes']['primary'])
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")
        logger.info("Finished loading historical data.")

    def _load_historical_data(self, symbol: str, timeframe: str, limit: int = 200):
        """
        Loads historical k-line data for a symbol to warm up indicators.
        """
        logger.info(f"Loading historical data for {symbol} on {timeframe} timeframe...")
        df = self.client.get_klines(symbol=symbol, interval=timeframe, limit=limit)
        # Calculate all indicators for the historical data
        df_with_indicators = self.indicators.add_all_indicators(df, self.config)
        self.historical_data[symbol] = df_with_indicators
        logger.info(f"Loaded {len(df)} candles for {symbol}.")

    def _start_websocket_streams(self):
        """Starts the k-line websocket streams for all trading pairs."""
        logger.info("Starting websocket streams...")
        streams = [
            f"{symbol.lower()}@kline_{self.config['strategy']['timeframes']['primary']}"
            for symbol in self.trading_pairs
        ]
        self.client.start_multiplex_socket(streams, self._process_kline_message)

    def _process_kline_message(self, msg: Dict[str, Any]):
        """
        Callback function to process incoming k-line messages from the websocket.
        """
        if not self.is_running:
            return

        if 'e' in msg and msg['e'] == 'error':
            logger.error(f"Websocket error received: {msg}")
            return

        if 'k' not in msg:
            return # Not a kline message

        kline = msg['k']
        symbol = kline['s']

        # Check if the candle is closed
        if kline['x']:
            logger.debug(f"Closed kline received for {symbol}: C={kline['c']}")
            with self._lock:
                self._update_dataframe(symbol, kline)
                self._run_strategy_cycle(symbol)

    def _update_dataframe(self, symbol: str, kline: Dict[str, Any]):
        """Updates the historical data DataFrame with a new closed candle."""
        if symbol not in self.historical_data:
            self._load_historical_data(symbol, self.config['strategy']['timeframes']['primary'])
            return

        new_candle = pd.DataFrame([{
            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
        }]).set_index('timestamp')

        df = self.historical_data[symbol]
        df = pd.concat([df, new_candle])
        # Keep the DataFrame size manageable
        df = df.iloc[-500:]
        
        # Recalculate indicators
        self.historical_data[symbol] = self.indicators.add_all_indicators(df, self.config)

    def _run_strategy_cycle(self, symbol: str):
        """
        Runs the main strategy logic for a given symbol.
        This is triggered on each new closed candle.
        """
        logger.debug(f"Running strategy cycle for {symbol}...")
        
        # 1. Check exit conditions for any active trade on this symbol
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            current_price = self.historical_data[symbol]['close'].iloc[-1]
            self._update_trailing_stop(trade, current_price)
            exit_reason = self._check_exit_signals(trade, current_price)
            if exit_reason:
                logger.info(f"Exit signal for {symbol}: {exit_reason}. Closing trade.")
                self._exit_trade(trade, current_price, exit_reason)
            return # Don't check for entry if we have an active trade

        # 2. Check for new entry signals
        if not self._check_risk_conditions():
            return # Risk limits reached or bot is paused

        if not self._check_pre_filter(symbol):
            return # Market conditions not met

        entry_signal, signal_strength = self._check_entry_signals(symbol)
        if entry_signal != 0:
            side = OrderSide.BUY if entry_signal == 1 else OrderSide.SELL
            current_price = self.historical_data[symbol]['close'].iloc[-1]
            logger.info(f"Entry signal for {symbol}: {side.value} at {current_price} with strength {signal_strength:.2f}")
            self._enter_trade(symbol, side, current_price, signal_strength)

    def _check_risk_conditions(self) -> bool:
        """Checks global risk management rules."""
        risk_cfg = self.config['risk_management']

        # Reset daily stats if it's a new day
        if datetime.utcnow().date() > self.risk_state['last_trade_date']:
            self.risk_state['trades_today'] = 0
            self.risk_state['daily_drawdown'] = 0.0
            self.risk_state['last_trade_date'] = datetime.utcnow().date()
            logger.info("New day detected, resetting daily risk stats.")

        if self.risk_state['is_paused']:
            if datetime.utcnow() > self.risk_state['pause_until']:
                self.risk_state['is_paused'] = False
                self.risk_state['pause_until'] = None
                logger.info("Trading pause has ended.")
            else:
                logger.debug("Trading is currently paused due to consecutive losses.")
                return False

        if len(self.active_trades) >= risk_cfg['max_open_trades']:
            logger.debug(f"Max open trades ({risk_cfg['max_open_trades']}) reached.")
            return False

        if self.risk_state['trades_today'] >= risk_cfg['max_trades_per_day']:
            logger.info(f"Max trades per day ({risk_cfg['max_trades_per_day']}) reached.")
            return False
        
        # NOTE: Daily drawdown check would require PnL tracking from a portfolio manager
        # This is a simplified version.
        
        return True

    def _check_pre_filter(self, symbol: str) -> bool:
        """Checks the pre-filter market conditions for a symbol."""
        filter_cfg = self.config['strategy']['pre_filter']
        df = self.historical_data[symbol]
        
        # 24h Volume Check (approximated with available data)
        # A proper implementation would use the 24h ticker data.
        volume_24h_approx = df['volume'].iloc[-1440:].sum() # Approx 24h for 1m timeframe
        price = df['close'].iloc[-1]
        volume_usd = volume_24h_approx * price
        if volume_usd < filter_cfg['min_24h_volume_usd']:
            logger.debug(f"{symbol} failed pre-filter: 24h volume {volume_usd:.2f} < {filter_cfg['min_24h_volume_usd']}")
            return False

        # Volatility Check
        volatility_period = 30 # 30 minutes
        returns = df['close'].pct_change()
        volatility = returns.iloc[-volatility_period:].std() * np.sqrt(volatility_period) * 100
        if volatility < filter_cfg['min_volatility_pct']:
            logger.debug(f"{symbol} failed pre-filter: Volatility {volatility:.2f}% < {filter_cfg['min_volatility_pct']}%")
            return False
            
        logger.debug(f"{symbol} passed pre-filter checks.")
        return True

    def _check_entry_signals(self, symbol: str) -> (int, float):
        """
        Checks all entry conditions for a symbol based on the latest data.
        Returns:
            A tuple of (signal, strength), where signal is 1 for buy, -1 for sell, 0 for none.
        """
        entry_cfg = self.config['strategy']['entry']
        df = self.historical_data[symbol]
        last = df.iloc[-1]
        prev = df.iloc[-2]

        conditions_met = []
        
        # EMA Crossover
        if entry_cfg['ema_crossover']['enabled']:
            fast_ema = last[f"ema_{entry_cfg['ema_crossover']['fast_period']}"]
            slow_ema = last[f"ema_{entry_cfg['ema_crossover']['slow_period']}"]
            prev_fast_ema = prev[f"ema_{entry_cfg['ema_crossover']['fast_period']}"]
            prev_slow_ema = prev[f"ema_{entry_cfg['ema_crossover']['slow_period']}"]
            
            if fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema:
                conditions_met.append(1) # Bullish cross
            elif fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema:
                conditions_met.append(-1) # Bearish cross
            else:
                conditions_met.append(0)

        # RSI
        if entry_cfg['rsi']['enabled']:
            rsi = last[f"rsi_{entry_cfg['rsi']['period']}"]
            if entry_cfg['rsi']['lower_bound'] <= rsi <= entry_cfg['rsi']['upper_bound']:
                conditions_met.append(1) # Condition met for both buy and sell
            else:
                conditions_met.append(0)
        
        # Bollinger Bands
        if entry_cfg['bollinger_bands']['enabled']:
            if last['low'] <= last['bb_lower']:
                conditions_met.append(1) # Buy signal
            elif last['high'] >= last['bb_upper']:
                conditions_met.append(-1) # Sell signal
            else:
                conditions_met.append(0)
                
        # VWAP
        if entry_cfg['vwap']['enabled']:
            if last['close'] > last['vwap'] and prev['close'] <= prev['vwap']:
                conditions_met.append(1) # Reclaimed VWAP
            elif last['close'] < last['vwap'] and prev['close'] >= prev['vwap']:
                conditions_met.append(-1) # Rejected by VWAP
            else:
                conditions_met.append(0)

        # Volume Spike
        if entry_cfg['volume']['enabled']:
            avg_vol = df['volume'].iloc[-entry_cfg['volume']['lookback_periods']-1:-1].mean()
            if last['volume'] > avg_vol * entry_cfg['volume']['spike_multiplier']:
                conditions_met.append(1) # Volume spike met for both buy/sell
            else:
                conditions_met.append(0)
        
        # Candlestick Pattern
        if entry_cfg['candlestick_patterns']['enabled']:
            # This requires a more complex pattern detection logic from indicators.py
            # Assuming indicators.py adds a 'pattern_signal' column (1 for bullish, -1 for bearish)
            if 'pattern_signal' in last and last['pattern_signal'] != 0:
                conditions_met.append(last['pattern_signal'])
            else:
                conditions_met.append(0)

        # AI Decision Node
        if self.config['ai_model']['enabled']:
            ai_prob = self._get_ai_decision(last) # Pass features to the model
            if ai_prob < self.config['ai_model']['win_probability_threshold']:
                logger.debug(f"AI model rejected trade for {symbol}. Probability: {ai_prob}")
                return 0, 0
        
        # Final Signal Aggregation
        num_conditions = len(conditions_met)
        if num_conditions == 0: return 0, 0
        
        buy_signals = sum(1 for c in conditions_met if c > 0)
        sell_signals = sum(1 for c in conditions_met if c < 0)

        if entry_cfg['require_all_conditions']:
            if buy_signals == num_conditions:
                return 1, 1.0
            if sell_signals == num_conditions:
                return -1, 1.0
        else: # A weighted or threshold-based approach could be used here
            if buy_signals > sell_signals and buy_signals / num_conditions > 0.6:
                return 1, buy_signals / num_conditions
            if sell_signals > buy_signals and sell_signals / num_conditions > 0.6:
                return -1, sell_signals / num_conditions

        return 0, 0
    
    def _get_ai_decision(self, features: pd.Series) -> float:
        """
        Placeholder for the AI/ML decision model.
        In a real implementation, this would load a trained model and predict.
        """
        # For now, return a high probability to allow trades to pass
        return 0.85

    def _check_exit_signals(self, trade: Trade, current_price: float) -> Optional[str]:
        """Checks all exit conditions for an active trade."""
        exit_cfg = self.config['strategy']['exit']

        # Stop Loss
        if trade.side == OrderSide.BUY and current_price <= trade.current_stop_loss:
            return f"Stop Loss hit at {trade.current_stop_loss}"
        if trade.side == OrderSide.SELL and current_price >= trade.current_stop_loss:
            return f"Stop Loss hit at {trade.current_stop_loss}"

        # Take Profit
        if trade.side == OrderSide.BUY and current_price >= trade.take_profit:
            return f"Take Profit hit at {trade.take_profit}"
        if trade.side == OrderSide.SELL and current_price <= trade.take_profit:
            return f"Take Profit hit at {trade.take_profit}"
            
        # Time Stop
        if exit_cfg['time_stop']['enabled']:
            duration = datetime.utcnow() - trade.entry_time
            if duration.total_seconds() / 60 > exit_cfg['time_stop']['max_trade_duration_minutes']:
                return f"Time Stop after {exit_cfg['time_stop']['max_trade_duration_minutes']} minutes"

        # VWAP Exit
        if exit_cfg['vwap_exit']['enabled']:
            last_candle = self.historical_data[trade.symbol].iloc[-1]
            vwap = last_candle['vwap']
            if trade.side == OrderSide.BUY and current_price < vwap:
                return "VWAP crossed downwards"
            if trade.side == OrderSide.SELL and current_price > vwap:
                return "VWAP crossed upwards"

        return None
    
    def _check_time_stops(self):
        """Iterate through active trades and check for time-based exits."""
        exit_cfg = self.config['strategy']['exit']
        if not exit_cfg['time_stop']['enabled']:
            return
            
        trades_to_exit = []
        for symbol, trade in self.active_trades.items():
            duration = datetime.utcnow() - trade.entry_time
            if duration.total_seconds() / 60 > exit_cfg['time_stop']['max_trade_duration_minutes']:
                reason = f"Time Stop after {exit_cfg['time_stop']['max_trade_duration_minutes']} minutes"
                trades_to_exit.append((trade, reason))

        for trade, reason in trades_to_exit:
            logger.info(f"Time-based exit signal for {trade.symbol}: {reason}. Closing trade.")
            current_price = self.historical_data[trade.symbol]['close'].iloc[-1]
            self._exit_trade(trade, current_price, reason)
            
    def _update_trailing_stop(self, trade: Trade, current_price: float):
        """Adjusts the stop loss price if trailing stop is enabled and activated."""
        ts_cfg = self.config['strategy']['exit']['trailing_stop']
        if not ts_cfg['enabled']:
            return

        pnl_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100
        if trade.side == OrderSide.SELL:
            pnl_percent *= -1

        if not trade.trailing_stop_activated and pnl_percent >= ts_cfg['activation_percentage']:
            trade.trailing_stop_activated = True
            logger.info(f"Trailing stop activated for {trade.symbol} at {pnl_percent:.2f}% profit.")

        if trade.trailing_stop_activated:
            if trade.side == OrderSide.BUY:
                new_stop_loss = current_price * (1 - ts_cfg['callback_rate'] / 100)
                if new_stop_loss > trade.current_stop_loss:
                    trade.current_stop_loss = new_stop_loss
                    logger.debug(f"Trailing stop for {trade.symbol} updated to {new_stop_loss}")
            elif trade.side == OrderSide.SELL:
                new_stop_loss = current_price * (1 + ts_cfg['callback_rate'] / 100)
                if new_stop_loss < trade.current_stop_loss:
                    trade.current_stop_loss = new_stop_loss
                    logger.debug(f"Trailing stop for {trade.symbol} updated to {new_stop_loss}")
    
    def _enter_trade(self, symbol: str, side: OrderSide, price: float, signal_strength: float):
        """Executes an entry order and creates a new trade object."""
        risk_cfg = self.config['risk_management']
        exit_cfg = self.config['strategy']['exit']
        
        try:
            # Calculate Stop Loss and Take Profit prices
            if side == OrderSide.BUY:
                stop_loss_price = price * (1 - exit_cfg['stop_loss']['percentage'] / 100)
                take_profit_price = price * (1 + exit_cfg['take_profit']['percentage'] / 100)
            else: # SELL
                stop_loss_price = price * (1 + exit_cfg['stop_loss']['percentage'] / 100)
                take_profit_price = price * (1 - exit_cfg['take_profit']['percentage'] / 100)

            # Calculate position size
            quantity = self.client.calculate_position_size(
                symbol=symbol,
                risk_percentage=risk_cfg['position_sizing']['percentage'],
                stop_loss_price=stop_loss_price,
                entry_price=price
            )
            
            if quantity == 0.0:
                logger.warning(f"Position size for {symbol} is zero. Skipping trade.")
                return

            # Place the market order
            if side == OrderSide.BUY:
                order = self.client.place_market_buy(symbol, quantity)
            else:
                order = self.client.place_market_sell(symbol, quantity)
            
            # Create and store the trade object
            entry_price = float(order['fills'][0]['price']) if order.get('fills') else price
            
            trade = Trade(
                trade_id=order['clientOrderId'],
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                initial_stop_loss=stop_loss_price,
                current_stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                entry_signal_strength=signal_strength
            )
            self.active_trades[symbol] = trade
            self.risk_state['trades_today'] += 1
            
            logger.info(f"SUCCESSFULLY ENTERED TRADE: {side.value} {quantity} {symbol} at {entry_price}")
            logger.info(f"  - SL: {stop_loss_price}, TP: {take_profit_price}")

        except Exception as e:
            logger.error(f"Failed to enter trade for {symbol}: {e}", exc_info=True)

    def _exit_trade(self, trade: Trade, exit_price: float, reason: str):
        """Executes an exit order and updates trade history."""
        try:
            # Place the closing market order
            if trade.side == OrderSide.BUY:
                order = self.client.place_market_sell(trade.symbol, trade.quantity)
            else:
                order = self.client.place_market_buy(trade.symbol, trade.quantity)
            
            # Update trade object
            actual_exit_price = float(order['fills'][0]['price']) if order.get('fills') else exit_price
            trade.exit_price = actual_exit_price
            trade.exit_time = datetime.utcnow()
            trade.exit_reason = reason
            trade.status = TradeStatus.CLOSED

            # Calculate PnL
            if trade.side == OrderSide.BUY:
                trade.pnl = (actual_exit_price - trade.entry_price) * trade.quantity
            else: # SELL
                trade.pnl = (trade.entry_price - actual_exit_price) * trade.quantity
            
            trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100

            # Update performance stats
            self._update_performance_stats(trade)

            # Move from active trades to history
            self.trade_history.append(trade)
            del self.active_trades[trade.symbol]

            logger.info(f"SUCCESSFULLY EXITED TRADE: {trade.symbol} for reason: {reason}")
            logger.info(f"  - PnL: ${trade.pnl:.4f} ({trade.pnl_percent:.2f}%)")

        except Exception as e:
            logger.error(f"Failed to exit trade for {trade.symbol}: {e}", exc_info=True)
            trade.status = TradeStatus.ERROR
            trade.exit_reason = f"Error on exit: {e}"
            # Decide how to handle the stuck trade (e.g., retry, manual intervention)

    def _update_performance_stats(self, trade: Trade):
        """Updates risk and performance metrics after a trade is closed."""
        if trade.pnl > 0:
            self.risk_state['consecutive_losses'] = 0
        else:
            self.risk_state['consecutive_losses'] += 1

        # Check for pause condition
        pause_cfg = self.config['risk_management']['consecutive_losses']
        if self.risk_state['consecutive_losses'] >= pause_cfg['pause_after']:
            self.risk_state['is_paused'] = True
            self.risk_state['pause_until'] = datetime.utcnow() + timedelta(minutes=pause_cfg['pause_duration_minutes'])
            self.risk_state['consecutive_losses'] = 0 # Reset after pausing
            logger.warning(
                f"Paused trading for {pause_cfg['pause_duration_minutes']} minutes "
                f"due to {pause_cfg['pause_after']} consecutive losses."
            )
            
    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the engine."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "active_trades_count": len(self.active_trades),
                "trades_history_count": len(self.trade_history),
                "risk_state": self.risk_state,
                "active_trades": [vars(t) for t in self.active_trades.values()]
            }

