"""
Binance API Client Wrapper for Neural Pulse Scalper

This module provides a comprehensive wrapper around the python-binance library,
offering enhanced functionality for spot trading, market data retrieval, account
management, and real-time websocket streams. It includes robust error handling,
rate limit management, and risk control features.

Features:
- Spot trading operations (market, limit, OCO orders)
- Market data retrieval (OHLCV, order book, ticker)
- Account information and balance management
- Real-time websocket data streams
- Order tracking and management
- Risk management and trading limits
- Comprehensive logging and error handling
- Support for both testnet and mainnet

Usage:
    client = BinanceClient(api_key, api_secret, testnet=True)
    
    # Get market data
    klines = client.get_klines("BTCUSDT", "1m", limit=100)
    
    # Place orders
    order_id = client.place_market_buy("BTCUSDT", quantity=0.001)
    
    # Subscribe to websocket streams
    client.start_kline_socket("BTCUSDT", "1m", callback_function)
"""

import os
import time
import json
import hmac
import hashlib
import logging
import threading
import traceback
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from datetime import datetime, timedelta
from urllib.parse import urlencode
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP

import pandas as pd
import numpy as np
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceOrderException
from binance.helpers import round_step_size
from binance.streams import BinanceSocketManager
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET,
    TIME_IN_FORCE_GTC, TIME_IN_FORCE_FOK, TIME_IN_FORCE_IOC,
    ORDER_STATUS_NEW, ORDER_STATUS_PARTIALLY_FILLED, ORDER_STATUS_FILLED,
    ORDER_STATUS_CANCELED, ORDER_STATUS_PENDING_CANCEL, ORDER_STATUS_REJECTED,
    ORDER_STATUS_EXPIRED
)

# Configure logging
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Enum for order sides."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Enum for order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class OrderStatus(Enum):
    """Enum for order statuses."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Enum for time in force options."""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class BinanceClientError(Exception):
    """Base exception for BinanceClient errors."""
    pass


class InsufficientBalanceError(BinanceClientError):
    """Exception raised when account has insufficient balance."""
    pass


class InvalidOrderError(BinanceClientError):
    """Exception raised when an order is invalid."""
    pass


class RateLimitExceededError(BinanceClientError):
    """Exception raised when rate limit is exceeded."""
    pass


class RiskLimitExceededError(BinanceClientError):
    """Exception raised when a risk limit is exceeded."""
    pass


class WebSocketError(BinanceClientError):
    """Exception raised for websocket errors."""
    pass


@dataclass
class SymbolInfo:
    """Dataclass to store symbol information."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_price: float
    max_price: float
    tick_size: float
    min_qty: float
    max_qty: float
    step_size: float
    min_notional: float
    status: str
    is_trading: bool
    precision: Dict[str, int]


@dataclass
class OrderInfo:
    """Dataclass to store order information."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    type: str
    time_in_force: str
    quantity: float
    price: float
    stop_price: float = 0.0
    iceberg_qty: float = 0.0
    original_quantity: float = 0.0
    executed_quantity: float = 0.0
    status: str = "NEW"
    time: int = 0
    update_time: int = 0
    is_working: bool = True
    
    @classmethod
    def from_dict(cls, order_dict: Dict[str, Any]) -> 'OrderInfo':
        """Create OrderInfo from dictionary returned by Binance API."""
        return cls(
            order_id=int(order_dict.get('orderId', 0)),
            client_order_id=order_dict.get('clientOrderId', ''),
            symbol=order_dict.get('symbol', ''),
            side=order_dict.get('side', ''),
            type=order_dict.get('type', ''),
            time_in_force=order_dict.get('timeInForce', ''),
            quantity=float(order_dict.get('quantity', 0.0)),
            price=float(order_dict.get('price', 0.0)),
            stop_price=float(order_dict.get('stopPrice', 0.0)),
            iceberg_qty=float(order_dict.get('icebergQty', 0.0)),
            original_quantity=float(order_dict.get('origQty', 0.0)),
            executed_quantity=float(order_dict.get('executedQty', 0.0)),
            status=order_dict.get('status', 'NEW'),
            time=int(order_dict.get('time', 0)),
            update_time=int(order_dict.get('updateTime', 0)),
            is_working=bool(order_dict.get('isWorking', True))
        )


class OrderTracker:
    """Class to track and manage orders."""
    
    def __init__(self):
        """Initialize the OrderTracker."""
        self.orders: Dict[int, OrderInfo] = {}
        self.symbol_orders: Dict[str, List[int]] = {}
        self.active_orders: Dict[int, OrderInfo] = {}
        self.filled_orders: Dict[int, OrderInfo] = {}
        self.canceled_orders: Dict[int, OrderInfo] = {}
        self.rejected_orders: Dict[int, OrderInfo] = {}
        self._lock = threading.RLock()
    
    def add_order(self, order: OrderInfo) -> None:
        """Add an order to the tracker."""
        with self._lock:
            self.orders[order.order_id] = order
            
            # Add to symbol orders
            if order.symbol not in self.symbol_orders:
                self.symbol_orders[order.symbol] = []
            self.symbol_orders[order.symbol].append(order.order_id)
            
            # Add to active orders if not filled or canceled
            if order.status not in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, 
                                   OrderStatus.REJECTED.value, OrderStatus.EXPIRED.value]:
                self.active_orders[order.order_id] = order
            
            # Add to appropriate status collection
            if order.status == OrderStatus.FILLED.value:
                self.filled_orders[order.order_id] = order
            elif order.status == OrderStatus.CANCELED.value:
                self.canceled_orders[order.order_id] = order
            elif order.status == OrderStatus.REJECTED.value:
                self.rejected_orders[order.order_id] = order
    
    def update_order(self, order_id: int, update_data: Dict[str, Any]) -> None:
        """Update an existing order with new data."""
        with self._lock:
            if order_id not in self.orders:
                logger.warning(f"Attempted to update non-existent order: {order_id}")
                return
            
            order = self.orders[order_id]
            
            # Update order attributes
            for key, value in update_data.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Update status collections
            if order.status == OrderStatus.FILLED.value:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.filled_orders[order_id] = order
            elif order.status == OrderStatus.CANCELED.value:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.canceled_orders[order_id] = order
            elif order.status == OrderStatus.REJECTED.value:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.rejected_orders[order_id] = order
            elif order.status == OrderStatus.EXPIRED.value:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
    
    def get_order(self, order_id: int) -> Optional[OrderInfo]:
        """Get an order by ID."""
        with self._lock:
            return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """Get all active orders, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                return [order for order in self.active_orders.values() if order.symbol == symbol]
            return list(self.active_orders.values())
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """Get all filled orders, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                return [order for order in self.filled_orders.values() if order.symbol == symbol]
            return list(self.filled_orders.values())
    
    def get_symbol_orders(self, symbol: str) -> List[OrderInfo]:
        """Get all orders for a symbol."""
        with self._lock:
            if symbol not in self.symbol_orders:
                return []
            return [self.orders[order_id] for order_id in self.symbol_orders[symbol] 
                   if order_id in self.orders]
    
    def clear_old_orders(self, max_age_hours: int = 24) -> None:
        """Clear orders older than the specified age."""
        with self._lock:
            current_time = int(time.time() * 1000)
            max_age_ms = max_age_hours * 60 * 60 * 1000
            
            to_remove = []
            for order_id, order in self.orders.items():
                if current_time - order.time > max_age_ms:
                    # Only remove if not active
                    if order_id not in self.active_orders:
                        to_remove.append(order_id)
            
            for order_id in to_remove:
                self._remove_order(order_id)
    
    def _remove_order(self, order_id: int) -> None:
        """Remove an order from all collections."""
        if order_id in self.orders:
            order = self.orders[order_id]
            
            # Remove from symbol orders
            if order.symbol in self.symbol_orders and order_id in self.symbol_orders[order.symbol]:
                self.symbol_orders[order.symbol].remove(order_id)
            
            # Remove from status collections
            for collection in [self.active_orders, self.filled_orders, 
                              self.canceled_orders, self.rejected_orders]:
                if order_id in collection:
                    del collection[order_id]
            
            # Remove from main orders dict
            del self.orders[order_id]


class RateLimitManager:
    """Class to manage API rate limits."""
    
    def __init__(self, max_requests_per_minute: int = 1200, max_orders_per_second: int = 10):
        """Initialize the RateLimitManager."""
        self.max_requests_per_minute = max_requests_per_minute
        self.max_orders_per_second = max_orders_per_second
        self.request_timestamps: List[float] = []
        self.order_timestamps: List[float] = []
        self._lock = threading.RLock()
    
    def check_rate_limit(self, is_order: bool = False) -> bool:
        """
        Check if the current request would exceed rate limits.
        
        Args:
            is_order: Whether this is an order request (stricter limits)
            
        Returns:
            bool: True if request is allowed, False if it would exceed limits
        """
        with self._lock:
            current_time = time.time()
            
            # Clean up old timestamps
            self._clean_timestamps(current_time)
            
            # Check order rate limit
            if is_order:
                # Check orders per second
                order_count_last_second = sum(1 for ts in self.order_timestamps 
                                             if current_time - ts <= 1.0)
                if order_count_last_second >= self.max_orders_per_second:
                    return False
            
            # Check overall request rate limit
            request_count_last_minute = len(self.request_timestamps)
            if request_count_last_minute >= self.max_requests_per_minute:
                return False
            
            return True
    
    def add_request(self, is_order: bool = False) -> None:
        """
        Record a new request.
        
        Args:
            is_order: Whether this is an order request
        """
        with self._lock:
            current_time = time.time()
            self.request_timestamps.append(current_time)
            
            if is_order:
                self.order_timestamps.append(current_time)
    
    def _clean_timestamps(self, current_time: float) -> None:
        """
        Remove timestamps older than the rate limit windows.
        
        Args:
            current_time: Current timestamp
        """
        # Keep only timestamps from the last minute
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if current_time - ts <= 60.0]
        
        # Keep only order timestamps from the last second
        self.order_timestamps = [ts for ts in self.order_timestamps 
                               if current_time - ts <= 1.0]
    
    def wait_if_needed(self, is_order: bool = False) -> float:
        """
        Wait if necessary to avoid exceeding rate limits.
        
        Args:
            is_order: Whether this is an order request
            
        Returns:
            float: Time waited in seconds
        """
        with self._lock:
            start_time = time.time()
            
            while not self.check_rate_limit(is_order):
                time.sleep(0.05)  # Sleep for 50ms and check again
                
                # Safety check to avoid infinite loops
                if time.time() - start_time > 10.0:  # Max wait 10 seconds
                    if is_order:
                        raise RateLimitExceededError("Rate limit wait timeout for order request")
                    else:
                        raise RateLimitExceededError("Rate limit wait timeout for API request")
            
            # Record this request
            self.add_request(is_order)
            
            return time.time() - start_time


class BinanceClient:
    """
    Comprehensive Binance API client wrapper.
    
    This class provides a robust interface to the Binance API, handling:
    - Spot trading operations
    - Market data retrieval
    - Account information
    - Real-time websocket streams
    - Order tracking and management
    - Risk management
    
    It supports both testnet (paper trading) and mainnet (live trading) modes.
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        request_timeout: int = 10000,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the BinanceClient.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (paper trading)
            request_timeout: Timeout for API requests in milliseconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            config: Configuration dictionary with additional settings
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.config = config or {}
        
        # Initialize the python-binance client
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
            requests_params={"timeout": self.request_timeout / 1000}
        )
        
        # Initialize websocket manager
        self.socket_manager = None
        self.socket_connections = {}
        self.socket_callbacks = {}
        
        # Initialize order tracker
        self.order_tracker = OrderTracker()
        
        # Initialize rate limit manager
        self.rate_limit_manager = RateLimitManager()
        
        # Cache for exchange info and symbol data
        self.exchange_info = None
        self.symbol_info_cache = {}
        self.ticker_cache = {}
        self.ticker_cache_time = 0
        self.ticker_cache_ttl = 5  # seconds
        
        # Risk management settings
        self.risk_limits = {
            'max_order_value': self.config.get('max_order_value_usd', 1000.0),
            'max_daily_trade_value': self.config.get('max_daily_trade_value_usd', 10000.0),
            'max_open_orders': self.config.get('max_open_orders', 10),
            'max_daily_drawdown_pct': self.config.get('max_daily_drawdown_pct', 5.0),
            'min_order_value': self.config.get('min_order_value_usd', 10.0)
        }
        
        # Trading stats for risk management
        self.trading_stats = {
            'daily_trade_value': 0.0,
            'daily_pnl': 0.0,
            'initial_balance': 0.0,
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'consecutive_losses': 0,
            'start_time': time.time()
        }
        
        # Initialize trading stats with current balance
        self._init_trading_stats()
        
        # Load exchange info
        self.update_exchange_info()
        
        logger.info(f"Initialized BinanceClient in {'testnet' if testnet else 'mainnet'} mode")
    
    def _init_trading_stats(self) -> None:
        """Initialize trading statistics with current balance."""
        try:
            account = self.get_account()
            usdt_balance = float(next((asset['free'] for asset in account['balances'] 
                                     if asset['asset'] == 'USDT'), 0.0))
            self.trading_stats['initial_balance'] = usdt_balance
            logger.info(f"Initialized trading stats with USDT balance: {usdt_balance}")
        except Exception as e:
            logger.error(f"Failed to initialize trading stats: {str(e)}")
            self.trading_stats['initial_balance'] = 0.0
    
    def _api_request(
        self,
        method_name: str,
        is_order: bool = False,
        retry_on_error: bool = True,
        *args, **kwargs
    ) -> Any:
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            method_name: Name of the Client method to call
            is_order: Whether this is an order request (stricter rate limits)
            retry_on_error: Whether to retry on certain errors
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Any: Response from the API
            
        Raises:
            Various exceptions based on the API response
        """
        # Check if we have the method
        if not hasattr(self.client, method_name):
            raise AttributeError(f"Client has no method named '{method_name}'")
        
        method = getattr(self.client, method_name)
        
        # Wait if needed to respect rate limits
        self.rate_limit_manager.wait_if_needed(is_order)
        
        # Make the request with retries
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                response = method(*args, **kwargs)
                return response
            
            except BinanceAPIException as e:
                last_error = e
                logger.warning(f"Binance API error in {method_name}: {str(e)}")
                
                # Handle specific error codes
                if e.code == -1021:  # Timestamp for this request was 1000ms ahead of the server's time
                    # Sync local time with server time
                    self._sync_time()
                    retries += 1
                    time.sleep(self.retry_delay)
                    continue
                
                elif e.code == -1015:  # Too many requests, rate limit exceeded
                    if retry_on_error and retries < self.max_retries:
                        wait_time = 1.0 * (2 ** retries)  # Exponential backoff
                        logger.warning(f"Rate limit exceeded, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        retries += 1
                        continue
                    else:
                        raise RateLimitExceededError(f"Rate limit exceeded: {str(e)}")
                
                elif e.code == -2010:  # Account has insufficient balance
                    raise InsufficientBalanceError(f"Insufficient balance: {str(e)}")
                
                elif e.code == -2011:  # Invalid order
                    raise InvalidOrderError(f"Invalid order: {str(e)}")
                
                else:
                    # For other errors, retry if requested
                    if retry_on_error and retries < self.max_retries:
                        retries += 1
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise
            
            except (BinanceRequestException, requests.exceptions.RequestException) as e:
                last_error = e
                logger.warning(f"Request error in {method_name}: {str(e)}")
                
                # Retry on network errors
                if retry_on_error and retries < self.max_retries:
                    retries += 1
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in {method_name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Retry on unexpected errors
                if retry_on_error and retries < self.max_retries:
                    retries += 1
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        # If we got here, we've exhausted our retries
        if last_error:
            raise last_error
        else:
            raise BinanceClientError(f"Failed to execute {method_name} after {self.max_retries} retries")
    
    def _sync_time(self) -> None:
        """Synchronize local time with server time."""
        try:
            server_time = self.client.get_server_time()
            server_time_ms = server_time['serverTime']
            local_time_ms = int(time.time() * 1000)
            time_diff = server_time_ms - local_time_ms
            
            logger.info(f"Time difference between local and server: {time_diff}ms")
            
            # If difference is significant, adjust request timestamps
            if abs(time_diff) > 1000:
                logger.warning(f"Significant time difference detected: {time_diff}ms")
        
        except Exception as e:
            logger.error(f"Error synchronizing time: {str(e)}")
    
    def update_exchange_info(self) -> Dict[str, Any]:
        """
        Update the cached exchange information.
        
        Returns:
            Dict[str, Any]: Exchange information
        """
        try:
            self.exchange_info = self._api_request('get_exchange_info')
            
            # Update symbol info cache
            for symbol_data in self.exchange_info['symbols']:
                if symbol_data['status'] == 'TRADING':
                    symbol = symbol_data['symbol']
                    self._update_symbol_info_cache(symbol, symbol_data)
            
            logger.info(f"Updated exchange info with {len(self.symbol_info_cache)} trading symbols")
            return self.exchange_info
            
        except Exception as e:
            logger.error(f"Error updating exchange info: {str(e)}")
            raise
    
    def _update_symbol_info_cache(self, symbol: str, symbol_data: Dict[str, Any]) -> None:
        """
        Update the cached information for a symbol.
        
        Args:
            symbol: Symbol to update
            symbol_data: Symbol data from exchange_info
        """
        try:
            # Extract price filter
            price_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            lot_size_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            min_notional_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            
            if price_filter and lot_size_filter:
                # Create SymbolInfo object
                info = SymbolInfo(
                    symbol=symbol,
                    base_asset=symbol_data['baseAsset'],
                    quote_asset=symbol_data['quoteAsset'],
                    min_price=float(price_filter['minPrice']),
                    max_price=float(price_filter['maxPrice']),
                    tick_size=float(price_filter['tickSize']),
                    min_qty=float(lot_size_filter['minQty']),
                    max_qty=float(lot_size_filter['maxQty']),
                    step_size=float(lot_size_filter['stepSize']),
                    min_notional=float(min_notional_filter['minNotional']) if min_notional_filter else 0.0,
                    status=symbol_data['status'],
                    is_trading=symbol_data['status'] == 'TRADING',
                    precision={
                        'price': symbol_data['quotePrecision'],
                        'quantity': symbol_data['baseAssetPrecision']
                    }
                )
                
                self.symbol_info_cache[symbol] = info
                
        except Exception as e:
            logger.error(f"Error updating symbol info cache for {symbol}: {str(e)}")
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get information about a symbol.
        
        Args:
            symbol: Symbol to get information for
            
        Returns:
            Optional[SymbolInfo]: Symbol information or None if not found
        """
        # Check if we have it in cache
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        # If not, try to get it from exchange info
        if not self.exchange_info:
            self.update_exchange_info()
        
        # Check cache again after update
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        return None
    
    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Get ticker data for all symbols.
        
        Returns:
            List[Dict[str, Any]]: List of ticker data
        """
        current_time = time.time()
        
        # Check if cache is valid
        if current_time - self.ticker_cache_time <= self.ticker_cache_ttl and self.ticker_cache:
            return list(self.ticker_cache.values())
        
        # Get fresh data
        tickers = self._api_request('get_all_tickers')
        
        # Update cache
        self.ticker_cache = {ticker['symbol']: ticker for ticker in tickers}
        self.ticker_cache_time = current_time
        
        return tickers
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol: Symbol to get ticker for
            
        Returns:
            Dict[str, Any]: Ticker data
        """
        current_time = time.time()
        
        # Check if cache is valid and contains this symbol
        if (current_time - self.ticker_cache_time <= self.ticker_cache_ttl and 
            symbol in self.ticker_cache):
            return self.ticker_cache[symbol]
        
        # Get fresh data
        ticker = self._api_request('get_ticker', symbol=symbol)
        
        # Update cache for this symbol
        self.ticker_cache[symbol] = ticker
        
        # If cache is old, refresh all tickers
        if current_time - self.ticker_cache_time > self.ticker_cache_ttl:
            threading.Thread(target=self.get_all_tickers, daemon=True).start()
        
        return ticker
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get klines (candlestick data) for a symbol.
        
        Args:
            symbol: Symbol to get klines for
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to get (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            pd.DataFrame: DataFrame with kline data
        """
        try:
            # Make API request
            klines = self._api_request(
                'get_klines',
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_time,
                endTime=end_time
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting klines for {symbol} {interval}: {str(e)}")
            raise
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_str: str,
        end_str: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical klines for a symbol.
        
        Args:
            symbol: Symbol to get klines for
            interval: Kline interval
            start_str: Start time as string (e.g., "1 day ago", "1 Jan, 2020")
            end_str: End time as string
            limit: Number of klines per request (max 1000)
            
        Returns:
            pd.DataFrame: DataFrame with kline data
        """
        try:
            # Make API request
            klines = self._api_request(
                'get_historical_klines',
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical klines for {symbol} {interval}: {str(e)}")
            raise
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            limit: Number of bids and asks to get (max 5000)
            
        Returns:
            Dict[str, Any]: Order book data
        """
        try:
            return self._api_request('get_order_book', symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            raise
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            limit: Number of trades to get (max 1000)
            
        Returns:
            List[Dict[str, Any]]: List of trade data
        """
        try:
            return self._api_request('get_recent_trades', symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {str(e)}")
            raise
    
    def get_historical_trades(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            limit: Number of trades to get (max 1000)
            from_id: Trade ID to start from
            
        Returns:
            List[Dict[str, Any]]: List of trade data
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            if from_id:
                params['fromId'] = from_id
            
            return self._api_request('get_historical_trades', **params)
        except Exception as e:
            logger.error(f"Error getting historical trades for {symbol}: {str(e)}")
            raise
    
    def get_aggregate_trades(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Get aggregate trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of trades to get (max 1000)
            
        Returns:
            List[Dict[str, Any]]: List of aggregate trade data
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            return self._api_request('get_aggregate_trades', **params)
        except Exception as e:
            logger.error(f"Error getting aggregate trades for {symbol}: {str(e)}")
            raise
    
    def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24-hour ticker for a symbol.
        
        Args:
            symbol: Symbol to get ticker for
            
        Returns:
            Dict[str, Any]: 24-hour ticker data
        """
        try:
            return self._api_request('get_ticker', symbol=symbol)
        except Exception as e:
            logger.error(f"Error getting 24h ticker for {symbol}: {str(e)}")
            raise
    
    def get_all_24h_tickers(self) -> List[Dict[str, Any]]:
        """
        Get 24-hour tickers for all symbols.
        
        Returns:
            List[Dict[str, Any]]: List of 24-hour ticker data
        """
        try:
            return self._api_request('get_ticker')
        except Exception as e:
            logger.error(f"Error getting all 24h tickers: {str(e)}")
            raise
    
    def get_avg_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get average price for a symbol.
        
        Args:
            symbol: Symbol to get average price for
            
        Returns:
            Dict[str, Any]: Average price data
        """
        try:
            return self._api_request('get_avg_price', symbol=symbol)
        except Exception as e:
            logger.error(f"Error getting average price for {symbol}: {str(e)}")
            raise
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            return self._api_request('get_account')
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise
    
    def get_asset_balance(self, asset: str) -> Dict[str, str]:
        """
        Get balance for an asset.
        
        Args:
            asset: Asset to get balance for
            
        Returns:
            Dict[str, str]: Asset balance data
        """
        try:
            return self._api_request('get_asset_balance', asset=asset)
        except Exception as e:
            logger.error(f"Error getting balance for {asset}: {str(e)}")
            raise
    
    def get_all_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get balances for all assets.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of asset balances
        """
        try:
            account = self._api_request('get_account')
            balances = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            
            return balances
        except Exception as e:
            logger.error(f"Error getting all balances: {str(e)}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol: Symbol to get open orders for (None for all symbols)
            
        Returns:
            List[Dict[str, Any]]: List of open orders
        """
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            orders = self._api_request('get_open_orders', **params)
            
            # Update order tracker
            for order in orders:
                self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            return orders
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            raise
    
    def get_order(self, symbol: str, order_id: Optional[int] = None, 
                 orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            symbol: Symbol the order is for
            order_id: Order ID
            orig_client_order_id: Original client order ID
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            params = {'symbol': symbol}
            if order_id:
                params['orderId'] = order_id
            elif orig_client_order_id:
                params['origClientOrderId'] = orig_client_order_id
            else:
                raise ValueError("Either order_id or orig_client_order_id must be provided")
            
            order = self._api_request('get_order', **params)
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            return order
        except Exception as e:
            logger.error(f"Error getting order for {symbol}: {str(e)}")
            raise
    
    def get_all_orders(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Get all orders for a symbol.
        
        Args:
            symbol: Symbol to get orders for
            order_id: Order ID to start from
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of orders to get (max 1000)
            
        Returns:
            List[Dict[str, Any]]: List of orders
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            if order_id:
                params['orderId'] = order_id
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            orders = self._api_request('get_all_orders', **params)
            
            # Update order tracker
            for order in orders:
                self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            return orders
        except Exception as e:
            logger.error(f"Error getting all orders for {symbol}: {str(e)}")
            raise
    
    def get_my_trades(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        from_id: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Get trades for a symbol.
        
        Args:
            symbol: Symbol to get trades for
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            from_id: Trade ID to start from
            limit: Number of trades to get (max 1000)
            
        Returns:
            List[Dict[str, Any]]: List of trades
        """
        try:
            params = {'symbol': symbol, 'limit': limit}
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            if from_id:
                params['fromId'] = from_id
            
            return self._api_request('get_my_trades', **params)
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {str(e)}")
            raise
    
    def _check_order_risk_limits(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> bool:
        """
        Check if an order would exceed risk limits.
        
        Args:
            symbol: Symbol to check
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Order price (None for market orders)
            
        Returns:
            bool: True if order is within risk limits, False otherwise
            
        Raises:
            RiskLimitExceededError: If order would exceed risk limits
        """
        try:
            # Get current price if not provided
            if price is None:
                ticker = self.get_ticker(symbol)
                price = float(ticker['lastPrice'])
            
            # Calculate order value
            order_value = quantity * price
            
            # Check minimum order value
            if order_value < self.risk_limits['min_order_value']:
                raise RiskLimitExceededError(
                    f"Order value ${order_value:.2f} is below minimum ${self.risk_limits['min_order_value']:.2f}"
                )
            
            # Check maximum order value
            if order_value > self.risk_limits['max_order_value']:
                raise RiskLimitExceededError(
                    f"Order value ${order_value:.2f} exceeds maximum ${self.risk_limits['max_order_value']:.2f}"
                )
            
            # Check daily trade value
            if self.trading_stats['daily_trade_value'] + order_value > self.risk_limits['max_daily_trade_value']:
                raise RiskLimitExceededError(
                    f"Daily trade value would exceed maximum ${self.risk_limits['max_daily_trade_value']:.2f}"
                )
            
            # Check maximum open orders
            open_orders = self.order_tracker.get_active_orders()
            if len(open_orders) >= self.risk_limits['max_open_orders']:
                raise RiskLimitExceededError(
                    f"Maximum number of open orders ({self.risk_limits['max_open_orders']}) reached"
                )
            
            # Check daily drawdown
            if self.trading_stats['initial_balance'] > 0:
                current_balance = self._get_account_value()
                drawdown_pct = (self.trading_stats['initial_balance'] - current_balance) / self.trading_stats['initial_balance'] * 100
                
                if drawdown_pct > self.risk_limits['max_daily_drawdown_pct']:
                    raise RiskLimitExceededError(
                        f"Daily drawdown ({drawdown_pct:.2f}%) exceeds maximum ({self.risk_limits['max_daily_drawdown_pct']:.2f}%)"
                    )
            
            return True
            
        except RiskLimitExceededError:
            # Re-raise risk limit errors
            raise
        except Exception as e:
            logger.error(f"Error checking order risk limits: {str(e)}")
            # Default to allowing the order if we can't check risk limits
            return True
    
    def _get_account_value(self) -> float:
        """
        Get total account value in USDT.
        
        Returns:
            float: Account value in USDT
        """
        try:
            balances = self.get_all_balances()
            total_value = 0.0
            
            for asset, balance in balances.items():
                if asset == 'USDT':
                    total_value += balance['total']
                else:
                    try:
                        # Try to get price in USDT
                        symbol = f"{asset}USDT"
                        ticker = self.get_ticker(symbol)
                        price = float(ticker['lastPrice'])
                        asset_value = balance['total'] * price
                        total_value += asset_value
                    except:
                        # If we can't get price in USDT, try BTC
                        try:
                            symbol = f"{asset}BTC"
                            ticker = self.get_ticker(symbol)
                            price_btc = float(ticker['lastPrice'])
                            
                            # Convert BTC to USDT
                            btc_usdt = self.get_ticker("BTCUSDT")
                            price_usdt = float(btc_usdt['lastPrice'])
                            
                            asset_value = balance['total'] * price_btc * price_usdt
                            total_value += asset_value
                        except:
                            # If we can't get price, ignore this asset
                            pass
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error getting account value: {str(e)}")
            return 0.0
    
    def _update_trading_stats(self, order_value: float, is_profit: bool = False, 
                             profit_amount: float = 0.0) -> None:
        """
        Update trading statistics after a trade.
        
        Args:
            order_value: Value of the order
            is_profit: Whether the trade was profitable
            profit_amount: Amount of profit/loss
        """
        try:
            # Update daily trade value
            self.trading_stats['daily_trade_value'] += order_value
            
            # Update trade count
            self.trading_stats['trade_count'] += 1
            
            # Update win/loss count and consecutive losses
            if is_profit:
                self.trading_stats['win_count'] += 1
                self.trading_stats['consecutive_losses'] = 0
            else:
                self.trading_stats['loss_count'] += 1
                self.trading_stats['consecutive_losses'] += 1
            
            # Update daily PnL
            self.trading_stats['daily_pnl'] += profit_amount
            
            # Reset stats if it's a new day
            current_time = time.time()
            if current_time - self.trading_stats['start_time'] > 86400:  # 24 hours
                self._reset_daily_stats()
                
        except Exception as e:
            logger.error(f"Error updating trading stats: {str(e)}")
    
    def _reset_daily_stats(self) -> None:
        """Reset daily trading statistics."""
        try:
            # Store current balance as initial balance
            current_balance = self._get_account_value()
            
            self.trading_stats['daily_trade_value'] = 0.0
            self.trading_stats['daily_pnl'] = 0.0
            self.trading_stats['initial_balance'] = current_balance
            self.trading_stats['start_time'] = time.time()
            
            logger.info(f"Reset daily trading stats. New initial balance: {current_balance}")
            
        except Exception as e:
            logger.error(f"Error resetting daily stats: {str(e)}")
    
    def _adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Adjust quantity to the correct precision for a symbol.
        
        Args:
            symbol: Symbol to adjust quantity for
            quantity: Quantity to adjust
            
        Returns:
            float: Adjusted quantity
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No symbol info found for {symbol}, using original quantity")
                return quantity
            
            # Round to step size
            step_size = symbol_info.step_size
            return round_step_size(quantity, step_size)
            
        except Exception as e:
            logger.error(f"Error adjusting quantity precision: {str(e)}")
            return quantity
    
    def _adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Adjust price to the correct precision for a symbol.
        
        Args:
            symbol: Symbol to adjust price for
            price: Price to adjust
            
        Returns:
            float: Adjusted price
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No symbol info found for {symbol}, using original price")
                return price
            
            # Round to tick size
            tick_size = symbol_info.tick_size
            return round_step_size(price, tick_size)
            
        except Exception as e:
            logger.error(f"Error adjusting price precision: {str(e)}")
            return price
    
    def place_market_buy(self, symbol: str, quantity: float) -> Dict[str, Any]:
        """
        Place a market buy order.
        
        Args:
            symbol: Symbol to buy
            quantity: Quantity to buy
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, SIDE_BUY, quantity)
            
            # Adjust quantity precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            
            # Place order
            order = self._api_request(
                'order_market_buy',
                is_order=True,
                symbol=symbol,
                quantity=quantity
            )
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            # Update trading stats
            price = float(order.get('price', 0)) or float(self.get_ticker(symbol)['lastPrice'])
            order_value = quantity * price
            self._update_trading_stats(order_value)
            
            logger.info(f"Placed market buy order for {quantity} {symbol} at ~${price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing market buy order for {symbol}: {str(e)}")
            raise
    
    def place_market_sell(self, symbol: str, quantity: float) -> Dict[str, Any]:
        """
        Place a market sell order.
        
        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, SIDE_SELL, quantity)
            
            # Adjust quantity precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            
            # Place order
            order = self._api_request(
                'order_market_sell',
                is_order=True,
                symbol=symbol,
                quantity=quantity
            )
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            # Update trading stats
            price = float(order.get('price', 0)) or float(self.get_ticker(symbol)['lastPrice'])
            order_value = quantity * price
            self._update_trading_stats(order_value)
            
            logger.info(f"Placed market sell order for {quantity} {symbol} at ~${price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing market sell order for {symbol}: {str(e)}")
            raise
    
    def place_limit_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        time_in_force: str = TIME_IN_FORCE_GTC
    ) -> Dict[str, Any]:
        """
        Place a limit buy order.
        
        Args:
            symbol: Symbol to buy
            quantity: Quantity to buy
            price: Price to buy at
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, SIDE_BUY, quantity, price)
            
            # Adjust quantity and price precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            price = self._adjust_price_precision(symbol, price)
            
            # Place order
            order = self._api_request(
                'order_limit_buy',
                is_order=True,
                symbol=symbol,
                quantity=quantity,
                price=price,
                timeInForce=time_in_force
            )
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            # Update trading stats
            order_value = quantity * price
            self._update_trading_stats(order_value)
            
            logger.info(f"Placed limit buy order for {quantity} {symbol} at ${price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing limit buy order for {symbol}: {str(e)}")
            raise
    
    def place_limit_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        time_in_force: str = TIME_IN_FORCE_GTC
    ) -> Dict[str, Any]:
        """
        Place a limit sell order.
        
        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell
            price: Price to sell at
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, SIDE_SELL, quantity, price)
            
            # Adjust quantity and price precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            price = self._adjust_price_precision(symbol, price)
            
            # Place order
            order = self._api_request(
                'order_limit_sell',
                is_order=True,
                symbol=symbol,
                quantity=quantity,
                price=price,
                timeInForce=time_in_force
            )
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            # Update trading stats
            order_value = quantity * price
            self._update_trading_stats(order_value)
            
            logger.info(f"Placed limit sell order for {quantity} {symbol} at ${price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing limit sell order for {symbol}: {str(e)}")
            raise
    
    def place_stop_loss_limit(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_price: float,
        time_in_force: str = TIME_IN_FORCE_GTC
    ) -> Dict[str, Any]:
        """
        Place a stop loss limit order.
        
        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell
            price: Price to sell at
            stop_price: Stop price
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, SIDE_SELL, quantity, price)
            
            # Adjust quantity and price precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            price = self._adjust_price_precision(symbol, price)
            stop_price = self._adjust_price_precision(symbol, stop_price)
            
            # Place order
            order = self._api_request(
                'create_order',
                is_order=True,
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=time_in_force,
                quantity=quantity,
                price=price,
                stopPrice=stop_price
            )
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            logger.info(f"Placed stop loss limit order for {quantity} {symbol} at ${price} (stop: ${stop_price})")
            return order
            
        except Exception as e:
            logger.error(f"Error placing stop loss limit order for {symbol}: {str(e)}")
            raise
    
    def place_take_profit_limit(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_price: float,
        time_in_force: str = TIME_IN_FORCE_GTC
    ) -> Dict[str, Any]:
        """
        Place a take profit limit order.
        
        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell
            price: Price to sell at
            stop_price: Stop price
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, SIDE_SELL, quantity, price)
            
            # Adjust quantity and price precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            price = self._adjust_price_precision(symbol, price)
            stop_price = self._adjust_price_precision(symbol, stop_price)
            
            # Place order
            order = self._api_request(
                'create_order',
                is_order=True,
                symbol=symbol,
                side=SIDE_SELL,
                type="TAKE_PROFIT_LIMIT",
                timeInForce=time_in_force,
                quantity=quantity,
                price=price,
                stopPrice=stop_price
            )
            
            # Update order tracker
            self.order_tracker.add_order(OrderInfo.from_dict(order))
            
            logger.info(f"Placed take profit limit order for {quantity} {symbol} at ${price} (stop: ${stop_price})")
            return order
            
        except Exception as e:
            logger.error(f"Error placing take profit limit order for {symbol}: {str(e)}")
            raise
    
    def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place an OCO (One-Cancels-the-Other) order.
        
        Args:
            symbol: Symbol to trade
            side: Order side (BUY or SELL)
            quantity: Quantity to trade
            price: Limit price
            stop_price: Stop price
            stop_limit_price: Stop limit price (defaults to stop_price if None)
            
        Returns:
            Dict[str, Any]: Order information
        """
        try:
            # Check risk limits
            self._check_order_risk_limits(symbol, side, quantity, price)
            
            # Adjust quantity and price precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            price = self._adjust_price_precision(symbol, price)
            stop_price = self._adjust_price_precision(symbol, stop_price)
            
            if stop_limit_price is None:
                stop_limit_price = stop_price
            else:
                stop_limit_price = self._adjust_price_precision(symbol, stop_limit_price)
            
            # Place order
            order = self._api_request(
                'order_oco_sell' if side == SIDE_SELL else 'order_oco_buy',
                is_order=True,
                symbol=symbol,
                quantity=quantity,
                price=price,
                stopPrice=stop_price,
                stopLimitPrice=stop_limit_price
            )
            
            logger.info(f"Placed OCO {side} order for {quantity} {symbol} at ${price} / ${stop_price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing OCO order for {symbol}: {str(e)}")
            raise
    
    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: Symbol the order is for
            order_id: Order ID
            orig_client_order_id: Original client order ID
            
        Returns:
            Dict[str, Any]: Cancellation information
        """
        try:
            params = {'symbol': symbol}
            if order_id:
                params['orderId'] = order_id
            elif orig_client_order_id:
                params['origClientOrderId'] = orig_client_order_id
            else:
                raise ValueError("Either order_id or orig_client_order_id must be provided")
            
            result = self._api_request('cancel_order', is_order=True, **params)
            
            # Update order tracker
            if order_id:
                order = self.order_tracker.get_order(order_id)
                if order:
                    self.order_tracker.update_order(order_id, {'status': OrderStatus.CANCELED.value})
            
            logger.info(f"Cancelled order for {symbol}: {order_id or orig_client_order_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error cancelling order for {symbol}: {str(e)}")
            raise
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Symbol to cancel orders for (None for all symbols)
            
        Returns:
            List[Dict[str, Any]]: List of cancellation information
        """
        try:
            if symbol:
                # Cancel all orders for a specific symbol
                open_orders = self.get_open_orders(symbol=symbol)
                if not open_orders:
                    return []
                result = self._api_request('cancel_open_orders', is_order=True, symbol=symbol)
                
                # Update order tracker
                for order in self.order_tracker.get_active_orders(symbol):
                    self.order_tracker.update_order(order.order_id, {'status': OrderStatus.CANCELED.value})
                
                logger.info(f"Cancelled all orders for {symbol}")
                return result
            else:
                # Get all open orders and cancel them one by one
                open_orders = self.get_open_orders()
                results = []
                
                for order in open_orders:
                    symbol_to_cancel = order['symbol']
                    order_id = order['orderId']
                    try:
                        result = self.cancel_order(symbol_to_cancel, order_id=order_id)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order_id} for {symbol_to_cancel}: {e}")

                logger.info(f"Cancelled all orders across all symbols: {len(results)} orders")
                return results
                
        except Exception as e:
            logger.error(f"Error cancelling all orders: {str(e)}")
            raise
    
    def start_user_socket(self, callback: Callable) -> str:
        """
        Start a user data stream socket.
        
        Args:
            callback: Callback function for socket messages
            
        Returns:
            str: Socket connection key
        """
        try:
            if not self.socket_manager:
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Start the user socket
            conn_key = self.socket_manager.start_user_socket(callback)
            self.socket_connections[conn_key] = 'user'
            self.socket_callbacks[conn_key] = callback
            
            # Start the socket manager if not already running
            if not self.socket_manager.is_manager_stopping():
                self.socket_manager.start()
            
            logger.info("Started user data stream socket")
            return conn_key
            
        except Exception as e:
            logger.error(f"Error starting user socket: {str(e)}")
            raise WebSocketError(f"Failed to start user socket: {str(e)}")
    
    def start_symbol_ticker_socket(self, symbol: str, callback: Callable) -> str:
        """
        Start a symbol ticker socket.
        
        Args:
            symbol: Symbol to get ticker for
            callback: Callback function for socket messages
            
        Returns:
            str: Socket connection key
        """
        try:
            if not self.socket_manager:
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Start the ticker socket
            conn_key = self.socket_manager.start_symbol_ticker_socket(symbol, callback)
            self.socket_connections[conn_key] = f'ticker_{symbol}'
            self.socket_callbacks[conn_key] = callback
            
            # Start the socket manager if not already running
            if not self.socket_manager.is_manager_stopping():
                self.socket_manager.start()
            
            logger.info(f"Started ticker socket for {symbol}")
            return conn_key
            
        except Exception as e:
            logger.error(f"Error starting ticker socket for {symbol}: {str(e)}")
            raise WebSocketError(f"Failed to start ticker socket: {str(e)}")
    
    def start_kline_socket(self, symbol: str, interval: str, callback: Callable) -> str:
        """
        Start a kline socket.
        
        Args:
            symbol: Symbol to get klines for
            interval: Kline interval (1m, 3m, 5m, etc.)
            callback: Callback function for socket messages
            
        Returns:
            str: Socket connection key
        """
        try:
            if not self.socket_manager:
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Start the kline socket
            conn_key = self.socket_manager.start_kline_socket(symbol, callback, interval=interval)
            self.socket_connections[conn_key] = f'kline_{symbol}_{interval}'
            self.socket_callbacks[conn_key] = callback
            
            # Start the socket manager if not already running
            if not self.socket_manager.is_manager_stopping():
                self.socket_manager.start()
            
            logger.info(f"Started kline socket for {symbol} {interval}")
            return conn_key
            
        except Exception as e:
            logger.error(f"Error starting kline socket for {symbol} {interval}: {str(e)}")
            raise WebSocketError(f"Failed to start kline socket: {str(e)}")
    
    def start_depth_socket(self, symbol: str, callback: Callable, depth: str = "5") -> str:
        """
        Start an order book depth socket.
        
        Args:
            symbol: Symbol to get depth for
            callback: Callback function for socket messages
            depth: Depth of the order book (5, 10, 20)
            
        Returns:
            str: Socket connection key
        """
        try:
            if not self.socket_manager:
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Start the depth socket
            conn_key = self.socket_manager.start_depth_socket(symbol, callback, depth=depth)
            self.socket_connections[conn_key] = f'depth_{symbol}_{depth}'
            self.socket_callbacks[conn_key] = callback
            
            # Start the socket manager if not already running
            if not self.socket_manager.is_manager_stopping():
                self.socket_manager.start()
            
            logger.info(f"Started depth socket for {symbol} (depth {depth})")
            return conn_key
            
        except Exception as e:
            logger.error(f"Error starting depth socket for {symbol}: {str(e)}")
            raise WebSocketError(f"Failed to start depth socket: {str(e)}")
    
    def start_multiplex_socket(self, streams: List[str], callback: Callable) -> str:
        """
        Start a multiplex socket with multiple streams.
        
        Args:
            streams: List of stream names
            callback: Callback function for socket messages
            
        Returns:
            str: Socket connection key
        """
        try:
            if not self.socket_manager:
                self.socket_manager = BinanceSocketManager(self.client)
            
            # Start the multiplex socket
            conn_key = self.socket_manager.start_multiplex_socket(streams, callback)
            self.socket_connections[conn_key] = f'multiplex_{",".join(streams)}'
            self.socket_callbacks[conn_key] = callback
            
            # Start the socket manager if not already running
            if not self.socket_manager.is_manager_stopping():
                self.socket_manager.start()
            
            logger.info(f"Started multiplex socket with streams: {streams}")
            return conn_key
            
        except Exception as e:
            logger.error(f"Error starting multiplex socket: {str(e)}")
            raise WebSocketError(f"Failed to start multiplex socket: {str(e)}")
    
    def stop_socket(self, conn_key: str) -> None:
        """
        Stop a socket connection.
        
        Args:
            conn_key: Socket connection key
        """
        try:
            if not self.socket_manager:
                logger.warning("Socket manager not initialized")
                return
            
            # Stop the socket
            self.socket_manager.stop_socket(conn_key)
            
            # Remove from connections and callbacks
            if conn_key in self.socket_connections:
                del self.socket_connections[conn_key]
            if conn_key in self.socket_callbacks:
                del self.socket_callbacks[conn_key]
            
            logger.info(f"Stopped socket connection: {conn_key}")
            
        except Exception as e:
            logger.error(f"Error stopping socket {conn_key}: {str(e)}")
    
    def stop_all_sockets(self) -> None:
        """Stop all socket connections."""
        try:
            if not self.socket_manager:
                logger.warning("Socket manager not initialized")
                return
            
            # Stop all sockets
            self.socket_manager.close()
            
            # Clear connections and callbacks
            self.socket_connections = {}
            self.socket_callbacks = {}
            
            # Reset socket manager
            self.socket_manager = None
            
            logger.info("Stopped all socket connections")
            
        except Exception as e:
            logger.error(f"Error stopping all sockets: {str(e)}")
    
    def calculate_position_size(
        self,
        symbol: str,
        risk_percentage: float,
        stop_loss_price: float,
        entry_price: Optional[float] = None,
        max_position_size_usd: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk percentage and stop loss.
        
        Args:
            symbol: Symbol to calculate position size for.
            risk_percentage: Percentage of total account value to risk (e.g., 1.0 for 1%).
            stop_loss_price: The price at which the stop loss will be triggered.
            entry_price: The intended entry price. If None, current market price is used.
            max_position_size_usd: Maximum position size in USD.
            
        Returns:
            float: The calculated quantity of the base asset to trade, or 0.0 if invalid.
        """
        try:
            # Get account value in USDT
            account_value = self._get_account_value()
            if account_value <= 0:
                logger.warning("Cannot calculate position size with zero or negative account value.")
                return 0.0

            # Calculate risk amount in USD
            risk_amount_usd = account_value * (risk_percentage / 100.0)

            # Get entry price if not provided
            if entry_price is None:
                ticker = self.get_ticker(symbol)
                entry_price = float(ticker['lastPrice'])

            # Ensure stop loss and entry prices are valid
            if stop_loss_price == entry_price:
                logger.warning("Entry price cannot be the same as stop loss price.")
                return 0.0

            # Calculate risk per share/unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit <= 0:
                return 0.0

            # Calculate desired quantity
            quantity = risk_amount_usd / risk_per_unit
            
            # Get symbol info for constraints
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Could not retrieve symbol info for {symbol}. Cannot validate position size.")
                return 0.0

            # Check against max position size in USD if provided
            if max_position_size_usd is not None:
                max_quantity_from_usd = max_position_size_usd / entry_price
                quantity = min(quantity, max_quantity_from_usd)

            # Clamp quantity by symbol's max_qty
            quantity = min(quantity, symbol_info.max_qty)

            # Adjust for precision
            adjusted_quantity = self._adjust_quantity_precision(symbol, quantity)

            # Check against min_qty and min_notional
            if adjusted_quantity < symbol_info.min_qty:
                logger.warning(f"Calculated quantity {adjusted_quantity} is below min_qty {symbol_info.min_qty} for {symbol}.")
                return 0.0
            
            notional_value = adjusted_quantity * entry_price
            if notional_value < symbol_info.min_notional:
                logger.warning(f"Calculated notional value {notional_value} is below min_notional {symbol_info.min_notional} for {symbol}.")
                return 0.0

            return adjusted_quantity

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def get_symbol_info_v2(self, symbol: str) -> Optional[SymbolInfo]:
        """Alias for get_symbol_info."""
        return self.get_symbol_info(symbol)

    def get_trading_stats(self) -> Dict[str, Any]:
        """
        Get current trading statistics.
        
        Returns:
            Dict[str, Any]: A copy of the trading statistics dictionary.
        """
        return self.trading_stats.copy()

    def get_connection_status(self) -> Dict[str, str]:
        """
        Check the connection status of the API and websockets.
        
        Returns:
            Dict[str, str]: A dictionary with the status of API and websockets.
        """
        status = {
            'api_status': 'disconnected',
            'websocket_status': 'disconnected'
        }
        try:
            self.client.ping()
            status['api_status'] = 'connected'
        except Exception as e:
            logger.warning(f"API ping failed: {e}")
            status['api_status'] = 'disconnected'

        if self.socket_manager and not self.socket_manager.is_manager_stopping():
            status['websocket_status'] = 'connected'
        
        return status

    def cleanup(self) -> None:
        """Alias for the close method for graceful shutdown."""
        self.close()

    def close(self) -> None:
        """
        Gracefully close all connections.
        """
        logger.info("Closing BinanceClient and all connections...")
        self.stop_all_sockets()
        logger.info("BinanceClient closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
