#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Fetcher Module

This module handles all data retrieval operations from cryptocurrency exchanges,
with a primary focus on Binance. It provides functionality for fetching both
historical and real-time market data with proper error handling, rate limiting,
data validation, and caching mechanisms.

Key features:
- Historical OHLCV data retrieval
- Real-time market data streaming
- Data validation and cleaning
- Efficient caching to minimize API calls
- Rate limiting to comply with exchange restrictions
- Robust error handling with exponential backoff
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import ccxt
from ccxt.base.errors import NetworkError, ExchangeError, RequestTimeout
import requests
from requests.exceptions import RequestException
from functools import lru_cache
import threading
import queue
import hashlib
import pickle

# Import local modules
import config

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO if not config.DEBUG_MODE else logging.DEBUG)

class RateLimiter:
    """
    Rate limiter to prevent exceeding exchange API limits.
    Implements a token bucket algorithm for rate limiting.
    """
    def __init__(self, max_calls: int, period: int):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        # Calculate tokens to add
        new_tokens = (elapsed / self.period) * self.max_calls
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.max_calls)
            self.last_refill_time = now
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, blocking if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Delay time in seconds if had to wait
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0
            
            # Calculate time to wait for enough tokens
            required_tokens = tokens - self.tokens
            wait_time = (required_tokens / self.max_calls) * self.period
            
            # Sleep for the required time
            time.sleep(wait_time)
            
            # After waiting, we should have enough tokens
            self.tokens = 0
            self.last_refill_time = time.time()
            return wait_time


class DataCache:
    """
    Cache for storing fetched market data to minimize API calls.
    Implements both memory and disk caching with TTL (Time-To-Live).
    """
    def __init__(
        self, 
        cache_dir: Path = config.DATA_DIR / "cache",
        memory_ttl: int = 300,  # 5 minutes
        disk_ttl: int = 86400,   # 24 hours
        max_memory_items: int = 100
    ):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached data
            memory_ttl: Time-to-live for memory cache in seconds
            disk_ttl: Time-to-live for disk cache in seconds
            max_memory_items: Maximum number of items to keep in memory
        """
        self.cache_dir = cache_dir
        self.memory_ttl = memory_ttl
        self.disk_ttl = disk_ttl
        self.max_memory_items = max_memory_items
        self.memory_cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_key(self, params: Dict) -> str:
        """
        Generate a unique cache key from parameters.
        
        Args:
            params: Dictionary of parameters to hash
            
        Returns:
            String hash key
        """
        # Sort the dictionary to ensure consistent hashing
        serialized = json.dumps(params, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, params: Dict) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available and not expired.
        
        Args:
            params: Dictionary of parameters to look up
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        key = self._generate_key(params)
        
        # Check memory cache first
        with self.lock:
            if key in self.memory_cache:
                cache_time, data = self.memory_cache[key]
                if time.time() - cache_time <= self.memory_ttl:
                    self.access_times[key] = time.time()
                    logger.debug(f"Memory cache hit for {params}")
                    return data
                else:
                    # Expired from memory
                    del self.memory_cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_time, data = pickle.load(f)
                
                if time.time() - cache_time <= self.disk_ttl:
                    # Update memory cache
                    with self.lock:
                        self._manage_memory_cache()
                        self.memory_cache[key] = (time.time(), data)
                        self.access_times[key] = time.time()
                    logger.debug(f"Disk cache hit for {params}")
                    return data
                else:
                    # Remove expired disk cache
                    cache_path.unlink(missing_ok=True)
            except (pickle.PickleError, EOFError, IOError) as e:
                logger.warning(f"Error reading cache file {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, params: Dict, data: pd.DataFrame) -> None:
        """
        Store data in both memory and disk cache.
        
        Args:
            params: Dictionary of parameters used to fetch the data
            data: DataFrame to cache
        """
        if data is None or data.empty:
            return
            
        key = self._generate_key(params)
        current_time = time.time()
        
        # Update memory cache
        with self.lock:
            self._manage_memory_cache()
            self.memory_cache[key] = (current_time, data)
            self.access_times[key] = current_time
        
        # Update disk cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((current_time, data), f)
            logger.debug(f"Cached data for {params}")
        except (IOError, pickle.PickleError) as e:
            logger.warning(f"Error writing to cache file {cache_path}: {e}")
    
    def _manage_memory_cache(self) -> None:
        """
        Manage memory cache size by removing least recently used items.
        Should be called with the lock held.
        """
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove least recently accessed items
            items_to_remove = len(self.memory_cache) - self.max_memory_items + 1
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
            
            for key, _ in sorted_keys[:items_to_remove]:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def clear(self, older_than: Optional[int] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            older_than: If provided, only clear entries older than this many seconds
        """
        current_time = time.time()
        
        # Clear memory cache
        with self.lock:
            if older_than is None:
                self.memory_cache.clear()
                self.access_times.clear()
            else:
                keys_to_remove = []
                for key, (cache_time, _) in self.memory_cache.items():
                    if current_time - cache_time > older_than:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
        
        # Clear disk cache
        if older_than is None:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_time, _ = pickle.load(f)
                    
                    if current_time - cache_time > older_than:
                        cache_file.unlink(missing_ok=True)
                except (pickle.PickleError, EOFError, IOError):
                    # If there's an error reading the file, remove it
                    cache_file.unlink(missing_ok=True)


class DataFetcher:
    """
    Main class for fetching cryptocurrency market data from exchanges.
    Handles both historical and real-time data with proper error handling,
    rate limiting, and caching.
    """
    def __init__(
        self,
        exchange_id: str = config.API_CONFIG["exchange"],
        api_key: str = config.API_CONFIG["api_key"],
        api_secret: str = config.API_CONFIG["api_secret"],
        use_testnet: bool = config.API_CONFIG["testnet"],
        timeout: int = config.API_CONFIG["timeout"],
        rate_limit: bool = config.API_CONFIG["rate_limit"]
    ):
        """
        Initialize the DataFetcher with exchange connection.
        
        Args:
            exchange_id: Exchange ID (e.g., 'binance')
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            use_testnet: Whether to use the exchange's testnet
            timeout: API request timeout in seconds
            rate_limit: Whether to enable rate limiting
        """
        self.exchange_id = exchange_id
        self.use_testnet = use_testnet
        
        # Initialize exchange connection
        exchange_class = getattr(ccxt, exchange_id)
        
        # Exchange options
        options = {
            'timeout': timeout * 1000,  # ccxt uses milliseconds
            'enableRateLimit': rate_limit,
            'adjustForTimeDifference': True
        }
        
        # Add API credentials if provided
        if api_key and api_secret:
            options['apiKey'] = api_key
            options['secret'] = api_secret
        
        # Add testnet if applicable
        if use_testnet:
            if exchange_id == 'binance':
                options['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
        
        # Initialize exchange
        self.exchange = exchange_class(options)
        
        # Initialize rate limiter (default: 10 requests per second)
        self.rate_limiter = RateLimiter(
            max_calls=10,
            period=1
        )
        
        # Initialize data cache
        self.cache = DataCache()
        
        # For real-time data streaming
        self.websocket_running = False
        self.websocket_thread = None
        self.data_queue = queue.Queue()
        
        # Load exchange info
        self._load_exchange_info()
        
        logger.info(f"Initialized DataFetcher for {exchange_id}" + 
                   f" {'(testnet)' if use_testnet else ''}")
    
    def _load_exchange_info(self) -> None:
        """Load exchange information (markets, timeframes, etc.)"""
        try:
            self.exchange.load_markets()
            logger.info(f"Loaded {len(self.exchange.markets)} markets from {self.exchange_id}")
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            # Continue without markets loaded - will try again when needed
    
    def _execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        max_retries: int = 3, 
        initial_delay: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries in seconds
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retries fail
        """
        retries = 0
        last_exception = None
        delay = initial_delay
        
        while retries <= max_retries:
            try:
                # Apply rate limiting
                self.rate_limiter.acquire()
                
                # Execute the function
                return func(*args, **kwargs)
                
            except (NetworkError, RequestTimeout) as e:
                retries += 1
                last_exception = e
                
                if retries <= max_retries:
                    # Exponential backoff with jitter
                    jitter = 0.1 * delay * (2 * np.random.random() - 1)
                    sleep_time = delay + jitter
                    logger.warning(
                        f"Network error on attempt {retries}/{max_retries}. "
                        f"Retrying in {sleep_time:.2f}s. Error: {e}"
                    )
                    time.sleep(sleep_time)
                    delay *= 2  # Exponential backoff
                
            except ExchangeError as e:
                # Exchange errors might be permanent, so log and raise
                logger.error(f"Exchange error: {e}")
                raise
                
            except Exception as e:
                # Unexpected error
                logger.error(f"Unexpected error in API call: {e}")
                raise
        
        # If we get here, all retries failed
        logger.error(f"All retries failed. Last error: {last_exception}")
        raise last_exception

    def get_exchange_info(self) -> Dict:
        """
        Get exchange information including trading pairs and limits.
        
        Returns:
            Dictionary with exchange information
        """
        # Reload markets if needed
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            self._load_exchange_info()
            
        # Extract relevant information
        pairs = {}
        for symbol, market in self.exchange.markets.items():
            if market['active'] and '/' in symbol:
                base, quote = symbol.split('/')
                pairs[symbol] = {
                    'base': base,
                    'quote': quote,
                    'precision': market.get('precision', {}),
                    'limits': market.get('limits', {}),
                    'min_notional': market.get('limits', {}).get('cost', {}).get('min', 0),
                    'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0),
                }
        
        return {
            'name': self.exchange.name,
            'timeframes': list(self.exchange.timeframes.keys()) if hasattr(self.exchange, 'timeframes') else [],
            'pairs': pairs,
            'has': self.exchange.has,
        }
    
    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of timeframes supported by the exchange.
        
        Returns:
            List of supported timeframe strings
        """
        if hasattr(self.exchange, 'timeframes'):
            return list(self.exchange.timeframes.keys())
        return []
    
    def get_supported_pairs(self) -> List[str]:
        """
        Get list of trading pairs supported by the exchange.
        
        Returns:
            List of supported trading pair symbols
        """
        # Reload markets if needed
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            self._load_exchange_info()
            
        return [symbol for symbol, market in self.exchange.markets.items() 
                if market['active'] and '/' in symbol]

    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with ticker information
        """
        return self._execute_with_retry(self.exchange.fetch_ticker, symbol)
    
    def get_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Get current ticker information for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols or None for all
            
        Returns:
            Dictionary with ticker information for each symbol
        """
        return self._execute_with_retry(self.exchange.fetch_tickers, symbols)
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Maximum number of orders to retrieve
            
        Returns:
            Dictionary with order book information
        """
        return self._execute_with_retry(self.exchange.fetch_order_book, symbol, limit)
    
    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[Union[int, str, datetime]] = None,
        limit: Optional[int] = None,
        end_time: Optional[Union[int, str, datetime]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            since: Start time as timestamp, ISO string, or datetime object
            limit: Maximum number of candles to retrieve
            end_time: End time as timestamp, ISO string, or datetime object
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert datetime objects to timestamps
        if isinstance(since, datetime):
            since = int(since.timestamp() * 1000)
        elif isinstance(since, str):
            since = int(pd.to_datetime(since).timestamp() * 1000)
        
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)
        elif isinstance(end_time, str):
            end_time = int(pd.to_datetime(end_time).timestamp() * 1000)
            
        # Default to recent data if no start time provided
        if since is None and end_time is None:
            # Default to last 100 candles or user-specified limit
            limit = limit or 100
            
        # Create cache parameters
        cache_params = {
            'type': 'ohlcv',
            'symbol': symbol,
            'timeframe': timeframe,
            'since': since,
            'limit': limit,
            'end_time': end_time
        }
        
        # Try to get data from cache
        if use_cache:
            cached_data = self.cache.get(cache_params)
            if cached_data is not None:
                return cached_data
        
        # If we need to fetch data for a specific time range that's too large,
        # we'll need to make multiple API calls and combine the results
        all_candles = []
        
        if end_time and since:
            # Calculate the timeframe in milliseconds
            tf_ms = self._timeframe_to_milliseconds(timeframe)
            
            # Maximum candles per request (exchange dependent)
            max_limit = 1000  # Most exchanges limit to 1000 candles per request
            
            # Calculate how many candles we need
            total_time = end_time - since
            total_candles = total_time // tf_ms
            
            if total_candles > max_limit:
                # We need multiple requests
                current_since = since
                
                while current_since < end_time:
                    # Calculate the end of this batch
                    batch_end = min(current_since + (tf_ms * max_limit), end_time)
                    
                    # Fetch this batch
                    batch_candles = self._execute_with_retry(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        timeframe,
                        since=current_since,
                        limit=max_limit
                    )
                    
                    if not batch_candles:
                        break
                    
                    all_candles.extend(batch_candles)
                    
                    # Update for next batch
                    if len(batch_candles) < max_limit:
                        # We got fewer candles than requested, so we're done
                        break
                    
                    # Move to the next batch
                    current_since = batch_candles[-1][0] + tf_ms
            else:
                # We can fetch all data in one request
                all_candles = self._execute_with_retry(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    timeframe,
                    since=since,
                    limit=limit or int(total_candles)
                )
        else:
            # Simple case - just fetch with the provided parameters
            all_candles = self._execute_with_retry(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since=since,
                limit=limit
            )
        
        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end_time if provided
            if end_time:
                end_datetime = pd.to_datetime(end_time, unit='ms')
                df = df[df.index <= end_datetime]
            
            # Limit to requested number of candles if provided
            if limit and len(df) > limit:
                df = df.iloc[-limit:]
            
            # Cache the result
            self.cache.set(cache_params, df)
            
            return df
        
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def _timeframe_to_milliseconds(self, timeframe: str) -> int:
        """
        Convert a timeframe string to milliseconds.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            Timeframe in milliseconds
        """
        # Parse the timeframe string
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        # Convert to milliseconds
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Maximum number of trades to retrieve
            
        Returns:
            DataFrame with trade information
        """
        trades = self._execute_with_retry(self.exchange.fetch_trades, symbol, limit=limit)
        
        if trades:
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            
            return df
        
        return pd.DataFrame()
    
    def get_account_balance(self) -> Dict:
        """
        Get account balance information.
        
        Returns:
            Dictionary with balance information
        """
        if not self.exchange.apiKey:
            raise ValueError("API key is required for account balance")
            
        return self._execute_with_retry(self.exchange.fetch_balance)
    
    def start_websocket_stream(
        self,
        symbol: str,
        channels: List[str] = ['ticker'],
        callback: Optional[Callable] = None
    ) -> None:
        """
        Start a websocket stream for real-time data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            channels: List of channels to subscribe to
            callback: Callback function for data processing
        """
        # Not all exchanges in ccxt support websockets directly
        # For simplicity, we'll use a separate thread to simulate real-time data
        if self.websocket_running:
            logger.warning("Websocket stream is already running")
            return
            
        self.websocket_running = True
        self.websocket_thread = threading.Thread(
            target=self._websocket_worker,
            args=(symbol, channels, callback),
            daemon=True
        )
        self.websocket_thread.start()
        
        logger.info(f"Started websocket stream for {symbol} on channels {channels}")
    
    def _websocket_worker(
        self, 
        symbol: str, 
        channels: List[str],
        callback: Optional[Callable]
    ) -> None:
        """
        Worker thread for websocket simulation.
        
        Args:
            symbol: Trading pair symbol
            channels: List of channels to subscribe to
            callback: Callback function for data processing
        """
        update_interval = 1.0  # seconds
        
        try:
            while self.websocket_running:
                data = {}
                
                # Fetch data based on requested channels
                if 'ticker' in channels:
                    try:
                        ticker = self.get_ticker(symbol)
                        data['ticker'] = ticker
                    except Exception as e:
                        logger.error(f"Error fetching ticker: {e}")
                
                if 'orderbook' in channels:
                    try:
                        orderbook = self.get_orderbook(symbol, limit=10)
                        data['orderbook'] = orderbook
                    except Exception as e:
                        logger.error(f"Error fetching orderbook: {e}")
                
                if 'trades' in channels:
                    try:
                        trades = self.get_recent_trades(symbol, limit=10)
                        data['trades'] = trades.to_dict('records') if not trades.empty else []
                    except Exception as e:
                        logger.error(f"Error fetching trades: {e}")
                
                # Add timestamp
                data['timestamp'] = datetime.now().isoformat()
                data['symbol'] = symbol
                
                # Put data in queue
                self.data_queue.put(data)
                
                # Call callback if provided
                if callback and callable(callback):
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in websocket callback: {e}")
                
                # Sleep until next update
                time.sleep(update_interval)
                
        except Exception as e:
            logger.error(f"Error in websocket worker: {e}")
        finally:
            self.websocket_running = False
            logger.info("Websocket worker stopped")
    
    def stop_websocket_stream(self) -> None:
        """Stop the websocket stream"""
        if self.websocket_running:
            self.websocket_running = False
            if self.websocket_thread:
                self.websocket_thread.join(timeout=5.0)
            logger.info("Stopped websocket stream")
    
    def get_latest_websocket_data(self) -> Optional[Dict]:
        """
        Get the latest data from the websocket stream.
        
        Returns:
            Dictionary with latest data or None if queue is empty
        """
        if not self.websocket_running:
            logger.warning("Websocket stream is not running")
            return None
            
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by the exchange.
        
        Args:
            symbol: Trading pair symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            # Reload markets if needed
            if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                self._load_exchange_info()
                
            return symbol in self.exchange.markets
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate if a timeframe is supported by the exchange.
        
        Args:
            timeframe: Timeframe string to validate
            
        Returns:
            True if timeframe is valid, False otherwise
        """
        try:
            if hasattr(self.exchange, 'timeframes'):
                return timeframe in self.exchange.timeframes
            return False
        except Exception as e:
            logger.error(f"Error validating timeframe {timeframe}: {e}")
            return False
    
    def clear_cache(self, older_than: Optional[int] = None) -> None:
        """
        Clear the data cache.
        
        Args:
            older_than: If provided, only clear entries older than this many seconds
        """
        self.cache.clear(older_than)
        logger.info(f"Cleared data cache {'(older entries)' if older_than else '(all)'}")
    
    def get_historical_data_as_dict(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[Union[int, str, datetime]] = None,
        limit: Optional[int] = None,
        end_time: Optional[Union[int, str, datetime]] = None,
    ) -> Dict:
        """
        Get historical data as a dictionary (useful for API responses).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            since: Start time as timestamp, ISO string, or datetime object
            limit: Maximum number of candles to retrieve
            end_time: End time as timestamp, ISO string, or datetime object
            
        Returns:
            Dictionary with historical data
        """
        df = self.get_historical_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
            end_time=end_time
        )
        
        if df.empty:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': []
            }
        
        # Reset index to include timestamp in records
        df = df.reset_index()
        
        # Convert to ISO format for better JSON serialization
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': df.to_dict('records')
        }


# Singleton instance for global access
_data_fetcher_instance = None

def get_data_fetcher() -> DataFetcher:
    """
    Get or create a singleton instance of DataFetcher.
    
    Returns:
        DataFetcher instance
    """
    global _data_fetcher_instance
    
    if _data_fetcher_instance is None:
        _data_fetcher_instance = DataFetcher(
            exchange_id=config.API_CONFIG["exchange"],
            api_key=config.API_CONFIG["api_key"],
            api_secret=config.API_CONFIG["api_secret"],
            use_testnet=config.API_CONFIG["testnet"],
            timeout=config.API_CONFIG["timeout"],
            rate_limit=config.API_CONFIG["rate_limit"]
        )
    
    return _data_fetcher_instance


if __name__ == "__main__":
    # Example usage
    fetcher = get_data_fetcher()
    
    # Get exchange info
    print("Exchange Info:")
    info = fetcher.get_exchange_info()
    print(f"Exchange: {info['name']}")
    print(f"Supported timeframes: {info['timeframes']}")
    print(f"Number of trading pairs: {len(info['pairs'])}")
    
    # Get historical data for BTC/USDT
    print("\nFetching historical data for BTC/USDT (15m timeframe)...")
    df = fetcher.get_historical_ohlcv(
        symbol="BTC/USDT",
        timeframe="15m",
        limit=10
    )
    
    if not df.empty:
        print(df)
    else:
        print("No data available")
