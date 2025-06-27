#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for Crypto Trading Analysis Application

This file contains all configurable parameters for the trading bot, including:
- API settings
- Trading parameters
- Technical indicators configuration
- Risk management settings
- Backtesting parameters
- UI configuration
- Logging settings

All values can be overridden via environment variables or through the UI.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, LOG_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Application settings
APP_NAME = "Crypto Trading Analyzer"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# API Configuration
API_CONFIG = {
    "exchange": os.getenv("EXCHANGE", "binance"),
    "api_key": os.getenv("API_KEY", ""),
    "api_secret": os.getenv("API_SECRET", ""),
    "testnet": os.getenv("USE_TESTNET", "True").lower() == "true",
    "timeout": int(os.getenv("API_TIMEOUT", "10")),
    "rate_limit": True,
}

# Available trading pairs (default options)
TRADING_PAIRS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
    "XRP/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT",
    "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT"
]

# Available timeframes
TIMEFRAMES = {
    "1m": {"label": "1 minute", "seconds": 60},
    "3m": {"label": "3 minutes", "seconds": 180},
    "5m": {"label": "5 minutes", "seconds": 300},
    "15m": {"label": "15 minutes", "seconds": 900},
    "30m": {"label": "30 minutes", "seconds": 1800},
    "1h": {"label": "1 hour", "seconds": 3600},
    "2h": {"label": "2 hours", "seconds": 7200},
    "4h": {"label": "4 hours", "seconds": 14400},
    "6h": {"label": "6 hours", "seconds": 21600},
    "12h": {"label": "12 hours", "seconds": 43200},
    "1d": {"label": "1 day", "seconds": 86400},
    "3d": {"label": "3 days", "seconds": 259200},
    "1w": {"label": "1 week", "seconds": 604800},
}

# Default trading parameters
DEFAULT_TRADING_CONFIG = {
    "trading_pair": "BTC/USDT",
    "timeframe": "15m",
    "capital_amount": 1000.0,
    "risk_level": "medium",  # low, medium, high
    "max_open_trades": 3,
    "position_size_pct": 20,  # percentage of capital per trade
    "auto_trade": False,  # default to simulation mode
}

# Risk management settings
RISK_LEVELS = {
    "low": {
        "max_position_size_pct": 10,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "max_daily_drawdown_pct": 5.0,
        "trailing_stop_activation_pct": 2.5,
        "trailing_stop_distance_pct": 1.5,
    },
    "medium": {
        "max_position_size_pct": 20,
        "stop_loss_pct": 3.0,
        "take_profit_pct": 6.0,
        "max_daily_drawdown_pct": 10.0,
        "trailing_stop_activation_pct": 3.5,
        "trailing_stop_distance_pct": 2.0,
    },
    "high": {
        "max_position_size_pct": 30,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "max_daily_drawdown_pct": 15.0,
        "trailing_stop_activation_pct": 5.0,
        "trailing_stop_distance_pct": 3.0,
    }
}

# Technical indicators configuration
INDICATOR_SETTINGS = {
    "rsi": {
        "enabled": True,
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "weight": 1.0,
    },
    "macd": {
        "enabled": True,
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "weight": 1.0,
    },
    "bollinger_bands": {
        "enabled": True,
        "period": 20,
        "std_dev": 2.0,
        "weight": 1.0,
    },
    "ema": {
        "enabled": True,
        "fast_period": 9,
        "slow_period": 21,
        "weight": 1.0,
    },
    "volume_profile": {
        "enabled": True,
        "period": 14,
        "weight": 0.8,
    },
    "stochastic": {
        "enabled": True,
        "k_period": 14,
        "d_period": 3,
        "overbought": 80,
        "oversold": 20,
        "weight": 0.7,
    },
    "atr": {
        "enabled": True,
        "period": 14,
        "weight": 0.5,
    },
    "ichimoku": {
        "enabled": False,
        "conversion_line_period": 9,
        "base_line_period": 26,
        "lagging_span_period": 52,
        "displacement": 26,
        "weight": 0.6,
    }
}

# Strategy configuration
STRATEGY_CONFIG = {
    "name": "Multi-Indicator Strategy",
    "description": "A strategy combining multiple technical indicators with confirmation layers",
    "signal_threshold": 0.7,  # Minimum combined signal strength (0-1) to trigger a trade
    "confirmation_needed": 2,  # Number of indicators that must confirm a signal
    "trend_confirmation": True,  # Require trend confirmation before trading
    "use_market_sentiment": True,  # Consider overall market sentiment
    "exit_on_opposite_signal": True,  # Exit position when opposite signal appears
    "use_trailing_stop": True,  # Use trailing stops for exit
    "enable_dynamic_sizing": True,  # Adjust position size based on signal strength
}

# Backtesting configuration
BACKTEST_CONFIG = {
    "default_start_date": "2023-01-01",
    "default_end_date": "now",  # 'now' will be replaced with current date
    "commission_pct": 0.1,  # Default commission percentage (0.1%)
    "slippage_pct": 0.05,  # Default slippage percentage (0.05%)
    "initial_capital": 10000,  # Default initial capital for backtesting
    "data_source": "binance",  # Default data source
    "include_trading_fees": True,
    "plot_equity_curve": True,
    "save_results": True,
}

# UI Configuration
UI_CONFIG = {
    "theme": "light",  # light or dark
    "chart_height": 600,
    "refresh_interval": 60,  # seconds
    "show_trade_markers": True,
    "show_indicators_on_chart": True,
    "max_displayed_trades": 50,
    "notifications_enabled": True,
    "sound_alerts": False,
}

# Logging configuration
LOG_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "log_trades": True,
    "log_signals": True,
    "log_performance": True,
    "max_log_files": 10,
    "max_file_size_mb": 10,
}

# Database configuration
DB_CONFIG = {
    "engine": "sqlite",  # sqlite, mysql, postgresql
    "path": str(DATA_DIR / "trading.db"),  # for sqlite
    "host": "localhost",  # for mysql/postgresql
    "port": 3306,  # for mysql/postgresql
    "username": "",  # for mysql/postgresql
    "password": "",  # for mysql/postgresql
    "database": "trading",  # for mysql/postgresql
}

# Email notification settings (optional)
EMAIL_CONFIG = {
    "enabled": False,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "",
    "password": "",
    "from_email": "",
    "to_email": "",
    "use_tls": True,
}

# Disclaimer message (important for legal compliance)
DISCLAIMER = """
IMPORTANT DISCLAIMER:
This software is for educational and informational purposes only.
It is not financial advice and comes with absolutely no warranty.
Cryptocurrency trading involves substantial risk and may not be suitable for everyone.
Past performance is not indicative of future results.
You are solely responsible for your trading decisions and any resulting gains or losses.
Always do your own research before trading.
"""

# Function to get config as dict (useful for serialization)
def get_config_dict() -> Dict[str, Any]:
    """Return all configuration as a dictionary for serialization"""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "api_config": API_CONFIG,
        "trading_pairs": TRADING_PAIRS,
        "timeframes": TIMEFRAMES,
        "default_trading_config": DEFAULT_TRADING_CONFIG,
        "risk_levels": RISK_LEVELS,
        "indicator_settings": INDICATOR_SETTINGS,
        "strategy_config": STRATEGY_CONFIG,
        "backtest_config": BACKTEST_CONFIG,
        "ui_config": UI_CONFIG,
        "log_config": LOG_CONFIG,
        "db_config": DB_CONFIG,
        "email_config": EMAIL_CONFIG,
        "disclaimer": DISCLAIMER,
    }
