"""
Neural Pulse Scalper - Source Package

This package contains all the core source code for the trading bot,
including the Binance client, strategy engine, technical indicators,
AI models, and the main application logic.
"""

__version__ = "1.0.0"
__author__ = "Neural Pulse Scalper Team"
__email__ = "contact@example.com"

# Import key classes to make them accessible at the package level
from .binance_client import BinanceClient, OrderSide, OrderType
from .indicators import TechnicalIndicators
from .strategy_engine import StrategyEngine, Trade, TradeStatus
from .ai_model import ModelFactory, FeatureEngineer, AIModel
from .main import NeuralPulseTrader

# Define the public API of the package
__all__ = [
    'BinanceClient',
    'OrderSide',
    'OrderType',
    'TechnicalIndicators',
    'StrategyEngine',
    'Trade',
    'TradeStatus',
    'ModelFactory',
    'FeatureEngineer',
    'AIModel',
    'NeuralPulseTrader',
]
