"""
Neural Pulse Scalper - Main Application Entry Point

This script serves as the main entry point for the Neural Pulse Scalper bot.
It handles configuration loading, logging setup, component initialization, and
provides a command-line interface to run the bot in different modes:
- trade: Live or paper trading mode.
- backtest: (Not yet implemented) Backtesting mode.
- train: AI/ML model training mode.

Usage:
    python -m src.main trade
    python -m src.main train
    python -m src.main backtest
"""

import os
import sys
import yaml
import argparse
import logging
import signal
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import pandas as pd
from sklearn.impute import SimpleImputer

# Add src to the Python path
# This allows us to run the script from the root directory
# (e.g., python src/main.py) and have imports work correctly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binance_client import BinanceClient
from indicators import TechnicalIndicators
from strategy_engine import StrategyEngine
from ai_model import ModelFactory, FeatureEngineer

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Global variable to hold the main application instance for the signal handler
app_instance = None

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Loads configuration from a YAML file and merges it with environment variables.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the final configuration.
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Exiting.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}. Exiting.")
        sys.exit(1)

    # Load environment variables from .env file
    load_dotenv()
    logger.info("Loaded environment variables from .env file.")

    # Override YAML config with environment variables
    # Example: BINANCE_MAINNET_API_KEY env var overrides config['api']['binance']['mainnet']['api_key']
    api_cfg = config['api']['binance']
    api_cfg['mainnet']['api_key'] = os.getenv('BINANCE_MAINNET_API_KEY', api_cfg['mainnet']['api_key'])
    api_cfg['mainnet']['api_secret'] = os.getenv('BINANCE_MAINNET_API_SECRET', api_cfg['mainnet']['api_secret'])
    api_cfg['testnet']['api_key'] = os.getenv('BINANCE_TESTNET_API_KEY', api_cfg['testnet']['api_key'])
    api_cfg['testnet']['api_secret'] = os.getenv('BINANCE_TESTNET_API_SECRET', api_cfg['testnet']['api_secret'])

    config['trading_mode'] = os.getenv('TRADING_MODE', 'testnet')
    
    logger.info("Configuration loaded successfully.")
    return config

def setup_logging(config: Dict[str, Any]):
    """
    Sets up the logging configuration for the application.
    """
    log_cfg = config.get('logging', {})
    log_level = log_cfg.get('level', 'INFO').upper()
    log_file_enabled = log_cfg.get('file', {}).get('enabled', False)
    log_file_path = log_cfg.get('file', {}).get('path', 'logs/app.log')

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Console handler
    if log_cfg.get('console', {}).get('enabled', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file_enabled:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured. Level: {log_level}, File logging: {'Enabled' if log_file_enabled else 'Disabled'}")

class NeuralPulseTrader:
    """
    Main application class that orchestrates all components of the trading bot.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the application and all its components.
        """
        self.config = config
        self.is_running = False

        # Determine trading mode
        self.is_testnet = self.config.get('trading_mode', 'testnet') == 'testnet'
        api_creds = self.config['api']['binance']['testnet' if self.is_testnet else 'mainnet']
        
        # Initialize components
        logger.info("Initializing components...")
        self.client = BinanceClient(
            api_key=api_creds['api_key'],
            api_secret=api_creds['api_secret'],
            testnet=self.is_testnet
        )
        self.indicators = TechnicalIndicators()
        self.strategy_engine = StrategyEngine(self.config, self.client, self.indicators)
        
        if self.config['ai_model']['enabled']:
            self.feature_engineer = FeatureEngineer(self.config)
            self.ai_model = ModelFactory.create_model(self.config)
        else:
            self.feature_engineer = None
            self.ai_model = None
            
        logger.info("All components initialized.")

    def run_trade_mode(self):
        """
        Starts the bot in live or paper trading mode.
        """
        if self.is_running:
            logger.warning("Trading bot is already running.")
            return

        logger.info(f"Starting trading bot in {'TESTNET' if self.is_testnet else 'MAINNET'} mode...")
        self.is_running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        self.strategy_engine.run()

    def run_train_mode(self):
        """
        Runs the AI/ML model training process.
        """
        if not self.config['ai_model']['enabled']:
            logger.error("AI model is disabled in the configuration. Cannot run training.")
            return

        logger.info("Starting AI model training process...")
        try:
            # 1. Load data
            # For simplicity, we use one symbol. A real scenario might use multiple.
            train_cfg = self.config['ai_model']['training']
            symbol = self.config['trading']['custom_pairs'][0]
            logger.info(f"Loading historical data for {symbol} for training...")
            df = self.client.get_historical_klines(
                symbol=symbol,
                interval=self.config['strategy']['timeframes']['primary'],
                start_str=f"{train_cfg['data_lookback_days']} days ago UTC"
            )
            
            # 2. Add indicators (features)
            logger.info("Calculating indicators and features...")
            df_with_indicators = self.indicators.add_all_indicators(df, self.config)
            
            # 3. Prepare data
            X, y = self.feature_engineer.prepare_data_for_training(df_with_indicators)
            
            if X.empty or y.empty:
                logger.error("No data available for training after feature engineering. Exiting.")
                return

            # Handle potential NaN values in features after alignment
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

            logger.info(f"Training data prepared. Features: {X_imputed.shape[1]}, Samples: {X_imputed.shape[0]}")
            
            # 4. Train model
            logger.info(f"Training '{self.config['ai_model']['model_type']}' model...")
            training_results = self.ai_model.train(X_imputed, y)
            logger.info(f"Model training complete. Evaluation results: {training_results}")
            
            # 5. Save model
            model_path = self.config['ai_model']['model_path']
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.ai_model.save(model_path)

        except Exception as e:
            logger.error(f"An error occurred during model training: {e}", exc_info=True)

    def run_backtest_mode(self):
        """
        (Placeholder) Runs the strategy backtesting process.
        """
        logger.warning("Backtesting mode is not yet implemented.")
        # Future implementation:
        # 1. Initialize a backtesting engine.
        # 2. Load historical data from file or API.
        # 3. Run the strategy engine over the historical data.
        # 4. Generate and display performance report.
        print("Backtesting mode is under construction. Please check back later.")

    def shutdown(self, signum, frame):
        """
        Handles graceful shutdown of the application.
        """
        logger.warning(f"Shutdown signal {signum} received. Stopping application...")
        if self.is_running:
            self.strategy_engine.stop()
            self.client.close()
        self.is_running = False
        logger.info("Application shut down gracefully.")
        sys.exit(0)

def main():
    """
    Main function to parse arguments and run the application.
    """
    parser = argparse.ArgumentParser(description="Neural Pulse Scalper - A crypto trading bot.")
    parser.add_argument(
        'mode',
        choices=['trade', 'backtest', 'train'],
        help="The mode to run the application in."
    )
    args = parser.parse_args()

    # Load configuration and set up logging
    config = load_config()
    setup_logging(config)

    # Create and run the application
    global app_instance
    try:
        app_instance = NeuralPulseTrader(config)
        if args.mode == 'trade':
            app_instance.run_trade_mode()
        elif args.mode == 'train':
            app_instance.run_train_mode()
        elif args.mode == 'backtest':
            app_instance.run_backtest_mode()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        if app_instance:
            app_instance.shutdown(signal.SIGABRT, None)
        sys.exit(1)

if __name__ == "__main__":
    main()
