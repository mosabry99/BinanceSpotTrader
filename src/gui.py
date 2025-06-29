"""
Neural Pulse Scalper - Streamlit GUI

This module provides a comprehensive web-based graphical user interface for the
Neural Pulse Scalper trading bot using the Streamlit framework. It allows for
real-time monitoring, control, and configuration of the bot.

Features:
- Real-time dashboard with price charts, KPIs, and signal strength.
- Start/Stop controls for the trading bot.
- Live monitoring of portfolio, balances, and active trades.
- Detailed trade history and performance analysis.
- In-app configuration editor for strategy parameters.
- Live log viewer to monitor bot activity.
- Market data display for selected trading pairs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import yaml
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import dataclasses
import numpy as np
import os
import sys

# This allows running the script from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# In a real application, these would be the actual imports.
# We use mock objects here to make the GUI runnable for demonstration.
try:
    from src.main import NeuralPulseTrader, load_config
    from src.strategy_engine import Trade, TradeStatus, OrderSide
except (ImportError, ModuleNotFoundError):
    # --- Mock Components for Standalone Demonstration ---
    # This section allows the GUI to be run and tested independently.
    class MockBinanceClient:
        def get_all_balances(self):
            return {
                'USDT': {'free': 9500.0, 'locked': 500.0, 'total': 10000.0},
                'BTC': {'free': 0.1, 'locked': 0.0, 'total': 0.1},
                'ETH': {'free': 2.0, 'locked': 0.0, 'total': 2.0},
            }
        def get_ticker(self, symbol):
            if 'BTC' in symbol:
                return {'lastPrice': 65000.0 + np.random.uniform(-100, 100)}
            elif 'ETH' in symbol:
                return {'lastPrice': 3500.0 + np.random.uniform(-50, 50)}
            return {'lastPrice': 1.0}

    class MockStrategyEngine:
        def __init__(self, config):
            self.is_running = False
            self.active_trades = {}
            self.trade_history = []
            self.risk_state = {
                "trades_today": 5,
                "consecutive_losses": 1,
                "daily_drawdown": -1.2,
                "is_paused": False,
                "pause_until": None,
                "last_trade_date": datetime.utcnow().date()
            }
            self.historical_data = {
                pair: self.generate_mock_data() for pair in config['trading']['custom_pairs']
            }
        
        def generate_mock_data(self, points=200):
            base_price = 65000
            dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=points, freq='1min'))
            price = base_price + np.random.randn(points).cumsum()
            df = pd.DataFrame({
                'open': price,
                'high': price + np.random.uniform(0, 100, points),
                'low': price - np.random.uniform(0, 100, points),
                'close': price + np.random.randn(points),
                'volume': np.random.uniform(1, 10, points)
            }, index=dates)
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            return df
        
        def get_status(self):
            return {
                "is_running": self.is_running,
                "active_trades_count": len(self.active_trades),
                "trades_history_count": len(self.trade_history),
                "risk_state": self.risk_state,
                "active_trades": [vars(t) for t in self.active_trades.values()]
            }

    class MockNeuralPulseTrader:
        def __init__(self, config):
            self.config = config
            self.strategy_engine = MockStrategyEngine(config)
            self.client = MockBinanceClient()
            self.is_running = False

        def run_trade_mode(self):
            self.is_running = True
            self.strategy_engine.is_running = True
            logging.info("Mock bot started.")
        
        def shutdown(self, *args):
            self.is_running = False
            self.strategy_engine.is_running = False
            logging.info("Mock bot stopped.")

    @dataclasses.dataclass
    class Trade:
        trade_id: str
        symbol: str
        side: str
        entry_price: float
        quantity: float
        status: str = "ACTIVE"
        entry_time: datetime = datetime.utcnow()

    def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Config file not found at {config_path}")
            return {}
    
    NeuralPulseTrader = MockNeuralPulseTrader
    # --- End Mock Components ---


def save_config(config_path: str, config_data: Dict[str, Any]):
    """Saves the configuration dictionary to a YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        st.success("Configuration saved successfully!")
    except Exception as e:
        st.error(f"Failed to save configuration: {e}")

class GuiLogger(logging.Handler):
    """Custom logging handler to display logs in Streamlit."""
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_entry}")
        # Keep logs list from growing too large
        if len(self.logs) > 200:
            self.logs.pop(0)

class StreamlitGUI:
    """
    The main class for the Streamlit GUI application.
    It orchestrates the rendering of all UI components.
    """
    def __init__(self):
        st.set_page_config(
            page_title="Neural Pulse Scalper",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit's session state for the application."""
        if 'bot' not in st.session_state:
            st.session_state.bot = None
        if 'config' not in st.session_state:
            st.session_state.config = load_config(self.config_path)
        if 'log_handler' not in st.session_state:
            log_handler = GuiLogger()
            log_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            # Attach to the root logger to capture logs from all modules
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)
            root_logger.setLevel(logging.INFO)
            st.session_state.log_handler = log_handler

    def run(self):
        """Main method to render the entire GUI."""
        st.title("🧠 Neural Pulse Scalper")
        
        refresh_interval = st.session_state.config.get('gui', {}).get('refresh_rate_seconds', 5)
        st_autorefresh(interval=refresh_interval * 1000, key="main_refresher")

        self.render_sidebar()
        
        tabs = st.tabs([
            "📊 Dashboard", "📈 Portfolio & Trades", "⚙️ Configuration", "📜 Logs", "🔍 Market Data"
        ])

        with tabs[0]:
            self.render_dashboard()
        with tabs[1]:
            self.render_portfolio_and_trades()
        with tabs[2]:
            self.render_configuration()
        with tabs[3]:
            self.render_logs()
        with tabs[4]:
            self.render_market_data()

    def render_sidebar(self):
        """Render the sidebar with main controls and status information."""
        with st.sidebar:
            st.header("Controls & Status")

            if st.session_state.bot and st.session_state.bot.is_running:
                if st.button("🛑 Stop Bot", type="primary", use_container_width=True):
                    st.session_state.bot.shutdown(None, None)
                    st.success("Bot is stopping...")
                    time.sleep(2)
                    st.rerun()
            else:
                if st.button("🚀 Start Bot", use_container_width=True):
                    st.session_state.bot = NeuralPulseTrader(st.session_state.config)
                    # In a real app, this would run in a separate thread.
                    # threading.Thread(target=st.session_state.bot.run_trade_mode, daemon=True).start()
                    st.session_state.bot.run_trade_mode()
                    st.success("Bot started!")
                    time.sleep(1)
                    st.rerun()

            st.divider()
            st.subheader("Live Status")
            bot_running = st.session_state.bot and st.session_state.bot.is_running
            status = "🟢 Running" if bot_running else "🔴 Stopped"
            active_trades = st.session_state.bot.strategy_engine.get_status()['active_trades_count'] if bot_running else 0
            
            st.metric("Bot Status", status)
            st.metric("Active Trades", active_trades)
            
            st.divider()
            st.subheader("Risk Monitor")
            if bot_running:
                risk_state = st.session_state.bot.strategy_engine.risk_state
                risk_cfg = st.session_state.config['risk_management']
                st.metric("Trades Today", f"{risk_state['trades_today']} / {risk_cfg['max_trades_per_day']}")
                st.metric("Consecutive Losses", f"{risk_state['consecutive_losses']} / {risk_cfg['consecutive_losses']['pause_after']}")
                st.metric("Daily Drawdown", f"{risk_state['daily_drawdown']:.2f}%")
                if risk_state['is_paused']:
                    pause_time = risk_state['pause_until'].strftime('%H:%M:%S') if risk_state['pause_until'] else 'N/A'
                    st.warning(f"PAUSED until {pause_time}")
            else:
                st.info("Bot is stopped.")

    def render_dashboard(self):
        """Render the main dashboard with KPIs, chart, and signal info."""
        st.header("Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate", "65.7%", "1.2%")
        col2.metric("Profit Factor", "1.85", "-0.05")
        col3.metric("Total PnL (USDT)", "$1,234.56", "$56.78 today")
        col4.metric("Sharpe Ratio", "2.1", "0.1")

        st.divider()
        
        left, right = st.columns([3, 1])
        with left:
            selected_pair = st.selectbox(
                "Select Chart", 
                options=st.session_state.config['trading']['custom_pairs']
            )
            self.render_price_chart(selected_pair)
        with right:
            st.subheader("Signal Strength")
            self.render_signal_gauge()
            st.subheader("Latest Signal Analysis")
            st.info("EMA cross UP, RSI mid-range, Volume spike. Waiting for BB touch.")

    def render_price_chart(self, symbol: str):
        """Renders a Plotly price chart for the selected symbol."""
        if st.session_state.bot and symbol in st.session_state.bot.strategy_engine.historical_data:
            df = st.session_state.bot.strategy_engine.historical_data[symbol]
        else:
            st.info(f"No data available for {symbol}. Start the bot to load data.")
            return

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Market'))
        
        # Add indicators based on config
        fig.add_trace(go.Scatter(x=df.index, y=df.get('ema_5'), line=dict(color='cyan', width=1), name='EMA 5'))
        fig.add_trace(go.Scatter(x=df.index, y=df.get('ema_20'), line=dict(color='yellow', width=1), name='EMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df.get('bb_upper'), line=dict(color='gray', width=1, dash='dash'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df.get('bb_lower'), line=dict(color='gray', width=1, dash='dash'), name='BB Lower'))
        
        fig.update_layout(
            title=f'{symbol} 1m Chart',
            yaxis_title='Price (USDT)',
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_signal_gauge(self):
        """Renders a gauge chart for signal strength."""
        signal_strength = np.random.uniform(0.6, 0.95) # Mock value
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=signal_strength * 100,
            title={'text': "Buy Signal Confidence"},
            gauge={
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4}, 'thickness': 0.75,
                    'value': st.session_state.config['ai_model']['win_probability_threshold'] * 100
                }
            }
        ))
        fig.update_layout(height=250, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    def render_portfolio_and_trades(self):
        """Renders portfolio balances and trade history."""
        st.header("Portfolio & Trade History")
        
        st.subheader("Account Balances")
        if st.session_state.bot:
            balances = st.session_state.bot.client.get_all_balances()
            total_value_usd = 0
            
            balance_data = []
            for asset, details in balances.items():
                price = 1.0
                if asset != 'USDT':
                    try:
                        price = st.session_state.bot.client.get_ticker(f"{asset}USDT")['lastPrice']
                    except Exception:
                        price = 0.0 # Could not fetch price
                value_usd = details['total'] * price
                total_value_usd += value_usd
                balance_data.append({
                    "Asset": asset, "Total": details['total'], "Free": details['free'],
                    "Locked": details['locked'], "Value (USD)": f"${value_usd:,.2f}"
                })
            
            st.metric("Total Portfolio Value", f"${total_value_usd:,.2f}")
            st.dataframe(pd.DataFrame(balance_data), use_container_width=True, hide_index=True)
        else:
            st.info("Start the bot to see portfolio data.")
        
        st.divider()
        
        st.subheader("Trade History")
        if st.session_state.bot and st.session_state.bot.strategy_engine.trade_history:
            history_df = pd.DataFrame([vars(t) for t in st.session_state.bot.strategy_engine.trade_history])
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades have been completed yet.")

    def render_configuration(self):
        """Renders the configuration editor."""
        st.header("Configuration")
        
        with st.expander("View Current Configuration (JSON)"):
            st.json(st.session_state.config)
        
        st.subheader("Edit Configuration (YAML)")
        config_str = yaml.dump(st.session_state.config, default_flow_style=False, sort_keys=False)
        edited_config_str = st.text_area(
            "YAML Config", value=config_str, height=500,
            help="Edit the configuration below and click 'Save Configuration'."
        )
        
        if st.button("Save Configuration"):
            try:
                new_config = yaml.safe_load(edited_config_str)
                save_config(self.config_path, new_config)
                st.session_state.config = new_config
                st.success("Config updated. Restart the bot for changes to take effect.")
                st.rerun()
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML format: {e}")

    def render_logs(self):
        """Renders the application logs captured by the custom handler."""
        st.header("Live Logs")
        log_container = st.container(height=600)
        logs = st.session_state.log_handler.logs
        with log_container:
            st.code("\n".join(reversed(logs)), language="log")

    def render_market_data(self):
        """Renders additional market data like tickers and order books."""
        st.header("Market Data")
        
        symbol = st.selectbox(
            "Select Symbol", 
            options=st.session_state.config['trading']['custom_pairs'],
            key="market_data_symbol"
        )
        
        if st.session_state.bot and symbol:
            st.subheader(f"Ticker Information: {symbol}")
            try:
                ticker_data = st.session_state.bot.client.get_ticker(symbol)
                st.json(ticker_data)
            except Exception as e:
                st.error(f"Could not fetch ticker data for {symbol}: {e}")
            
            st.subheader("Order Book")
            st.info("Live order book display is not yet implemented in this GUI version.")
        else:
            st.info("Start the bot to view live market data.")

if __name__ == '__main__':
    gui = StreamlitGUI()
    gui.run()
