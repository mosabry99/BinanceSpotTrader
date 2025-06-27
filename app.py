#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Trading Analysis Application

A comprehensive Streamlit application for cryptocurrency trading analysis,
backtesting, and education. This application integrates real-time data,
technical indicators, strategy configuration, backtesting, and performance
analytics in a user-friendly interface.

Key features:
- Real-time market data visualization
- Multiple technical indicators
- Strategy configuration and backtesting
- Performance analytics and reporting
- Educational content on trading concepts
- Risk management tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
import json
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import io
import sys
from PIL import Image
import requests

# Add local modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
import config
from data_fetcher import get_data_fetcher, DataFetcher
from technical_indicators import TechnicalIndicators, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NEUTRAL, SIGNAL_STRONG_BUY, SIGNAL_STRONG_SELL
from strategy import get_strategy_by_name, MultiIndicatorStrategy, TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy
from backtester import Backtester, run_backtest, BacktestResults

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Crypto Trading Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
PAGES = {
    "Home": "home",
    "Market Data": "market_data",
    "Technical Analysis": "technical_analysis",
    "Strategy Configuration": "strategy_config",
    "Backtesting": "backtesting",
    "Performance Analytics": "performance",
    "Education": "education",
    "Settings": "settings",
}

TIMEFRAMES = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "2h": "2 hours",
    "4h": "4 hours",
    "6h": "6 hours",
    "12h": "12 hours",
    "1d": "1 day",
    "3d": "3 days",
    "1w": "1 week",
}

INDICATORS = {
    "rsi": "Relative Strength Index (RSI)",
    "macd": "Moving Average Convergence Divergence (MACD)",
    "bollinger_bands": "Bollinger Bands",
    "ema_crossover": "EMA Crossover",
    "stochastic": "Stochastic Oscillator",
    "adx": "Average Directional Index (ADX)",
    "atr": "Average True Range (ATR)",
    "parabolic_sar": "Parabolic SAR",
    "volume_profile": "Volume Profile",
    "support_resistance": "Support & Resistance",
    "fibonacci": "Fibonacci Retracement",
    "ichimoku": "Ichimoku Cloud",
}

STRATEGIES = {
    "multi_indicator": "Multi-Indicator Strategy",
    "trend_following": "Trend Following Strategy",
    "mean_reversion": "Mean Reversion Strategy",
    "breakout": "Breakout Strategy",
}

RISK_LEVELS = {
    "low": "Low Risk",
    "medium": "Medium Risk",
    "high": "High Risk",
}

# Helper functions
def load_css():
    """Load custom CSS"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0277BD;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #1E88E5;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #FFF8E1;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #FFC107;
            margin: 1rem 0;
        }
        .danger-box {
            background-color: #FFEBEE;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #F44336;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #4CAF50;
            margin: 1rem 0;
        }
        .indicator-card {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border: 1px solid #E0E0E0;
        }
        .metric-card {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem;
            border: 1px solid #E0E0E0;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #616161;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            border-top: 1px solid #E0E0E0;
            font-size: 0.8rem;
            color: #9E9E9E;
        }
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Make the app full width */
        .block-container {
            max-width: 100%;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def get_data_fetcher_instance():
    """Get or create data fetcher instance"""
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = get_data_fetcher()
    return st.session_state.data_fetcher

def get_indicator_calculator():
    """Get or create indicator calculator instance"""
    if 'indicator_calculator' not in st.session_state:
        st.session_state.indicator_calculator = TechnicalIndicators()
    return st.session_state.indicator_calculator

def get_backtester():
    """Get or create backtester instance"""
    if 'backtester' not in st.session_state:
        st.session_state.backtester = Backtester(
            initial_capital=st.session_state.get('initial_capital', 10000.0),
            commission_pct=st.session_state.get('commission_pct', 0.1),
            slippage_pct=st.session_state.get('slippage_pct', 0.05),
            risk_level=st.session_state.get('risk_level', 'medium')
        )
    return st.session_state.backtester

def fetch_data(symbol, timeframe, limit=500):
    """Fetch historical data for a symbol"""
    data_fetcher = get_data_fetcher_instance()
    
    try:
        df = data_fetcher.get_historical_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def apply_indicators(df, indicators=None):
    """Apply technical indicators to data"""
    if df is None or df.empty:
        return None
        
    calculator = get_indicator_calculator()
    
    try:
        df = calculator.apply_all_indicators(df, include=indicators)
        return df
    except Exception as e:
        st.error(f"Error applying indicators: {e}")
        return df

def get_signal_color(signal):
    """Get color for signal value"""
    if signal > 1:
        return "#00CC00"  # Strong Buy - Bright Green
    elif signal > 0:
        return "#4CAF50"  # Buy - Green
    elif signal < -1:
        return "#FF0000"  # Strong Sell - Bright Red
    elif signal < 0:
        return "#F44336"  # Sell - Red
    else:
        return "#9E9E9E"  # Neutral - Gray

def get_signal_emoji(signal):
    """Get emoji for signal value"""
    if signal > 1:
        return "🔥 Strong Buy"
    elif signal > 0:
        return "✅ Buy"
    elif signal < -1:
        return "⛔ Strong Sell"
    elif signal < 0:
        return "❌ Sell"
    else:
        return "⚪ Neutral"

def format_number(value, precision=2):
    """Format number with specified precision"""
    try:
        return f"{float(value):,.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"

def download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for an object"""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Create a download link
    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    return href

def plot_candlestick_chart(df, indicators=None, height=600):
    """Plot interactive candlestick chart with indicators"""
    if df is None or df.empty:
        return None
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.8, 0.2]
    )
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume trace
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Add indicators if available
    if indicators:
        # Add moving averages
        for period in [9, 20, 50, 200]:
            col = f'ema_{period}'
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=f"EMA {period}",
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if all(x in df.columns for x in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    name="BB Upper",
                    line=dict(width=1, dash='dash', color='rgba(100, 100, 100, 0.5)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    name="BB Lower",
                    line=dict(width=1, dash='dash', color='rgba(100, 100, 100, 0.5)')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_middle'],
                    name="BB Middle",
                    line=dict(width=1, dash='dash', color='rgba(100, 100, 100, 0.5)')
                ),
                row=1, col=1
            )
        
        # Add signals if available
        if 'strategy_signal' in df.columns:
            buy_signals = df[df['strategy_signal'] > 0]
            sell_signals = df[df['strategy_signal'] < 0]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['low'] * 0.99,
                        name="Buy Signal",
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color="green",
                            line=dict(width=1, color="darkgreen")
                        )
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['high'] * 1.01,
                        name="Sell Signal",
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color="red",
                            line=dict(width=1, color="darkred")
                        )
                    ),
                    row=1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title="Price Chart with Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        height=height,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_indicator_chart(df, indicator, height=400):
    """Plot a specific indicator chart"""
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    if indicator == 'rsi':
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    name="RSI",
                    line=dict(color='purple')
                )
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=70,
                x1=df.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=30,
                x1=df.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash")
            )
            
            fig.update_layout(
                title="Relative Strength Index (RSI)",
                yaxis=dict(title="RSI", range=[0, 100])
            )
    
    elif indicator == 'macd':
        if all(x in df.columns for x in ['macd', 'macd_signal']):
            # Create MACD line trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    name="MACD",
                    line=dict(color='blue')
                )
            )
            
            # Create signal line trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    name="Signal",
                    line=dict(color='red')
                )
            )
            
            # Create histogram
            if 'macd_hist' in df.columns:
                colors = ['green' if val > 0 else 'red' for val in df['macd_hist']]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['macd_hist'],
                        name="Histogram",
                        marker_color=colors
                    )
                )
            
            fig.update_layout(title="MACD")
    
    elif indicator == 'bollinger_bands':
        if all(x in df.columns for x in ['close', 'bb_upper', 'bb_lower', 'bb_middle']):
            # Add price
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    name="Price",
                    line=dict(color='black')
                )
            )
            
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    name="Upper Band",
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_middle'],
                    name="Middle Band",
                    line=dict(color='blue', dash='dash')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    name="Lower Band",
                    line=dict(color='green', dash='dash')
                )
            )
            
            fig.update_layout(title="Bollinger Bands")
    
    elif indicator == 'stochastic':
        if all(x in df.columns for x in ['stoch_k', 'stoch_d']):
            # Create K line trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['stoch_k'],
                    name="%K",
                    line=dict(color='blue')
                )
            )
            
            # Create D line trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['stoch_d'],
                    name="%D",
                    line=dict(color='red')
                )
            )
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=80,
                x1=df.index[-1],
                y1=80,
                line=dict(color="red", width=1, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=20,
                x1=df.index[-1],
                y1=20,
                line=dict(color="green", width=1, dash="dash")
            )
            
            fig.update_layout(
                title="Stochastic Oscillator",
                yaxis=dict(title="Value", range=[0, 100])
            )
    
    # Add more indicators as needed...
    
    # Update layout
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'page': 'home',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'limit': 500,
        'indicators': ['rsi', 'macd', 'bollinger_bands', 'ema_crossover'],
        'strategy': 'trend_following',
        'risk_level': 'medium',
        'initial_capital': 10000.0,
        'commission_pct': 0.1,
        'slippage_pct': 0.05,
        'auto_refresh': False,
        'refresh_interval': 60,
        'last_refresh': time.time(),
        'dark_mode': False,
        'show_signals': True,
        'show_indicators': True,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/stocks-growth.png", width=80)
    st.sidebar.title("Crypto Trading Analyzer")
    
    # Navigation
    st.sidebar.subheader("Navigation")
    
    for page_name, page_key in PAGES.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            st.session_state.page = page_key
            # Clear any page-specific session state
            for key in list(st.session_state.keys()):
                if key.startswith(f"{page_key}_"):
                    del st.session_state[key]
    
    st.sidebar.markdown("---")
    
    # Symbol and timeframe selector (available on all pages)
    st.sidebar.subheader("Market Selection")
    
    # Fetch available trading pairs
    data_fetcher = get_data_fetcher_instance()
    try:
        available_pairs = data_fetcher.get_supported_pairs()
        if not available_pairs:
            available_pairs = config.TRADING_PAIRS
    except:
        available_pairs = config.TRADING_PAIRS
    
    # Symbol selector
    selected_symbol = st.sidebar.selectbox(
        "Trading Pair",
        options=available_pairs,
        index=available_pairs.index(st.session_state.symbol) if st.session_state.symbol in available_pairs else 0,
        key="sidebar_symbol"
    )
    
    if selected_symbol != st.session_state.symbol:
        st.session_state.symbol = selected_symbol
        # Clear any data-specific session state
        for key in ['data', 'data_with_indicators', 'backtest_results']:
            if key in st.session_state:
                del st.session_state[key]
    
    # Timeframe selector
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=list(TIMEFRAMES.keys()),
        format_func=lambda x: TIMEFRAMES[x],
        index=list(TIMEFRAMES.keys()).index(st.session_state.timeframe) if st.session_state.timeframe in TIMEFRAMES else 0,
        key="sidebar_timeframe"
    )
    
    if selected_timeframe != st.session_state.timeframe:
        st.session_state.timeframe = selected_timeframe
        # Clear any timeframe-specific session state
        for key in ['data', 'data_with_indicators', 'backtest_results']:
            if key in st.session_state:
                del st.session_state[key]
    
    # Auto-refresh toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Settings")
    
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh data",
        value=st.session_state.auto_refresh,
        key="sidebar_auto_refresh"
    )
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh interval (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=10,
            key="sidebar_refresh_interval"
        )
        st.session_state.refresh_interval = refresh_interval
    
    st.session_state.auto_refresh = auto_refresh
    
    # Manual refresh button
    if st.sidebar.button("Refresh Data Now"):
        # Clear data cache
        for key in ['data', 'data_with_indicators']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_refresh = time.time()
    
    # Show last refresh time
    if 'last_refresh' in st.session_state:
        last_refresh = datetime.datetime.fromtimestamp(st.session_state.last_refresh)
        st.sidebar.caption(f"Last refreshed: {last_refresh.strftime('%H:%M:%S')}")
    
    # Add disclaimer to sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption("""
    **DISCLAIMER:** This application is for educational purposes only. 
    Cryptocurrency trading involves substantial risk and may not be suitable for everyone.
    """)

def render_home_page():
    """Render the home page"""
    st.markdown('<h1 class="main-header">Welcome to Crypto Trading Analyzer</h1>', unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    This application provides tools for cryptocurrency trading analysis, backtesting, and education.
    Use the sidebar to navigate between different sections of the app.
    """)
    
    # Important disclaimer
    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
    st.markdown(f"""
    ### Important Disclaimer
    
    {config.DISCLAIMER}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown('<h2 class="sub-header">Quick Start Guide</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 1. Market Data
        
        View real-time cryptocurrency market data and charts.
        
        - Select your trading pair and timeframe
        - View candlestick charts
        - Monitor price movements
        """)
        if st.button("Go to Market Data", key="home_market_btn"):
            st.session_state.page = "market_data"
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 2. Technical Analysis
        
        Apply and visualize technical indicators.
        
        - RSI, MACD, Bollinger Bands
        - Support/Resistance levels
        - Trend analysis
        """)
        if st.button("Go to Technical Analysis", key="home_ta_btn"):
            st.session_state.page = "technical_analysis"
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 3. Backtesting
        
        Test trading strategies on historical data.
        
        - Configure strategy parameters
        - Run backtests
        - Analyze performance
        """)
        if st.button("Go to Backtesting", key="home_backtest_btn"):
            st.session_state.page = "backtesting"
        st.markdown('</div>', unsafe_allow_html=True)
    
    # App features
    st.markdown('<h2 class="sub-header">Key Features</h2>', unsafe_allow_html=True)
    
    features = [
        "📊 Real-time market data and visualization",
        "📈 Multiple technical indicators",
        "🔍 Advanced strategy configuration",
        "🧪 Comprehensive backtesting",
        "📉 Detailed performance analytics",
        "🛠️ Risk management tools",
        "📚 Educational resources",
        "💾 Export functionality for further analysis"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        if i % 2 == 0:
            col1.markdown(f"- {feature}")
        else:
            col2.markdown(f"- {feature}")
    
    # Recent market summary
    st.markdown('<h2 class="sub-header">Recent Market Summary</h2>', unsafe_allow_html=True)
    
    try:
        # Fetch recent data for current symbol
        data_fetcher = get_data_fetcher_instance()
        ticker = data_fetcher.get_ticker(st.session_state.symbol)
        
        if ticker:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${ticker['last']:.2f}",
                    delta=f"{ticker['percentage']:.2f}%" if 'percentage' in ticker else None
                )
            
            with col2:
                st.metric(
                    label="24h Volume",
                    value=f"${ticker['quoteVolume']:,.0f}" if 'quoteVolume' in ticker else "N/A"
                )
            
            with col3:
                st.metric(
                    label="24h High",
                    value=f"${ticker['high']:.2f}" if 'high' in ticker else "N/A"
                )
            
            with col4:
                st.metric(
                    label="24h Low",
                    value=f"${ticker['low']:.2f}" if 'low' in ticker else "N/A"
                )
        else:
            st.info("Market data not available. Please check your connection.")
    
    except Exception as e:
        st.error(f"Error fetching market summary: {e}")
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("""
    Crypto Trading Analyzer | Educational Tool | Not Financial Advice
    
    Data provided by Binance API. This application is not affiliated with Binance.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def render_market_data_page():
    """Render the market data page"""
    st.markdown('<h1 class="main-header">Market Data</h1>', unsafe_allow_html=True)
    
    # Fetch data if not already in session state or if auto-refresh is enabled
    current_time = time.time()
    should_refresh = (
        'data' not in st.session_state or
        (st.session_state.auto_refresh and 
         current_time - st.session_state.last_refresh > st.session_state.refresh_interval)
    )
    
    if should_refresh:
        with st.spinner("Fetching market data..."):
            df = fetch_data(
                st.session_state.symbol,
                st.session_state.timeframe,
                limit=st.session_state.limit
            )
            
            if df is not None and not df.empty:
                st.session_state.data = df
                st.session_state.last_refresh = current_time
            else:
                st.error("Failed to fetch market data. Please try again.")
                return
    
    # Display market information
    if 'data' in st.session_state:
        df = st.session_state.data
        
        # Market overview
        st.markdown('<h2 class="sub-header">Market Overview</h2>', unsafe_allow_html=True)
        
        # Get latest data point
        latest = df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = latest['close'] - latest['open']
            price_change_pct = (price_change / latest['open']) * 100
            
            st.metric(
                label="Current Price",
                value=f"${latest['close']:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col2:
            st.metric(
                label="24h Volume",
                value=f"${df['volume'].sum():,.0f}"
            )
        
        with col3:
            st.metric(
                label="24h High",
                value=f"${df['high'].max():.2f}"
            )
        
        with col4:
            st.metric(
                label="24h Low",
                value=f"${df['low'].min():.2f}"
            )
        
        # Price chart
        st.markdown('<h2 class="sub-header">Price Chart</h2>', unsafe_allow_html=True)
        
        # Chart options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_volume = st.checkbox("Show Volume", value=True)
        
        with col2:
            show_ma = st.checkbox("Show Moving Averages", value=True)
        
        with col3:
            candle_count = st.slider("Display Candles", 50, min(500, len(df)), 100)
        
        # Plot chart
        df_display = df.iloc[-candle_count:]
        
        fig = make_subplots(
            rows=2 if show_volume else 1, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.8, 0.2] if show_volume else [1]
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df_display.index,
                open=df_display['open'],
                high=df_display['high'],
                low=df_display['low'],
                close=df_display['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma:
            for period, color in [(9, 'blue'), (20, 'orange'), (50, 'red')]:
                ma_col = f'ema_{period}'
                if ma_col not in df_display.columns:
                    # Calculate EMA if not already in dataframe
                    df_display[ma_col] = df_display['close'].ewm(span=period, adjust=False).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=df_display.index,
                        y=df_display[ma_col],
                        name=f"EMA {period}",
                        line=dict(color=color, width=1)
                    ),
                    row=1, col=1
                )
        
        # Add volume if requested
        if show_volume:
            colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df_display.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=df_display.index,
                    y=df_display['volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{st.session_state.symbol} - {TIMEFRAMES[st.session_state.timeframe]} Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market statistics
        st.markdown('<h2 class="sub-header">Market Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price statistics
            st.markdown("#### Price Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Open', 'High', 'Low', 'Close',
                    'Average', 'Std Dev', 'Range',
                    'Change (%)', 'Volatility (%)'
                ],
                'Value': [
                    f"${latest['open']:.2f}",
                    f"${df['high'].max():.2f}",
                    f"${df['low'].min():.2f}",
                    f"${latest['close']:.2f}",
                    f"${df['close'].mean():.2f}",
                    f"${df['close'].std():.2f}",
                    f"${df['high'].max() - df['low'].min():.2f}",
                    f"{((latest['close'] / df['open'].iloc[0]) - 1) * 100:.2f}%",
                    f"{(df['close'].std() / df['close'].mean()) * 100:.2f}%"
                ]
            })
            
            st.dataframe(stats_df, hide_index=True)
        
        with col2:
            # Volume statistics
            st.markdown("#### Volume Statistics")
            
            vol_stats_df = pd.DataFrame({
                'Metric': [
                    'Total Volume', 'Average Volume', 
                    'Max Volume', 'Min Volume',
                    'Volume Std Dev', 'Current vs Avg (%)'
                ],
                'Value': [
                    f"${df['volume'].sum():,.0f}",
                    f"${df['volume'].mean():,.0f}",
                    f"${df['volume'].max():,.0f}",
                    f"${df['volume'].min():,.0f}",
                    f"${df['volume'].std():,.0f}",
                    f"{(latest['volume'] / df['volume'].mean() - 1) * 100:.2f}%"
                ]
            })
            
            st.dataframe(vol_stats_df, hide_index=True)
        
        # Data table
        st.markdown('<h2 class="sub-header">Raw Data</h2>', unsafe_allow_html=True)
        
        # Show the most recent data points
        display_df = df.iloc[-20:].copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M:%S')
        display_df = display_df.reset_index()
        display_df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = df.to_csv().encode('utf-8')
        
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{st.session_state.symbol.replace('/', '_')}_{st.session_state.timeframe}.csv",
            mime="text/csv",
        )
    
    else:
        st.info("No data available. Please fetch data first.")

def render_technical_analysis_page():
    """Render the technical analysis page"""
    st.markdown('<h1 class="main-header">Technical Analysis</h1>', unsafe_allow_html=True)
    
    # Fetch data if not already in session state or if auto-refresh is enabled
    current_time = time.time()
    should_refresh = (
        'data' not in st.session_state or
        (st.session_state.auto_refresh and 
         current_time - st.session_state.last_refresh > st.session_state.refresh_interval)
    )
    
    if should_refresh:
        with st.spinner("Fetching market data..."):
            df = fetch_data(
                st.session_state.symbol,
                st.session_state.timeframe,
                limit=st.session_state.limit
            )
            
            if df is not None and not df.empty:
                st.session_state.data = df
                st.session_state.last_refresh = current_time
            else:
                st.error("Failed to fetch market data. Please try again.")
                return
    
    # Indicator selection
    st.markdown('<h2 class="sub-header">Select Indicators</h2>', unsafe_allow_html=True)
    
    # Create columns for indicator selection
    col1, col2, col3 = st.columns(3)
    
    selected_indicators = []
    
    with col1:
        if st.checkbox("RSI", value="rsi" in st.session_state.indicators):
            selected_indicators.append("rsi")
        
        if st.checkbox("MACD", value="macd" in st.session_state.indicators):
            selected_indicators.append("macd")
        
        if st.checkbox("Bollinger Bands", value="bollinger_bands" in st.session_state.indicators):
            selected_indicators.append("bollinger_bands")
        
        if st.checkbox("EMA Crossover", value="ema_crossover" in st.session_state.indicators):
            selected_indicators.append("ema_crossover")
    
    with col2:
        if st.checkbox("Stochastic", value="stochastic" in st.session_state.indicators):
            selected_indicators.append("stochastic")
        
        if st.checkbox("ADX", value="adx" in st.session_state.indicators):
            selected_indicators.append("adx")
        
        if st.checkbox("ATR", value="atr" in st.session_state.indicators):
            selected_indicators.append("atr")
        
        if st.checkbox("Parabolic SAR", value="parabolic_sar" in st.session_state.indicators):
            selected_indicators.append("parabolic_sar")
    
    with col3:
        if st.checkbox("Volume Profile", value="volume_profile" in st.session_state.indicators):
            selected_indicators.append("volume_profile")
        
        if st.checkbox("Support/Resistance", value="support_resistance" in st.session_state.indicators):
            selected_indicators.append("support_resistance")
        
        if st.checkbox("Fibonacci", value="fibonacci" in st.session_state.indicators):
            selected_indicators.append("fibonacci")
        
        if st.checkbox("Ichimoku Cloud", value="ichimoku" in st.session_state.indicators):
            selected_indicators.append("ichimoku")
    
    # Update session state
    st.session_state.indicators = selected_indicators
    
    # Apply indicators button
    if st.button("Apply Indicators"):
        if 'data' in st.session_state:
            with st.spinner("Applying technical indicators..."):
                df_with_indicators = apply_indicators(st.session_state.data, selected_indicators)
                
                if df_with_indicators is not None:
                    st.session_state.data_with_indicators = df_with_indicators
                    
                    # Generate signals
                    strategy = get_strategy_by_name(st.session_state.strategy)
                    df_with_signals = strategy.generate_signals(df_with_indicators)
                    st.session_state.data_with_signals = df_with_signals
                else:
                    st.error("Failed to apply indicators. Please try again.")
        else:
            st.error("No data available. Please fetch data first.")
    
    # Display technical analysis
    if 'data_with_indicators' in st.session_state:
        df = st.session_state.data_with_indicators
        
        # Price chart with indicators
        st.markdown('<h2 class="sub-header">Price Chart with Indicators</h2>', unsafe_allow_html=True)
        
        # Chart options
        col1, col2 = st.columns(2)
        
        with col1:
            candle_count = st.slider(
                "Display Candles", 
                50, 
                min(500, len(df)), 
                100,
                key="ta_candle_count"
            )
        
        with col2:
            show_signals = st.checkbox("Show Signals", value=True, key="ta_show_signals")
        
        # Plot chart
        if 'data_with_signals' in st.session_state and show_signals:
            chart_df = st.session_state.data_with_signals.iloc[-candle_count:]
        else:
            chart_df = df.iloc[-candle_count:]
        
        fig = plot_candlestick_chart(chart_df, selected_indicators)
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual indicator charts
        st.markdown('<h2 class="sub-header">Indicator Analysis</h2>', unsafe_allow_html=True)
        
        # Create tabs for each selected indicator
        if selected_indicators:
            tabs = st.tabs([INDICATORS[ind] for ind in selected_indicators])
            
            for i, indicator in enumerate(selected_indicators):
                with tabs[i]:
                    # Display indicator chart
                    fig = plot_indicator_chart(chart_df, indicator)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display indicator explanation
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    
                    if indicator == 'rsi':
                        st.markdown("""
                        ### Relative Strength Index (RSI)
                        
                        RSI measures the speed and change of price movements, oscillating between 0 and 100.
                        
                        - **Overbought**: RSI > 70 (potential sell signal)
                        - **Oversold**: RSI < 30 (potential buy signal)
                        - **Bullish Divergence**: Price makes lower lows while RSI makes higher lows
                        - **Bearish Divergence**: Price makes higher highs while RSI makes lower highs
                        
                        RSI is most effective in ranging markets and should be used with other indicators for confirmation.
                        """)
                    
                    elif indicator == 'macd':
                        st.markdown("""
                        ### Moving Average Convergence Divergence (MACD)
                        
                        MACD shows the relationship between two moving averages of a security's price.
                        
                        - **MACD Line**: Difference between 12-period and 26-period EMA
                        - **Signal Line**: 9-period EMA of the MACD Line
                        - **Histogram**: Difference between MACD Line and Signal Line
                        
                        **Signals**:
                        - Bullish: MACD crosses above Signal Line
                        - Bearish: MACD crosses below Signal Line
                        - Bullish Divergence: Price makes lower lows while MACD makes higher lows
                        - Bearish Divergence: Price makes higher highs while MACD makes lower highs
                        """)
                    
                    elif indicator == 'bollinger_bands':
                        st.markdown("""
                        ### Bollinger Bands
                        
                        Bollinger Bands consist of a middle band (20-period SMA) and two outer bands placed 2 standard deviations away.
                        
                        - **Upper Band**: Middle Band + (2 × Standard Deviation)
                        - **Middle Band**: 20-period SMA
                        - **Lower Band**: Middle Band - (2 × Standard Deviation)
                        
                        **Signals**:
                        - Price touching upper band indicates potential overbought condition
                        - Price touching lower band indicates potential oversold condition
                        - Bands narrowing indicates low volatility (potential breakout setup)
                        - Bands widening indicates increasing volatility
                        """)
                    
                    # Add explanations for other indicators...
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Signal analysis
        if 'data_with_signals' in st.session_state:
            st.markdown('<h2 class="sub-header">Signal Analysis</h2>', unsafe_allow_html=True)
            
            signal_df = st.session_state.data_with_signals
            
            # Get the latest signals
            latest = signal_df.iloc[-1]
            
            # Create columns for signal display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Combined Signal")
                
                if 'combined_signal' in latest:
                    signal_value = latest['combined_signal']
                    signal_text = get_signal_emoji(signal_value)
                    signal_color = get_signal_color(signal_value)
                    
                    st.markdown(
                        f'<div style="background-color: {signal_color}20; padding: 1rem; '
                        f'border-radius: 0.5rem; border-left: 0.5rem solid {signal_color}; text-align: center;">'
                        f'<h2 style="color: {signal_color};">{signal_text}</h2>'
                        f'<p>Signal Strength: {abs(signal_value):.2f}</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No combined signal available.")
            
            with col2:
                st.markdown("#### Individual Indicators")
                
                # Show individual indicator signals
                signal_cols = [col for col in signal_df.columns if col.endswith('_signal') and col != 'combined_signal']
                
                if signal_cols:
                    for col in signal_cols[:5]:  # Show top 5 indicators
                        if col in latest:
                            indicator_name = col.replace('_signal', '').replace('_', ' ').title()
                            signal_value = latest[col]
                            signal_text = get_signal_emoji(signal_value)
                            signal_color = get_signal_color(signal_value)
                            
                            st.markdown(
                                f'<div style="margin-bottom: 0.5rem; padding: 0.5rem; '
                                f'border-left: 0.3rem solid {signal_color};">'
                                f'<b>{indicator_name}:</b> <span style="color: {signal_color};">{signal_text}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No individual signals available.")
            
            with col3:
                st.markdown("#### Market Trend")
                
                if 'trend' in latest:
                    trend = latest['trend']
                    
                    if trend == 'uptrend':
                        st.markdown(
                            '<div style="background-color: #E8F5E920; padding: 1rem; '
                            'border-radius: 0.5rem; border-left: 0.5rem solid #4CAF50; text-align: center;">'
                            '<h2 style="color: #4CAF50;">📈 Uptrend</h2>'
                            '<p>Price is above key moving averages</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div style="background-color: #FFEBEE20; padding: 1rem; '
                            'border-radius: 0.5rem; border-left: 0.5rem solid #F44336; text-align: center;">'
                            '<h2 style="color: #F44336;">📉 Downtrend</h2>'
                            '<p>Price is below key moving averages</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                else:
                    # Calculate simple trend based on EMA
                    if 'ema_50' in latest:
                        trend = 'uptrend' if latest['close'] > latest['ema_50'] else 'downtrend'
                        
                        if trend == 'uptrend':
                            st.markdown(
                                '<div style="background-color: #E8F5E920; padding: 1rem; '
                                'border-radius: 0.5rem; border-left: 0.5rem solid #4CAF50; text-align: center;">'
                                '<h2 style="color: #4CAF50;">📈 Uptrend</h2>'
                                '<p>Price is above 50 EMA</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div style="background-color: #FFEBEE20; padding: 1rem; '
                                'border-radius: 0.5rem; border-left: 0.5rem solid #F44336; text-align: center;">'
                                '<h2 style="color: #F44336;">📉 Downtrend</h2>'
                                '<p>Price is below 50 EMA</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Trend information not available.")
            
            # Signal history
            st.markdown("#### Recent Signal History")
            
            # Get recent signals where there was a change
            signal_changes = signal_df[signal_df['combined_signal'] != signal_df['combined_signal'].shift(1)].iloc[-10:]
            
            if not signal_changes.empty:
                # Create a DataFrame for display
                display_df = pd.DataFrame({
                    'Date': signal_changes.index.strftime('%Y-%m-%d %H:%M'),
                    'Price': signal_changes['close'].round(2),
                    'Signal': signal_changes['combined_signal'].apply(lambda x: get_signal_emoji(x)),
                    'Strength': signal_changes['signal_strength'].round(2) if 'signal_strength' in signal_changes.columns else None
                })
                
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("No recent signal changes.")
    
    else:
        st.info("No indicator data available. Please apply indicators first.")

def render_strategy_config_page():
    """Render the strategy configuration page"""
    st.markdown('<h1 class="main-header">Strategy Configuration</h1>', unsafe_allow_html=True)
    
    # Strategy selection
    st.markdown('<h2 class="sub-header">Select Strategy</h2>', unsafe_allow_html=True)
    
    strategy_type = st.selectbox(
        "Strategy Type",
        options=list(STRATEGIES.keys()),
        format_func=lambda x: STRATEGIES[x],
        index=list(STRATEGIES.keys()).index(st.session_state.strategy) if st.session_state.strategy in STRATEGIES else 0
    )
    
    # Update session state
    if strategy_type != st.session_state.strategy:
        st.session_state.strategy = strategy_type
        
        # Clear strategy-specific session state
        for key in list(st.session_state.keys()):
            if key.startswith("strategy_config_"):
                del st.session_state[key]
    
    # Strategy description
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    if strategy_type == "multi_indicator":
        st.markdown("""
        ### Multi-Indicator Strategy
        
        This strategy combines multiple technical indicators with confirmation layers to generate trading signals.
        It uses a weighted approach to combine signals from different indicators, with trend confirmation
        and market sentiment analysis.
        
        **Best for**: Traders who want a balanced approach using multiple indicators for confirmation.
        """)
    
    elif strategy_type == "trend_following":
        st.markdown("""
        ### Trend Following Strategy
        
        This strategy focuses on identifying and following established market trends. It uses trend indicators
        like moving averages, ADX, and Parabolic SAR to identify trending markets and enter positions in
        the direction of the trend.
        
        **Best for**: Traders who prefer to trade with the trend and hold positions for longer periods.
        """)
    
    elif strategy_type == "mean_reversion":
        st.markdown("""
        ### Mean Reversion Strategy
        
        This strategy focuses on identifying overbought and oversold conditions and trading counter to the trend.
        It uses oscillators like RSI, Stochastic, and Bollinger Bands to identify potential reversal points.
        
        **Best for**: Traders who look for price extremes and believe that prices will revert to the mean.
        """)
    
    elif strategy_type == "breakout":
        st.markdown("""
        ### Breakout Strategy
        
        This strategy focuses on identifying and trading price breakouts from consolidation patterns,
        support/resistance levels, or chart patterns. It uses volatility indicators, volume analysis,
        and support/resistance to identify potential breakout opportunities.
        
        **Best for**: Traders who want to capture strong price movements after periods of consolidation.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy parameters
    st.markdown('<h2 class="sub-header">Strategy Parameters</h2>', unsafe_allow_html=True)
    
    # Create a temporary strategy instance to get default parameters
    temp_strategy = get_strategy_by_name(strategy_type)
    default_params = temp_strategy.parameters
    
    # Create dictionary to store updated parameters
    if f"strategy_config_{strategy_type}" not in st.session_state:
        st.session_state[f"strategy_config_{strategy_type}"] = default_params.copy()
    
    strategy_params = st.session_state[f"strategy_config_{strategy_type}"]
    
    # Create tabs for different parameter categories
    tab1, tab2, tab3, tab4 = st.tabs(["General Settings", "Indicator Weights", "Entry/Exit Rules", "Risk Management"])
    
    with tab1:
        # General settings
        col1, col2 = st.columns(2)
        
        with col1:
            signal_threshold = st.slider(
                "Signal Threshold",
                min_value=0.1,
                max_value=1.0,
                value=strategy_params.get('signal_threshold', 0.7),
                step=0.05,
                help="Minimum combined signal strength (0-1) to trigger a trade"
            )
            strategy_params['signal_threshold'] = signal_threshold
            
            confirmation_needed = st.slider(
                "Confirmation Needed",
                min_value=1,
                max_value=5,
                value=strategy_params.get('confirmation_needed', 2),
                step=1,
                help="Number of indicators that must confirm a signal"
            )
            strategy_params['confirmation_needed'] = confirmation_needed
        
        with col2:
            trend_confirmation = st.checkbox(
                "Require Trend Confirmation",
                value=strategy_params.get('trend_confirmation', True),
                help="Only generate signals in the direction of the overall trend"
            )
            strategy_params['trend_confirmation'] = trend_confirmation
            
            use_market_sentiment = st.checkbox(
                "Use Market Sentiment",
                value=strategy_params.get('use_market_sentiment', True),
                help="Consider overall market sentiment in signal generation"
            )
            strategy_params['use_market_sentiment'] = use_market_sentiment
    
    with tab2:
        # Indicator weights
        st.markdown("Adjust the weight of each indicator in the strategy:")
        
        # Get indicator settings from parameters
        indicator_settings = strategy_params.get('indicators', {})
        
        # Create columns for indicator weights
        col1, col2, col3 = st.columns(3)
        
        columns = [col1, col2, col3]
        i = 0
        
        for indicator, info in INDICATORS.items():
            # Get default values
            enabled = indicator_settings.get(indicator, {}).get('enabled', False)
            weight = indicator_settings.get(indicator, {}).get('weight', 1.0)
            
            # Add to the appropriate column
            with columns[i % 3]:
                st.markdown(f"##### {info}")
                
                enabled_key = f"enabled_{indicator}"
                weight_key = f"weight_{indicator}"
                
                enabled = st.checkbox(
                    "Enabled",
                    value=enabled,
                    key=enabled_key
                )
                
                weight = st.slider(
                    "Weight",
                    min_value=0.1,
                    max_value=1.0,
                    value=weight,
                    step=0.1,
                    key=weight_key,
                    disabled=not enabled
                )
                
                # Update indicator settings
                if indicator not in indicator_settings:
                    indicator_settings[indicator] = {}
                
                indicator_settings[indicator]['enabled'] = enabled
                indicator_settings[indicator]['weight'] = weight
                
                st.markdown("---")
            
            i += 1
        
        # Update indicator settings in parameters
        strategy_params['indicators'] = indicator_settings
    
    with tab3:
        # Entry/Exit rules
        col1, col2 = st.columns(2)
        
        with col1:
            exit_on_opposite = st.checkbox(
                "Exit on Opposite Signal",
                value=strategy_params.get('exit_on_opposite_signal', True),
                help="Exit position when opposite signal appears"
            )
            strategy_params['exit_on_opposite_signal'] = exit_on_opposite
            
            max_holding_time = st.slider(
                "Max Holding Time (hours)",
                min_value=1,
                max_value=168,  # 1 week
                value=strategy_params.get('max_holding_time_hours', 72),
                step=1,
                help="Maximum time to hold a position before automatic exit"
            )
            strategy_params['max_holding_time_hours'] = max_holding_time
        
        with col2:
            # Strategy-specific parameters
            if strategy_type == "trend_following":
                min_adx = st.slider(
                    "Minimum ADX Value",
                    min_value=10,
                    max_value=50,
                    value=strategy_params.get('min_adx_value', 25),
                    step=1,
                    help="Minimum ADX value to confirm trend strength"
                )
                strategy_params['min_adx_value'] = min_adx
                
                trend_ema = st.slider(
                    "Trend EMA Period",
                    min_value=20,
                    max_value=200,
                    value=strategy_params.get('trend_ema_period', 50),
                    step=5,
                    help="EMA period for trend determination"
                )
                strategy_params['trend_ema_period'] = trend_ema
            
            elif strategy_type == "mean_reversion":
                rsi_oversold = st.slider(
                    "RSI Oversold Level",
                    min_value=10,
                    max_value=40,
                    value=strategy_params.get('rsi_oversold', 30),
                    step=1,
                    help="RSI level to consider market oversold"
                )
                strategy_params['rsi_oversold'] = rsi_oversold
                
                rsi_overbought = st.slider(
                    "RSI Overbought Level",
                    min_value=60,
                    max_value=90,
                    value=strategy_params.get('rsi_overbought', 70),
                    step=1,
                    help="RSI level to consider market overbought"
                )
                strategy_params['rsi_overbought'] = rsi_overbought
            
            elif strategy_type == "breakout":
                vol_increase = st.slider(
                    "Min Volume Increase",
                    min_value=1.0,
                    max_value=3.0,
                    value=strategy_params.get('min_volume_increase', 1.5),
                    step=0.1,
                    help="Minimum volume increase for breakout confirmation"
                )
                strategy_params['min_volume_increase'] = vol_increase
                
                volatility_threshold = st.slider(
                    "Volatility Expansion Threshold",
                    min_value=1.0,
                    max_value=2.0,
                    value=strategy_params.get('volatility_expansion_threshold', 1.3),
                    step=0.1,
                    help="Volatility expansion threshold for breakout detection"
                )
                strategy_params['volatility_expansion_threshold'] = volatility_threshold
    
    with tab4:
        # Risk management
        col1, col2 = st.columns(2)
        
        with col1:
            use_trailing_stop = st.checkbox(
                "Use Trailing Stop",
                value=strategy_params.get('use_trailing_stop', True),
                help="Use trailing stops for exit"
            )
            strategy_params['use_trailing_stop'] = use_trailing_stop
            
            dynamic_sizing = st.checkbox(
                "Enable Dynamic Position Sizing",
                value=strategy_params.get('enable_dynamic_sizing', True),
                help="Adjust position size based on signal strength and risk"
            )
            strategy_params['enable_dynamic_sizing'] = dynamic_sizing
        
        with col2:
            risk_level = st.selectbox(
                "Risk Level",
                options=list(RISK_LEVELS.keys()),
                format_func=lambda x: RISK_LEVELS[x],
                index=list(RISK_LEVELS.keys()).index(st.session_state.risk_level) if st.session_state.risk_level in RISK_LEVELS else 1
            )
            st.session_state.risk_level = risk_level
            
            # Show risk parameters based on selected level
            risk_params = config.RISK_LEVELS.get(risk_level, {})
            
            st.markdown(f"""
            **Risk Parameters:**
            - Max Position Size: {risk_params.get('max_position_size_pct', 0)}%
            - Stop Loss: {risk_params.get('stop_loss_pct', 0)}%
            - Take Profit: {risk_params.get('take_profit_pct', 0)}%
            - Max Daily Drawdown: {risk_params.get('max_daily_drawdown_pct', 0)}%
            """)
    
    # Update strategy parameters in session state
    st.session_state[f"strategy_config_{strategy_type}"] = strategy_params
    
    # Test strategy button
    st.markdown('<h2 class="sub-header">Test Strategy</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Apply Strategy to Current Data"):
            if 'data' in st.session_state:
                with st.spinner("Applying strategy..."):
                    # Create strategy instance with configured parameters
                    strategy = get_strategy_by_name(strategy_type, strategy_params)
                    
                    # Apply indicators and generate signals
                    df = apply_indicators(st.session_state.data)
                    
                    if df is not None:
                        df_with_signals = strategy.generate_signals(df)
                        st.session_state.data_with_signals = df_with_signals
                        st.session_state.current_strategy = strategy
                        
                        # Show success message
                        st.success("Strategy applied successfully!")
                    else:
                        st.error("Failed to apply indicators.")
            else:
                st.error("No data available. Please fetch data first.")
    
    with col2:
        if st.button("Go to Backtesting"):
            st.session_state.page = "backtesting"
    
    # Display strategy results if available
    if 'data_with_signals' in st.session_state and 'current_strategy' in st.session_state:
        st.markdown('<h2 class="sub-header">Strategy Results</h2>', unsafe_allow_html=True)
        
        df = st.session_state.data_with_signals
        
        # Show chart with signals
        fig = plot_candlestick_chart(df.iloc[-100:], st.session_state.indicators)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show signal statistics
        signal_stats = st.session_state.current_strategy.get_signal_statistics(df)
        
        if signal_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Signal Distribution")
                
                # Create pie chart of signal types
                signal_counts = signal_stats.get('signal_counts', {})
                
                if signal_counts:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(signal_counts.keys()),
                        values=list(signal_counts.values()),
                        hole=.3
                    )])
                    
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No signal data available.")
            
            with col2:
                st.markdown("#### Signal Metrics")
                
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Total Signals',
                        'Average Duration',
                        'Buy Signals (%)',
                        'Sell Signals (%)',
                        'Neutral Signals (%)'
                    ],
                    'Value': [
                        signal_stats.get('total_signals', 0),
                        f"{signal_stats.get('average_duration', 0):.2f} periods",
                        f"{signal_stats.get('signal_ratio', {}).get('buy', 0) * 100:.2f}%",
                        f"{signal_stats.get('signal_ratio', {}).get('sell', 0) * 100:.2f}%",
                        f"{signal_stats.get('signal_ratio', {}).get('neutral', 0) * 100:.2f}%"
                    ]
                })
                
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        # Show recent signals
        st.markdown("#### Recent Signals")
        
        # Get signals where there was a change
        signal_changes = df[df['strategy_signal'] != df['strategy_signal'].shift(1)].iloc[-10:]
        
        if not signal_changes.empty:
            # Create a DataFrame for display
            display_df = pd.DataFrame({
                'Date': signal_changes.index.strftime('%Y-%m-%d %H:%M'),
                'Price': signal_changes['close'].round(2),
                'Signal': signal_changes['strategy_signal'].apply(lambda x: get_signal_emoji(x)),
                'Strength': signal_changes['signal_strength'].round(2) if 'signal_strength' in signal_changes.columns else None,
                'Confirming Indicators': signal_changes['buy_indicators'] if 'buy_indicators' in signal_changes.columns else None
            })
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No recent signal changes.")
    
    # Save/load strategy configuration
    st.markdown('<h2 class="sub-header">Save/Load Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save configuration
        if st.button("Save Configuration"):
            # Convert parameters to JSON
            params_json = json.dumps(strategy_params, indent=2)
            
            # Create download link
            st.markdown(
                download_link(
                    params_json,
                    f"{strategy_type}_config.json",
                    "Download Configuration"
                ),
                unsafe_allow_html=True
            )
    
    with col2:
        # Load configuration
        uploaded_file = st.file_uploader("Load Configuration", type="json")
        
        if uploaded_file is not None:
            try:
                loaded_params = json.load(uploaded_file)
                st.session_state[f"strategy_config_{strategy_type}"] = loaded_params
                st.success("Configuration loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading configuration: {e}")

def render_backtesting_page():
    """Render the backtesting page"""
    st.markdown('<h1 class="main-header">Strategy Backtesting</h1>', unsafe_allow_html=True)
    
    # Backtesting parameters
    st.markdown('<h2 class="sub-header">Backtesting Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Strategy selection
        strategy_type = st.selectbox(
            "Strategy Type",
            options=list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x],
            index=list(STRATEGIES.keys()).index(st.session_state.strategy) if st.session_state.strategy in STRATEGIES else 0,
            key="backtest_strategy"
        )
        
        # Use saved configuration if available
        if f"strategy_config_{strategy_type}" in st.session_state:
            use_saved_config = st.checkbox(
                "Use Saved Configuration",
                value=True,
                help="Use the configuration from the Strategy Configuration page"
            )
        else:
            use_saved_config = False
        
        # Symbol selection
        symbol = st.selectbox(
            "Trading Pair",
            options=config.TRADING_PAIRS,
            index=config.TRADING_PAIRS.index(st.session_state.symbol) if st.session_state.symbol in config.TRADING_PAIRS else 0,
            key="backtest_symbol"
        )
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAMES.keys()),
            format_func=lambda x: TIMEFRAMES[x],
            index=list(TIMEFRAMES.keys()).index(st.session_state.timeframe) if st.session_state.timeframe in TIMEFRAMES else 0,
            key="backtest_timeframe"
        )
    
    with col2:
        # Date range selection inputs
        start_date = st.date_input(
            "Start Date",
            value=datetime.date.today() - datetime.timedelta(days=90),
            key="backtest_start_date",
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.date.today(),
            key="backtest_end_date",
        )

        # Capital / fees
        initial_capital = st.number_input(
            "Initial Capital (USDT)",
            min_value=1000.0,
            max_value=1_000_000.0,
            value=st.session_state.initial_capital,
            step=100.0,
            key="backtest_capital",
        )
        commission_pct = st.number_input(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.commission_pct,
            step=0.01,
            key="backtest_commission",
        )

    st.markdown("---")

    # Run backtest button
    if st.button("Run Backtest"):
        with st.spinner("Running backtest … this may take a while"):

            # Choose parameters
            params = (
                st.session_state.get(f"strategy_config_{strategy_type}", {})
                if use_saved_config
                else None
            )

            results = run_backtest(
                strategy_name=strategy_type,
                symbols=[symbol],
                timeframe=timeframe,
                start_date=start_date.isoformat(),
                end_date=(end_date + datetime.timedelta(days=1)).isoformat(),
                initial_capital=initial_capital,
                risk_level=st.session_state.risk_level,
                commission_pct=commission_pct,
                slippage_pct=st.session_state.slippage_pct,
                parameters=params,
                generate_report=False,
            )

            if results:
                st.session_state.backtest_results = results
                st.success("Backtest completed!")
            else:
                st.error("Backtest failed ‑ see logs for details.")

    # Show brief results if available
    if "backtest_results" in st.session_state:
        r: BacktestResults = st.session_state.backtest_results

        st.markdown("### Summary")
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Initial Capital",
                    "Final Capital",
                    "Total Return (%)",
                    "Total Trades",
                    "Win-rate (%)",
                    "Max Drawdown (%)",
                ],
                "Value": [
                    f"${r.initial_capital:,.2f}",
                    f"${r.final_capital:,.2f}",
                    f"{((r.final_capital / r.initial_capital) - 1)*100:,.2f}",
                    len(r.trades),
                    f"{r.performance_metrics.get('win_rate',0)*100:,.2f}",
                    f"{r.performance_metrics.get('max_drawdown',0):,.2f}",
                ],
            }
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # Simple equity curve
        if r.equity_curve:
            ec = pd.Series(r.equity_curve)
            fig = px.line(ec, title="Equity Curve")
            st.plotly_chart(fig, use_container_width=True)

######################################################################
#  Additional simple pages (stubs) so navigation does not break
######################################################################

def render_performance_page():
    st.markdown('<h1 class="main-header">Performance Analytics</h1>', unsafe_allow_html=True)
    if "backtest_results" not in st.session_state:
        st.info("Run a backtest first to see analytics.")
        return
    r: BacktestResults = st.session_state.backtest_results
    st.json(r.performance_metrics)


def render_education_page():
    st.markdown('<h1 class="main-header">Education</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        This section will include educational material on:
        - Technical indicators  
        - Risk management fundamentals  
        - Example strategy breakdowns  
        *Coming soon…*
        """
    )


def render_settings_page():
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    dark = st.checkbox("Dark Mode (beta)", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark


######################################################################
#  Main page dispatcher
######################################################################

def run_app():
    page = st.session_state.page

    if page == "home":
        render_home_page()
    elif page == "market_data":
        render_market_data_page()
    elif page == "technical_analysis":
        render_technical_analysis_page()
    elif page == "strategy_config":
        render_strategy_config_page()
    elif page == "backtesting":
        render_backtesting_page()
    elif page == "performance":
        render_performance_page()
    elif page == "education":
        render_education_page()
    elif page == "settings":
        render_settings_page()
    else:
        st.error("Unknown page.")


######################################################################
#  App bootstrap (run on import)
######################################################################

initialize_session_state()
load_css()
render_sidebar()
run_app()
