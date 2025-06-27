# Crypto Trading Analyzer

A modern, educational Streamlit application for **crypto-currency market analysis, strategy design and historical back-testing**.  
The project shows how to combine real-time Binance data, technical analysis, risk management and interactive dashboards in one coherent Python code base.

---

## ✨ Key Features

| Category | Highlights |
|----------|------------|
| Market Data | • Live OHLCV candles, depth, trades<br>• Multi-timeframe support (1 m – 1 w)<br>• Built-in caching & rate-limiting |
| Technical Analysis | • 15+ indicators (RSI, MACD, Bollinger Bands, EMA/ SMA, ADX, ATR, Ichimoku …)<br>• Auto-combination & signal weighting<br>• Beautiful Plotly + Streamlit visualisations |
| Strategy Engine | • Multi-Indicator, Trend-Following, Mean-Reversion & Breakout templates<br>• Weighted confirmation layers & dynamic position sizing<br>• Trailing-stop, stop-loss / take-profit logic |
| Back-Testing | • Portfolio simulation with commission & slippage<br>• Equity curve, drawdown, Sharpe, profit-factor, win-rate, streaks & more<br>• Walk-forward, Monte-Carlo & parameter optimisation helpers |
| GUI | • Clean sidebar navigation<br>• Interactive pages: Home, Market, TA, Strategy Builder, Backtesting, Analytics, Education, Settings |
| Developer Friendly | • Modular code (`data_fetcher.py`, `technical_indicators.py`, `strategy.py`, `backtester.py`, `app.py`)<br>• Docstrings & type-hints<br>• MIT licence |

---

## 🗂️ Folder Structure (important files)

```
├── app.py                    # Streamlit UI
├── run.py                    # Convenience launcher
├── config.py                 # Global settings & defaults
├── data_fetcher.py           # Binance integration & caching
├── technical_indicators.py   # All TA calculations
├── strategy.py               # Strategy framework & templates
├── backtester.py             # Portfolio level engine & analytics
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

1. **Clone**
   ```bash
   git clone https://github.com/your-user/crypto-trading-analyzer.git
   cd crypto-trading-analyzer
   ```

2. **Python 3.9+ & virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *TA-Lib* wheels for some platforms are pre-built, otherwise follow the [TA-Lib install guide](https://mrjbq7.github.io/ta-lib/install.html).

4. **Optional – set API keys**

   The default runs on Binance **testnet** without keys, but for private endpoints
   create a `.env` file or export environment variables:

   | Variable        | Description                         |
   |-----------------|-------------------------------------|
   | `API_KEY`       | Binance API key                     |
   | `API_SECRET`    | Binance API secret                  |
   | `USE_TESTNET`   | `True`/`False` (default `True`)     |

---

## 🚀 Quick Start

```bash
# Easiest:
python run.py              # launches Streamlit automatically

# Or manually
streamlit run app.py
```

Open the provided local URL in your browser (default http://localhost:8501).

---

## 🕹️ Usage Guide

### 1. Home
High-level overview, quick-start links and **critical disclaimer**.

### 2. Market Data
Real-time candlestick chart, volume, statistics. Choose pair & timeframe in the sidebar, set auto-refresh.

### 3. Technical Analysis
Select indicators → *Apply Indicators* → interactive price & indicator charts, signal dashboard.

### 4. Strategy Configuration
Pick a template, tweak weights / thresholds / risk.  
Save or load JSON configs. *Apply Strategy* to see signals on current data.

### 5. Backtesting
Choose symbol, timeframe and date range → *Run Backtest*.  
Results: summary table, equity curve.  
Full reports (trades CSV, plots) are stored in `results/` when `generate_report=True`.

### 6. Performance Analytics
Detailed metrics, drawdown plots, monthly heatmap, trade distribution.

### 7. Education
Theory on TA indicators, risk management, example strategies (work-in-progress).

### 8. Settings
Experimental dark mode and global preferences.

---

## ⚙️ Extending

* Add new indicator: implement function in `technical_indicators.py`, return df, then call it from `apply_all_indicators`.
* Create custom strategy: subclass `Strategy` (see `MultiIndicatorStrategy`), override `generate_signals`.
* Data sources: `data_fetcher.py` uses **ccxt** – switch exchange by changing `EXCHANGE` in `config.py`.

---

## 🛡️ Disclaimer & Risk Warning

> **This project is for educational and research purposes only.  
> It does *not* provide financial advice.  
> Trading crypto-assets is highly speculative and can lead to significant losses, including total capital loss.  
> No strategy in this repository guarantees profits or a 90 % win-rate.  
> Use at your own risk. The authors accept no liability for any damages arising from the use of this software.**

Always:

* Back-test thoroughly.
* Start on paper / testnet.
* Never risk money you cannot afford to lose.

---

## 🤝 Contributing

Issues and pull requests are welcome! Please open an issue to discuss major changes first.

---

## 📄 Licence

This project is released under the **MIT License** – see [LICENSE](LICENSE) for details.
