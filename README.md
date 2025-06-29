# 🧠 Neural Pulse Scalper

A sophisticated, high-frequency cryptocurrency scalping bot for Binance Spot Trading. It leverages a multi-indicator strategy combined with an optional AI/ML decision layer to execute trades on 1-minute and multi-minute timeframes. The project includes a real-time Streamlit GUI for monitoring and control.

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-development-orange.svg)

---

## ⚠️ Disclaimer & Risk Warning

**This project is for educational and research purposes only. Cryptocurrency trading is extremely risky and can result in significant financial loss.**

- **DO NOT** run this bot with real money unless you understand the code and the risks involved.
- The creators of this bot are not liable for any financial losses you may incur.
- Past performance is not indicative of future results.
- Always start with paper trading on the **Testnet** to validate your strategy.

---

## ✨ Features

- **Multi-Indicator Strategy**: Combines EMA, RSI, Bollinger Bands, VWAP, and Volume for robust signal confirmation.
- **AI/ML Decision Layer**: Optional ML model (Decision Tree, RandomForest, XGBoost, etc.) to filter trades and improve win probability.
- **Advanced Risk Management**: Features daily drawdown limits, max trades per day, position sizing, and automatic pauses after consecutive losses.
- **Real-Time Streamlit GUI**: A comprehensive dashboard to monitor performance, view live charts, manage trades, and edit configurations on the fly.
- **Modular Architecture**: Cleanly separated components for the Binance client, strategy engine, indicators, and AI models.
- **Testnet & Mainnet Support**: Easily switch between paper trading (testnet) and live trading (mainnet).
- **Comprehensive Configuration**: Fine-tune every aspect of the strategy via a detailed YAML config file.

---

## 🛠️ Installation

Follow these steps to set up the project on your local machine.

```bash
# 1. Clone the repository
git clone https://github.com/mosabry99/BinanceSpotTrader.git
cd BinanceSpotTrader

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install the required dependencies
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Configuration is managed through two main files:

1.  `.env`: For your secret API keys.
2.  `config/config.yaml`: For all strategy, risk, and application parameters.

### 1. API Key Setup

You need to provide your Binance API keys to allow the bot to connect to your account.

1.  **Create a `.env` file** by copying the example file:
    ```bash
    cp .env.example .env
    ```
2.  **Log in to your Binance account.**
3.  Go to **API Management** in your account settings.
4.  Create a new API key.
5.  **Permissions**:
    - ✅ Enable **Reading**
    - ✅ Enable **Spot & Margin Trading**
    - ❌ **DO NOT** enable **Withdrawals**. This is a critical security measure.
6.  Copy your **API Key** and **Secret Key**.
7.  Open the `.env` file and paste your keys into the appropriate fields (e.g., `BINANCE_TESTNET_API_KEY` for paper trading or `BINANCE_MAINNET_API_KEY` for live trading).

### 2. Strategy Configuration

Open `config/config.yaml` to adjust the strategy parameters. This file is extensively commented and allows you to configure:
- Trading pairs and timeframes.
- Pre-filter conditions (volume, volatility).
- Entry and exit conditions (indicator settings, take profit, stop loss).
- Risk management rules (position size, max drawdown).
- AI/ML model settings.
- GUI settings.

---

## 🚀 Usage

The application can be run in three different modes: **Trade**, **Train**, and **GUI**.

### 1. Live/Paper Trading Mode

This mode runs the strategy engine to execute trades based on your configuration.

```bash
python -m src.main trade
```

- To switch between paper trading and live trading, set the `TRADING_MODE` variable in your `.env` file to `testnet` or `mainnet`.

### 2. AI Model Training Mode

This mode uses historical data to train the AI/ML model specified in your `config.yaml`.

```bash
python -m src.main train
```

- After training, the model will be saved to the path specified in `ai_model.model_path` in the config file. The trading bot will automatically load this model if AI is enabled.

### 3. Streamlit GUI Mode

This mode launches the web-based graphical user interface for real-time monitoring and control.

```bash
streamlit run src/gui.py
```

- Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
- From the GUI, you can start/stop the bot, view live charts, monitor your portfolio, and even edit the configuration in real-time.

---

## 📈 Strategy: Neural Pulse Scalper

The core strategy is designed to capture small, quick profits from short-term market volatility.

-   **Objective**: Scalp profits in short bursts using multi-indicator confirmation.
-   **Timeframe**: Primarily designed for the 1-minute (`1m`) chart, with optional confirmation from `3m` and `5m` charts.

### Core Components

1.  **Pre-Filter**: The bot first checks if market conditions are suitable for trading. It only considers pairs with sufficient 24h volume and recent volatility.
2.  **Entry Conditions**: A trade entry is only triggered if **all** of the following conditions are met, creating a strong, confirmed signal:
    - **EMA Crossover**: A fast EMA crosses a slow EMA.
    - **RSI Range**: The RSI is within a neutral zone (e.g., 40-60) to avoid entering overbought/oversold conditions.
    - **Bollinger Bands**: The price touches an outer band, indicating a potential reversal or mean reversion.
    - **VWAP**: The price reclaims or is rejected by the Volume Weighted Average Price.
    - **Volume Spike**: Trading volume is significantly higher than its recent average.
    - **Candlestick Pattern**: A reversal pattern (e.g., engulfing, pin bar) is detected.
3.  **AI Decision Node (Optional)**: If enabled, the signal is passed to a trained machine learning model which gives a final go/no-go decision based on learned historical patterns.
4.  **Smart Exit Conditions**: The bot exits a position based on a hierarchy of rules:
    - **Take Profit**: A fixed percentage target (e.g., +0.5%).
    - **Stop Loss**: A fixed percentage to limit losses (e.g., -0.4%).
    - **Trailing Stop**: Activates after a certain profit is reached and trails the price to lock in gains.
    - **Time Stop**: Automatically exits a trade if it's not profitable after a set number of minutes.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the bot, please feel free to fork the repository and submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
