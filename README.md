# ASX Trading Analysis and Recommendation System

This project aims to provide a trading analysis and recommendation system for ASX-listed stocks.

## Features

This system provides tools for analyzing ASX-listed stocks.

**Core Functionality (accessible via CLI and Web UI):**
*   **Data Crawling:** Extracts historical stock prices (OHLCV) and company financial information (`yfinance`).
*   **Data Cleansing:** Standard preprocessing for time series data.
*   **Technical Analysis:** Calculates Simple Moving Averages (SMAs) and Relative Strength Index (RSI).
*   **Price Forecasting (User Selectable Models):**
    *   Simple Momentum-based forecast.
    *   ARIMA (Autoregressive Integrated Moving Average) using `pmdarima` for auto-parameter selection.
    *   LSTM (Long Short-Term Memory) neural network, using pre-trained models per ticker (training via `lstm_model_trainer.py`).
    *   All models provide a 30-day equivalent price forecast and expected return.
*   **Volatility Forecasting (Optional):**
    *   GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model using `arch` to forecast conditional volatility.
*   **Backtesting:** Optionally backtests a simple SMA/RSI based trading strategy with a 30-day mandatory holding period.
*   **Recommendation Engine:** Provides BUY/HOLD/SELL recommendations. Confidence levels are influenced by technical signals, selected price forecast (including ARIMA CI width), and GARCH volatility forecast.

**Phase 1: Command Line Interface (CLI)**
*   Detailed console output of analysis steps and results.
*   Saves a static plot of price and indicators.

**Phase 2: Web User Interface (Streamlit)**
*   Interactive input controls for analysis parameters, including selection of price forecast models (Simple, ARIMA, LSTM) and GARCH options.
*   Dynamic display of company information, analysis metrics, recommendations, and backtest results.
*   Interactive charting of prices, indicators (Plotly), and optionally LSTM forecast sequence or GARCH volatility.
*   Light/Dark mode theme selection.
*   Basic chatbot stub for ASX/trading questions.

**Phase 3: Advanced Modeling (Ongoing)**
*   Integration of ARIMA, GARCH, and LSTM models into the analysis pipeline.
*   Refinement of the recommendation engine to consider outputs from these advanced models.

## Setup

1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

The main entry point for the CLI is `backend/src/main.py`. From the project root directory, you can run it using:

```bash
python -m backend.src.main [OPTIONS]
```

**Basic CLI Example (using simple forecast model):**

```bash
python -m backend.src.main --ticker "BHP.AX" --period "1y" --interval "daily" --forecast-model "simple"
```

**Example with Backtesting, Custom Windows, and ARIMA forecast:**

```bash
python -m backend.src.main -t "CBA.AX" -p "2y" -i "daily" -b --short-sma 20 --long-sma 50 --rsi-window 10 --initial-capital 5000 --forecast-model "arima"
```

**CLI Options:**

*   `--ticker TEXT` / `-t TEXT`: Stock ticker (e.g., 'BHP.AX'). Must end with .AX. (Required)
*   `--period TEXT` / `-p TEXT`: Predefined period (e.g., '1y', '2y', '5y', 'ytd', 'max'). Overrides start/end dates if provided. Defaults to '1y' if no date range is given.
*   `--start-date TEXT` / `-s TEXT`: Start date in YYYY-MM-DD format. Used if `--period` is not set.
*   `--end-date TEXT` / `-e TEXT`: End date in YYYY-MM-DD format. Used if `--period` is not set.
*   `--interval [daily|weekly]` / `-i [daily|weekly]`: Data interval. (Required)
*   `--run-backtest` / `-b`: Run backtesting with the default strategy. (Flag)
*   `--short-sma INTEGER`: Short SMA window.
    *   Default for daily: 50
    *   Default for weekly: 10
*   `--long-sma INTEGER`: Long SMA window.
    *   Default for daily: 200
    *   Default for weekly: 40
*   `--rsi-window INTEGER`: RSI window. (Default: 14)
*   `--initial-capital FLOAT`: Initial capital for backtesting. (Default: 10000)
*   `--forecast-model [simple|arima|lstm]`: Forecast model to use. (Default: 'simple')
*   `--arima-seasonal`: Enable seasonality for ARIMA model (if selected). (Flag)


**Output:**

The CLI will output:
1.  Progress of data fetching, cleansing, and analysis.
2.  Summary of company information (if available).
3.  Latest technical indicator values (SMAs, RSI).
4.  A 30-day price forecast and expected return.
5.  Backtesting results (if `--run-backtest` is used).
6.  A final trading recommendation (BUY, HOLD, SELL) with a confidence level (High, Medium, Low) and supporting reasons.
7.  A message indicating where the analysis plot (price, SMAs, RSI) has been saved (in the `output/` directory).

Example plot filename: `TICKER_AX_interval_analysis_plot.png`

## Project Structure

*   `backend/`: Contains all backend logic.
    *   `backend/src/`: Source code for the data processing, analysis, and CLI.
    *   `backend/data/`: Stores downloaded and processed data.
    *   `backend/output/`: Default location for generated plots and other output files.
    *   `backend/models/`: Stores pre-trained LSTM models.
*   `frontend/`: Contains the Streamlit web UI application.
    *   `frontend/app.py`: Main Streamlit application file.
*   `docs/`: Contains documentation files (e.g., `METHODOLOGY.md`).
*   `tests/`: Unit tests (primarily for backend logic).
*   `requirements.txt`: Python dependencies for both backend and frontend.
*   `AGENTS.md`: Notes for AI agent development.
*   `README.md`: This file.

## Disclaimer

This tool is for educational and informational purposes only. Trading and investing involve risk, and this tool does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.
