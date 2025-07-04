## Agent Development Notes for ASX Analyzer

This file contains notes and conventions for AI agents working on this project.

### Project Structure Overview:

*   **`backend/`**: Houses all core logic, data, and the CLI.
    *   `backend/src/`: Python modules for data crawling, cleansing, analysis, forecasting, backtesting, recommendations, and the CLI (`main.py`).
    *   `backend/data/`: Default storage for raw and processed data.
    *   `backend/output/`: Default storage for generated plots or other file outputs.
    *   `backend/models/`: Stores pre-trained models, particularly for LSTM.
*   **`frontend/`**: Contains the Streamlit web application.
    *   `frontend/app.py`: The main UI application script.
*   **`docs/`**: Project documentation.
*   **`tests/`**: Unit tests, primarily targeting backend modules.
*   **`requirements.txt`**: All Python dependencies for both backend and frontend.

### General Guidelines:

1.  **Modularity:**
    *   Backend: Each major functionality (data_crawler, data_cleanser, analysis_engine, lstm_model_trainer, backtester, recommender) is a separate module in `backend/src/`.
    *   Frontend: `frontend/app.py` handles UI, calling backend services.
2.  **Error Handling:** Implement robust error handling, especially for external API calls (e.g., `yfinance`), file operations, and model loading/prediction. Provide informative error messages.
3.  **Configuration:** File paths (like `MODELS_DIR`, `DATA_DIR`) are generally constructed using `os.path.join` relative to the script locations. More complex configurations could use a dedicated config file if needed.
4.  **Data Storage:**
    *   Raw data: `backend/data/raw/`
    *   Processed data: `backend/data/processed/`
    *   LSTM models: `backend/models/` (e.g., `lstm_model_TICKER_AX.keras`, `lstm_scaler_TICKER_AX.joblib`)
    *   Filenames should be descriptive.
5.  **Logging:** Use the `logging` module for tracking application flow and errors, especially in backend processes.
6.  **Docstrings and Comments:** Maintain clear docstrings for functions/classes and comments for complex logic.
7.  **Testing:** Aim for unit tests for core backend logic.
8.  **Python Version:** Python 3.8+.
9.  **Dependencies:** Keep `requirements.txt` updated. Key libraries include `pandas`, `numpy`, `yfinance`, `matplotlib`, `click`, `streamlit`, `plotly`, `statsmodels`, `pmdarima`, `arch`, `tensorflow`, `scikit-learn`.

### Specific Module Notes (Backend - `backend/src/`):

*   **`data_crawler.py`:** Handles `yfinance` interaction.
*   **`data_cleanser.py`:** Standard data cleaning.
*   **`analysis_engine.py`:**
    *   Calculates technical indicators (SMAs, RSI).
    *   Orchestrates forecasting via `get_forecast_and_return`, supporting "simple" momentum, "arima", and "lstm" models.
    *   Includes GARCH volatility forecasting.
*   **`lstm_model_trainer.py`:**
    *   Handles training, saving, and loading of LSTM models and their scalers. Models are ticker-specific.
    *   Training is an offline process; the application uses pre-trained models.
    *   Ensure `tensorflow` and `scikit-learn` are correctly installed for this.
*   **`backtester.py`:** Evaluates a predefined trading strategy (currently SMA/RSI based) with a 30-day hold rule.
*   **`recommender.py`:**
    *   Generates BUY/HOLD/SELL recommendations based on technicals, forecast details (from any model), GARCH output, and backtest results.
    *   Refer to `docs/METHODOLOGY.md` for detailed logic.
*   **`main.py` (CLI):**
    *   Uses `click` for arguments.
    *   Allows selection of forecast model (`--forecast-model`).

### Frontend Notes (`frontend/app.py`):

*   Uses Streamlit.
*   Calls functions from backend modules (ensure `sys.path` is adjusted correctly to find `backend.src`).
*   Provides UI for selecting forecast models, parameters, and viewing results including interactive plots.

### Key Workflow for Analysis:

1.  Data Crawling (`data_crawler`)
2.  Data Cleansing (`data_cleanser`)
3.  Technical Indicator Calculation (`analysis_engine.add_technical_indicators`)
4.  Forecasting (selected model via `analysis_engine.get_forecast_and_return` which may call `get_arima_forecast` or `get_lstm_forecast`)
5.  GARCH Volatility (optional, via `analysis_engine.get_garch_volatility_forecast` - primarily from UI)
6.  Backtesting (optional, via `backtester.run_backtest`)
7.  Recommendation (`recommender.generate_recommendation` using inputs from steps 3-6)

By following these guidelines, we can build a more maintainable and scalable application.tool_code
create_file_with_block
asx_analyzer/docs/METHODOLOGY.md
# Analysis and Recommendation Methodology

This document outlines the methodologies used in the ASX Trading Analysis and Recommendation System.

## 1. Data Acquisition

*   **Source:** Historical stock prices (Open, High, Low, Close, Volume - OHLCV) and company fundamental information are sourced from Yahoo Finance using the `yfinance` Python library.
*   **Frequency:** Users can select either daily or weekly data intervals.
*   **Time Window:** Data can be fetched for a user-specified period, up to a maximum of 5 years from the current date.
*   **Company Information:** Key financial metrics and company profile data (e.g., marketCap, trailingPE, forwardPE, dividendYield, sector, longBusinessSummary) are fetched from `yfinance`.

## 2. Data Cleansing

*   **Missing Values:**
    *   For OHLCV data, rows with any missing critical values (Open, High, Low, Close) may be forward-filled for short periods if appropriate for the interval, or reported if extensive. The `yfinance` library often provides data that is already well-maintained.
    *   Volume data is checked for anomalies (e.g., zero volume on trading days, though this can be legitimate for illiquid stocks).
*   **Adjusted Prices:** `yfinance` is configured to provide prices adjusted for dividends and stock splits (`auto_adjust=True`), ensuring historical price continuity.
*   **Data Types:** Columns are verified to ensure correct data types (e.g., datetime objects for dates, numeric types for price/volume).

## 3. Technical Analysis Indicators

The following technical indicators are used on the historical price data:

*   **Moving Averages (MAs):**
    *   **Simple Moving Average (SMA):** Calculated for short-term and long-term periods (e.g., 50-day and 200-day for daily data; 10-week and 40-week for weekly data).
    *   **Purpose:** Identify trends and potential support/resistance levels. Crossovers between short-term and long-term MAs can signal trend changes.
*   **Relative Strength Index (RSI):**
    *   **Calculation:** A standard 14-period RSI is calculated.
    *   **Purpose:** A momentum oscillator that measures the speed and change of price movements.
    *   **Interpretation:**
        *   RSI > 70: Often indicates overbought conditions.
        *   RSI < 30: Often indicates oversold conditions.
        *   Divergences between RSI and price can also be significant.

## 4. Forecasting (30-Day Holding Period)

Given the constraint that a stock, if bought, is restricted from being sold for 30 calendar days, the forecasting aims to project a potential price range or expectation at the end of this holding period.

*   **Initial Approach (Simplified):**
    *   The forecast will not be a complex predictive model in Phase 1.
    *   It will be based on extrapolating recent price momentum or using the current state of technical indicators.
    *   For example, if MAs suggest a strong uptrend, the forecast might project a modest continuation. If indicators are neutral, the forecast might be more conservative.
    *   **Calculation:** One simple method could be to calculate the average daily/weekly price change over a recent period (e.g., the last 30 or 60 days) and project this forward for the 20-22 trading days that approximate 30 calendar days. This is a very basic momentum projection.
*   **Expected Return:** Calculated as `((Forecasted Price at T+30 days - Current Price) / Current Price) * 100%`.

**Note:** This forecasting method is a heuristic and carries significant limitations. It does not account for unexpected news, market shifts, or complex price dynamics.

## 5. Backtesting

*   **Objective:** To evaluate the historical performance of a trading strategy based on the defined technical indicators and the 30-day holding rule.
*   **Strategy Example:**
    *   **Entry (Buy):** e.g., Short-term MA crosses above Long-term MA, AND RSI is not in overbought territory (>70).
    *   **Exit (Sell):** Mandatory hold for 30 calendar days. After 30 days, sell if, e.g., Short-term MA crosses below Long-term MA OR a predefined profit target/stop-loss is hit (profit target/stop-loss not in initial scope).
*   **Process:**
    1.  Iterate through historical data.
    2.  Identify buy signals based on the strategy.
    3.  If a buy signal occurs, simulate a purchase at the closing price of that day/week.
    4.  Hold the position for at least 30 calendar days.
    5.  After 30 days, evaluate sell conditions.
*   **Metrics:**
    *   Total Return
    *   Number of Trades
    *   Win Rate (Percentage of profitable trades)
    *   Average Gain per Winning Trade
    *   Average Loss per Losing Trade

## 6. Recommendation Logic

The final recommendation (BUY, HOLD, SELL) is derived by aggregating the signals from the technical analysis, the 30-day forecast, and optionally, the backtesting results.

*   **BUY:**
    *   Strong bullish signals from MAs (e.g., golden cross, price above both MAs).
    *   RSI indicating upward momentum (e.g., above 50, not excessively overbought).
    *   Positive 30-day expected return from the forecast.
    *   (If backtesting enabled) Strategy shows positive historical performance for similar conditions.
*   **SELL:**
    *   (Primarily for existing positions, or if the tool evolves to manage a portfolio). Strong bearish signals from MAs (e.g., death cross, price below both MAs).
    *   RSI indicating downward momentum (e.g., below 50, not excessively oversold).
    *   Negative 30-day expected return.
    *   This recommendation is less emphasized for a "new trade" focused tool but would apply if a previous BUY signal's 30-day hold period concludes with bearish signals.
*   **HOLD:**
    *   Mixed or neutral signals from indicators.
    *   Unclear trend.
    *   RSI in neutral territory.
    *   Forecasted return is marginal or uncertain.

*   **Confidence Level (Heuristic - Phase 1):**
    *   **High:** Multiple strong, aligning signals from different indicators and a clear forecast.
    *   **Medium:** Some positive/negative signals, but not all indicators are in strong agreement, or the forecast is modest.
    *   **Low:** Mixed signals, high uncertainty, or indicators are contradictory.

## Disclaimer

The methodologies described are based on common technical analysis techniques and simplified forecasting. They do not guarantee future performance. Financial markets are complex and subject to various unpredictable factors. This system is intended for educational and informational purposes and should not be considered financial advice.
