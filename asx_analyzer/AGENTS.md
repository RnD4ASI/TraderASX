## Agent Development Notes for ASX Analyzer

This file contains notes and conventions for AI agents working on this project.

### General Guidelines:

1.  **Modularity:** Strive for modular code. Each major functionality (data crawling, cleansing, analysis, backtesting, recommendation, UI) should be in its own Python module within the `src/` directory.
2.  **Error Handling:** Implement robust error handling, especially for external API calls (e.g., `yfinance`) and file operations. Provide informative error messages to the user.
3.  **Configuration:** For now, configurations like file paths or API keys (if any in the future) can be managed as constants within relevant modules. For more complex configurations, consider a `config.py` or environment variables.
4.  **Data Storage:**
    *   Raw downloaded data should be stored in the `data/raw/` subdirectory (e.g., `data/raw/TICKER_prices.csv`, `data/raw/TICKER_info.json`).
    *   Processed/cleansed data can be stored in `data/processed/` if intermediate steps are saved.
    *   Filenames should be descriptive, including ticker, data type, and interval if applicable.
5.  **Logging:** Implement basic logging (using the `logging` module) for debugging and tracking application flow, especially in more complex functions.
6.  **Docstrings and Comments:** Write clear docstrings for all functions and classes, explaining their purpose, arguments, and return values. Use comments for complex logic.
7.  **Testing:** While full TDD might not be enforced for all initial exploratory code, aim to write unit tests for core logic, especially in data processing and analysis functions. Store tests in the `tests/` directory.
8.  **Python Version:** Assume Python 3.8+ for compatibility with modern libraries.
9.  **Dependencies:** Keep `requirements.txt` updated with all necessary packages.

### Specific Module Notes:

*   **`src/data_crawler.py`:**
    *   Ensure ASX tickers are correctly formatted (e.g., "BHP.AX").
    *   Handle API rate limits if they become an issue (though `yfinance` is generally robust for typical use).
    *   Clearly define the structure for saved data (CSV for tabular, JSON for dict-like info).
*   **`src/data_cleanser.py`:**
    *   Document cleansing steps taken.
    *   Be clear about how missing data (NaNs) is handled (e.g., fill, drop, interpolation).
*   **`src/analysis_engine.py`:**
    *   Ensure calculations for indicators (MAs, RSI) are standard.
    *   The 30-day forecast needs to be clearly defined. It's a simplification, so its limitations should be understood.
*   **`src/backtester.py`:**
    *   The 30-day holding rule is crucial. Trades should not be closed before this period unless a stop-loss is explicitly defined (not in initial scope).
*   **`src/recommender.py`:**
    *   The rules for BUY/HOLD/SELL should be transparent and documented in `METHODOLOGY.md`.
    *   Confidence level will be heuristic initially.
*   **`src/main.py` (CLI):**
    *   Use a library like `click` or `argparse` for clean command-line arguments.
    *   Provide clear output to the user.
    *   For plots, save them to a file and tell the user where to find them.

### Phase 2 (Web UI) Considerations (Future):

*   When moving to a web UI (Streamlit/Dash), ensure the backend logic (crawling, analysis, etc.) remains decoupled from the UI code as much as possible. The UI should primarily call functions from the core modules.
*   Think about how user inputs will be handled in the web interface.
*   Interactive plotting will be a key feature.

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
