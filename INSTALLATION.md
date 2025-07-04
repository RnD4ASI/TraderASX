# Installation and Usage Guide

This guide will walk you through installing the ASX Trading Analysis and Recommendation System and how to interact with its functionalities via the web front-end.

## 1. Prerequisites

*   **Python:** Ensure you have Python 3.8 or newer installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
*   **Git:** You'll need Git to clone the repository. You can download it from [git-scm.com](https://git-scm.com/downloads).

## 2. Installation Steps

Follow these steps to install the application:

### Step 2.1: Clone the Repository

Open your terminal or command prompt and run the following command to clone the project repository:

```bash
git clone <repository-url>
cd <repository-name>
```

### Step 2.2: Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

*   **Create the virtual environment:**

    ```bash
    python -m venv venv
    ```

*   **Activate the virtual environment:**
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    Your terminal prompt should change to indicate that the virtual environment is active (e.g., `(venv) Your-Computer:...`).

### Step 2.3: Install Dependencies

With the virtual environment active, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
This command will download and install all necessary libraries such as Streamlit, Pandas, yfinance, etc. This step might take a few minutes depending on your internet connection and the number of dependencies.

## 3. Running the Application (Web Front-end)

The primary way to interact with the application is through its Streamlit-based web interface.

### Step 3.1: Start the Streamlit Application

1.  Ensure your virtual environment is still active. If not, reactivate it (see Step 2.2).
2.  Navigate to the project's root directory in your terminal if you aren't already there.
3.  Run the following command to start the Streamlit application:

    ```bash
    streamlit run frontend/app.py
    ```

4.  Streamlit will typically open the application automatically in your default web browser. If it doesn't, your terminal will display a local URL (usually `http://localhost:8501`) which you can copy and paste into your browser's address bar.

## 4. Interacting with the Front-end

The web interface is designed for ease of use. Here's a breakdown of how to use its features:

### Main Layout

*   **Sidebar (Left Panel):** This is where you configure all the analysis parameters.
*   **Main Area (Right Panel):** This area displays the results of your analysis, including company information, recommendations, charts, and backtesting details.

### Sidebar: Configuration Options

1.  **Display Theme:**
    *   Choose between "Light" and "Dark" mode for the user interface.

2.  **Stock and Period:**
    *   **ASX Ticker:** Enter the ticker symbol for the Australian Securities Exchange (ASX) listed stock you want to analyze (e.g., `BHP.AX`, `CBA.AX`). **Important:** The ticker must end with `.AX`.
    *   **Analysis Period:** Select a predefined historical data range for the analysis (e.g., `1y` for one year, `5y` for five years, `YTD` for Year-To-Date).
    *   **Data Interval:** Choose between `Daily` or `Weekly` data points for the analysis.

3.  **Forecast & Volatility Models:**
    *   **Price Forecast Model:** Select the model to forecast future prices:
        *   `Simple Momentum`: A basic forecast based on recent price trends.
        *   `ARIMA`: A statistical model for time series forecasting. You can also enable/disable `ARIMA: Use Seasonality`.
        *   `LSTM`: A neural network model (requires a pre-trained model for the specific ticker to be available in the `backend/models/` directory).
    *   **Run GARCH Volatility Forecast:** Check this box to include a GARCH model for volatility forecasting. If checked, you can also set:
        *   `GARCH(p)`: The 'p' order of the GARCH model.
        *   `GARCH(q)`: The 'q' order of the GARCH model.

4.  **Technical Analysis Parameters:**
    *   **Reset TA Params to Defaults:** Click this button to reset SMA and RSI windows to their default values based on the selected Data Interval.
    *   **Short SMA Window:** Set the look-back period for the short-term Simple Moving Average.
    *   **Long SMA Window:** Set the look-back period for the long-term Simple Moving Average.
    *   **RSI Window:** Set the look-back period for the Relative Strength Index.

5.  **Backtesting (Optional):**
    *   **Run Backtest:** Check this box to simulate the performance of a trading strategy based on the configured SMAs and RSI.
    *   **Initial Capital:** If running a backtest, specify the starting amount of virtual money for the simulation.

6.  **Analyze Stock Button:**
    *   Once you have configured all your desired parameters, click the **`🚀 Analyze Stock`** button to run the analysis.

### Main Area: Viewing Results

After clicking "Analyze Stock," the main area will populate with the following sections (if applicable based on your selections):

1.  **Error Messages:** If any issues occur during analysis, error messages will be displayed here.
2.  **Company Information:**
    *   Displays the company name, sector, exchange, currency, and key financial metrics like Market Cap, P/E ratios, and Dividend Yield.
    *   Includes a business summary.
3.  **Recommendation & Forecast:**
    *   **Recommendation:** Shows the overall trading recommendation (BUY, HOLD, or SELL) along with a confidence level.
    *   **Price Forecast Model Used:** Indicates which model was used for the forecast.
    *   **30-Day Forecasted Price & Expected Return:** The projected price and percentage return for a 30-day outlook.
    *   **ARIMA Confidence Interval (if ARIMA used):** Shows the 95% confidence interval for the ARIMA forecast.
    *   **GARCH Volatility Forecast (if run):** An expandable section showing a chart of the forecasted conditional standard deviation (volatility).
4.  **Technical Analysis Summary:**
    *   Displays the latest values for Last Close Price, Short SMA, Long SMA, and RSI.
    *   An expandable section provides "Detailed Reasoning for Recommendation."
5.  **Price Chart & Indicators:**
    *   An interactive Plotly chart showing:
        *   Historical Close Price.
        *   Short and Long SMAs.
        *   Forecasted price sequence (if LSTM model was used and successful).
        *   Buy/Sell markers from the backtest (if backtesting was run).
    *   A separate panel below the price chart shows the RSI indicator with overbought (70) and oversold (30) levels.
6.  **Backtest Results Details (if backtesting was run):**
    *   An expandable section showing:
        *   Strategy Performance metrics: Final Portfolio Value, Total Return, Win Rate, Number of Trades.
        *   Trades History: A table detailing each simulated trade.

### Sidebar: ASX Helper Chatbot (Beta)

*   At the bottom of the sidebar, there's an expandable "ASX Helper Chatbot."
*   You can ask basic questions about trading concepts or ASX terms (e.g., "What is SMA?", "What is LSTM?").

## 5. Stopping the Application

To stop the Streamlit application, go back to your terminal where the application is running and press `Ctrl+C`.

## 6. Troubleshooting (Common Issues)

*   **`ModuleNotFoundError`:**
    *   Ensure your virtual environment is active.
    *   Verify that `pip install -r requirements.txt` completed successfully without errors.
    *   If the error pertains to backend modules when running `frontend/app.py`, it might be a path issue. The application attempts to handle this, but ensure your project structure matches the original.
*   **`yfinance` data fetching issues:**
    *   Ensure you have an active internet connection.
    *   Yahoo Finance might occasionally have temporary issues or change its API.
*   **LSTM Model Not Found:**
    *   If you select the LSTM forecast model, it requires a pre-trained model file (e.g., `lstm_model_TICKER_AX.keras`) and a scaler file (e.g., `lstm_scaler_TICKER_AX.joblib`) to be present in the `backend/models/` directory for the specific ticker. If these are missing, LSTM forecasting will fail for that ticker. Training these models is a separate process (see `backend/src/lstm_model_trainer.py`).

This guide should help you get the ASX Trading Analysis and Recommendation System up and running. Happy analyzing!
```
