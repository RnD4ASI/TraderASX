import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the data directory relative to this file's location
# Assuming this file is in asx_analyzer/src/
# So, the project root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Ensure data directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)

MAX_YEARS_WINDOW = 5

def validate_ticker(ticker: str) -> bool:
    """Validates if the ticker ends with '.AX'."""
    if not ticker.endswith(".AX"):
        logging.error(f"Invalid ticker format: {ticker}. Must end with '.AX'.")
        return False
    return True

def get_historical_prices(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame | None:
    """
    Fetches historical stock prices (OHLCV, Dividends, Splits) for a given ASX ticker.

    Args:
        ticker (str): The stock ticker (e.g., "BHP.AX").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        interval (str): Data interval, either "1d" (daily) or "1wk" (weekly).

    Returns:
        pd.DataFrame | None: DataFrame with historical data, or None if an error occurs.
    """
    if not validate_ticker(ticker):
        return None

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        return None

    # Enforce maximum 5-year time window
    if (end_dt - start_dt) > timedelta(days=MAX_YEARS_WINDOW * 365.25): # Account for leap years
        logging.error(f"Time window exceeds maximum of {MAX_YEARS_WINDOW} years.")
        # Adjust start_date to be 5 years before end_date
        start_dt = end_dt - timedelta(days=MAX_YEARS_WINDOW * 365.25)
        start_date = start_dt.strftime("%Y-%m-%d")
        logging.warning(f"Adjusted start date to {start_date} to meet 5-year limit.")


    if interval not in ["1d", "1wk"]:
        logging.error(f"Invalid interval: {interval}. Must be '1d' or '1wk'.")
        return None

    try:
        logging.info(f"Fetching historical prices for {ticker} from {start_date} to {end_date} with interval {interval}.")
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)

        if history.empty:
            logging.warning(f"No data found for ticker {ticker} for the given period/interval.")
            return None

        # Ensure the index is timezone-naive for consistent CSV output (yfinance can return tz-aware)
        if history.index.tz is not None:
            history.index = history.index.tz_localize(None)

        history.reset_index(inplace=True) # Make Date a column
        # Rename 'Date' or 'Datetime' column to 'Date' consistently
        if 'Datetime' in history.columns:
            history.rename(columns={'Datetime': 'Date'}, inplace=True)


        return history

    except Exception as e:
        logging.error(f"Error fetching historical prices for {ticker}: {e}")
        return None

def get_company_info(ticker: str) -> dict | None:
    """
    Fetches company financial information for a given ASX ticker.

    Args:
        ticker (str): The stock ticker (e.g., "BHP.AX").

    Returns:
        dict | None: Dictionary with company information, or None if an error occurs.
    """
    if not validate_ticker(ticker):
        return None

    try:
        logging.info(f"Fetching company information for {ticker}.")
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get('quoteType') == 'NONE': # Check for empty or minimal info
             logging.warning(f"Could not retrieve detailed company info for {ticker}. It might be delisted or invalid.")
             return None

        required_fields = ['marketCap', 'trailingPE', 'forwardPE', 'dividendYield',
                           'sector', 'longBusinessSummary', 'exchange', 'currency', 'shortName', 'symbol']

        selected_info = {}
        for field in required_fields:
            selected_info[field] = info.get(field) # Use .get() to avoid KeyError if a field is missing

        # Add a timestamp for when the data was fetched
        selected_info['fetchedTimestamp'] = datetime.now().isoformat()

        return selected_info

    except Exception as e:
        logging.error(f"Error fetching company information for {ticker}: {e}")
        return None

def save_data_to_csv(df: pd.DataFrame, ticker: str, interval: str, data_type: str = "prices") -> str | None:
    """
    Saves a DataFrame to a CSV file in the data/raw directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        ticker (str): The stock ticker.
        interval (str): Data interval ('1d' or '1wk'). Used in filename.
        data_type (str): Type of data, e.g., "prices".

    Returns:
        str | None: The path to the saved file, or None if error.
    """
    if df is None or df.empty:
        logging.warning("DataFrame is empty. Nothing to save.")
        return None

    safe_ticker = ticker.replace(".AX", "_AX") # Make filename safer
    filename = f"{safe_ticker}_{interval}_{data_type}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)

    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Successfully saved {data_type} data for {ticker} to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving data to CSV {filepath}: {e}")
        return None

def save_info_to_json(info: dict, ticker: str) -> str | None:
    """
    Saves company information (dictionary) to a JSON file in the data/raw directory.

    Args:
        info (dict): The company information dictionary.
        ticker (str): The stock ticker.

    Returns:
        str | None: The path to the saved file, or None if error.
    """
    if not info:
        logging.warning("Info dictionary is empty. Nothing to save.")
        return None

    safe_ticker = ticker.replace(".AX", "_AX")
    filename = f"{safe_ticker}_info.json"
    filepath = os.path.join(RAW_DATA_DIR, filename)

    try:
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=4)
        logging.info(f"Successfully saved company info for {ticker} to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving info to JSON {filepath}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Raw Data Dir: {RAW_DATA_DIR}")

    # Test fetching historical prices
    test_ticker_prices = "CBA.AX" # Commonwealth Bank
    test_start_date = (datetime.now() - timedelta(days=365*1)).strftime("%Y-%m-%d") # 1 year ago
    test_end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n--- Testing Historical Prices ({test_ticker_prices}) ---")
    daily_prices = get_historical_prices(test_ticker_prices, test_start_date, test_end_date, interval="1d")
    if daily_prices is not None:
        print(f"Daily prices for {test_ticker_prices}:\n{daily_prices.head()}")
        save_data_to_csv(daily_prices, test_ticker_prices, "1d")
        # Test with a very long period to check 5-year limit
        long_start = "2010-01-01"
        get_historical_prices(test_ticker_prices, long_start, test_end_date, interval="1d")


    weekly_prices = get_historical_prices(test_ticker_prices, test_start_date, test_end_date, interval="1wk")
    if weekly_prices is not None:
        print(f"\nWeekly prices for {test_ticker_prices}:\n{weekly_prices.head()}")
        save_data_to_csv(weekly_prices, test_ticker_prices, "1wk")

    # Test fetching company info
    test_ticker_info = "WES.AX" # Wesfarmers
    print(f"\n--- Testing Company Info ({test_ticker_info}) ---")
    company_info = get_company_info(test_ticker_info)
    if company_info:
        print(f"Company info for {test_ticker_info}:")
        for key, value in company_info.items():
            if isinstance(value, str) and len(value) > 70: # Truncate long strings for printing
                print(f"  {key}: {value[:70]}...")
            else:
                print(f"  {key}: {value}")
        save_info_to_json(company_info, test_ticker_info)

    # Test invalid ticker
    print("\n--- Testing Invalid Ticker ---")
    invalid_prices = get_historical_prices("INVALIDTICKER", test_start_date, test_end_date)
    invalid_info = get_company_info("INVALIDTICKER")

    # Test ticker not ending in .AX
    print("\n--- Testing Ticker not ending in .AX ---")
    non_ax_ticker = "GOOG"
    get_historical_prices(non_ax_ticker, test_start_date, test_end_date)
    get_company_info(non_ax_ticker)

    # Test with a ticker that might not have much info (e.g., a delisted or very small one)
    # This requires knowing such a ticker; for now, we assume yfinance handles it gracefully.
    # Example: test_ticker_problematic = "XYZ.AX" (if known to be problematic)
    # get_company_info(test_ticker_problematic)

    print("\n--- Data Crawler Module Test Complete ---")
