import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define data directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_price_data_from_csv(ticker: str, interval: str) -> pd.DataFrame | None:
    """
    Loads historical price data from a CSV file stored in data/raw/.

    Args:
        ticker (str): The stock ticker (e.g., "BHP.AX").
        interval (str): Data interval ('1d' or '1wk').

    Returns:
        pd.DataFrame | None: DataFrame with price data, or None if an error occurs.
    """
    safe_ticker = ticker.replace(".AX", "_AX")
    filename = f"{safe_ticker}_{interval}_prices.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        logging.error(f"Price data file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded price data from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading price data from {filepath}: {e}")
        return None

def cleanse_price_data(df: pd.DataFrame, ticker: str, interval: str) -> Tuple[pd.DataFrame | None, dict]:
    """
    Cleanses the historical price DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with price data.
        ticker (str): The stock ticker, for logging/reporting.
        interval (str): The data interval, for logging/reporting.

    Returns:
        Tuple[pd.DataFrame | None, dict]: A tuple containing the cleansed DataFrame
                                          and a dictionary with cleansing metadata/report.
                                          Returns (None, report) if critical error.
    """
    if df is None or df.empty:
        logging.warning(f"Input DataFrame for {ticker} ({interval}) is empty. No cleansing performed.")
        return None, {"status": "empty_input", "message": "Input DataFrame was empty."}

    report = {
        "ticker": ticker,
        "interval": interval,
        "original_rows": len(df),
        "rows_after_cleansing": 0,
        "datetime_conversion": "pending",
        "nan_handling": {"checked_columns": [], "nan_counts_before": {}, "nan_counts_after": {}, "strategy": "None"},
        "duplicate_dates_removed": 0,
        "status": "pending",
        "messages": []
    }

    cleansed_df = df.copy()

    # 1. DateTime Conversion
    # Ensure 'Date' column exists and convert to datetime objects
    if 'Date' not in cleansed_df.columns:
        logging.error(f"'Date' column not found in DataFrame for {ticker} ({interval}).")
        report["status"] = "error"
        report["messages"].append("'Date' column missing.")
        return None, report
    try:
        cleansed_df['Date'] = pd.to_datetime(cleansed_df['Date'])
        report["datetime_conversion"] = "success"
    except Exception as e:
        logging.error(f"Error converting 'Date' column to datetime for {ticker} ({interval}): {e}")
        report["datetime_conversion"] = f"failed: {e}"
        report["status"] = "error"
        report["messages"].append(f"Date conversion failed: {e}")
        return None, report

    # 2. Sort by Date (important for time series operations like ffill)
    cleansed_df.sort_values(by='Date', inplace=True)
    cleansed_df.reset_index(drop=True, inplace=True) # Reset index after sort

    # 3. Handle Missing Values (NaNs)
    # Define critical OHLCV columns
    ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    report["nan_handling"]["checked_columns"] = ohlcv_columns

    # Check if all required columns exist
    missing_cols = [col for col in ohlcv_columns if col not in cleansed_df.columns]
    if missing_cols:
        msg = f"Missing critical OHLCV columns: {missing_cols} for {ticker} ({interval})."
        logging.error(msg)
        report["status"] = "error"
        report["messages"].append(msg)
        return None, report

    # Initial NaN counts
    for col in ohlcv_columns:
        report["nan_handling"]["nan_counts_before"][col] = int(cleansed_df[col].isnull().sum())

    # Strategy: Forward-fill for OHLC, fill Volume with 0 (or mean/median if preferred)
    # For simplicity, yfinance data is often complete, but this handles potential gaps.
    # A small number of NaNs at the beginning might remain if ffill is used.
    cleansed_df['Open'] = cleansed_df['Open'].ffill()
    cleansed_df['High'] = cleansed_df['High'].ffill()
    cleansed_df['Low'] = cleansed_df['Low'].ffill()
    cleansed_df['Close'] = cleansed_df['Close'].ffill()
    cleansed_df['Volume'] = cleansed_df['Volume'].ffill().fillna(0) # ffill then fill any remaining with 0

    report["nan_handling"]["strategy"] = "ffill for OHLC, ffill then 0 for Volume"

    # NaN counts after filling
    for col in ohlcv_columns:
        report["nan_handling"]["nan_counts_after"][col] = int(cleansed_df[col].isnull().sum())
        if report["nan_handling"]["nan_counts_after"][col] > 0:
            msg = f"Warning: Column '{col}' still contains NaNs after ffill for {ticker} ({interval}). This might occur if NaNs are at the beginning of the series."
            logging.warning(msg)
            report["messages"].append(msg)

    # Drop any rows that still have NaNs in critical columns (e.g., if NaNs were at the start and couldn't be ffilled)
    initial_rows_before_na_drop = len(cleansed_df)
    cleansed_df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Volume already handled
    rows_dropped_na = initial_rows_before_na_drop - len(cleansed_df)
    if rows_dropped_na > 0:
        msg = f"Dropped {rows_dropped_na} rows with remaining NaNs in critical OHLC columns for {ticker} ({interval})."
        logging.info(msg)
        report["messages"].append(msg)


    # 4. Ensure Correct Numeric Data Types for OHLCV
    for col in ohlcv_columns:
        try:
            cleansed_df[col] = pd.to_numeric(cleansed_df[col])
        except Exception as e:
            msg = f"Error converting column '{col}' to numeric for {ticker} ({interval}): {e}"
            logging.error(msg)
            report["status"] = "error"
            report["messages"].append(msg)
            return None, report
    report["numeric_conversion"] = "success for OHLCV"


    # 5. Check for and Remove Duplicate Dates (keeping the first occurrence)
    initial_rows_before_duplicates_drop = len(cleansed_df)
    cleansed_df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
    report["duplicate_dates_removed"] = initial_rows_before_duplicates_drop - len(cleansed_df)
    if report["duplicate_dates_removed"] > 0:
        msg = f"Removed {report['duplicate_dates_removed']} duplicate date entries for {ticker} ({interval})."
        logging.info(msg)
        report["messages"].append(msg)


    # 6. Final checks
    if cleansed_df.empty:
        logging.warning(f"DataFrame became empty after cleansing for {ticker} ({interval}).")
        report["status"] = "empty_after_cleansing"
        report["messages"].append("DataFrame empty after cleansing.")
        report["rows_after_cleansing"] = 0
        return None, report

    report["rows_after_cleansing"] = len(cleansed_df)
    if report["status"] == "pending": # If no errors occurred
        report["status"] = "success"
        report["messages"].append(f"Data cleansing completed successfully for {ticker} ({interval}).")

    logging.info(f"Cleansing report for {ticker} ({interval}): {report}")
    return cleansed_df, report

def save_cleansed_data(df: pd.DataFrame, ticker: str, interval: str) -> str | None:
    """
    Saves the cleansed DataFrame to a CSV file in the data/processed directory.

    Args:
        df (pd.DataFrame): The cleansed DataFrame to save.
        ticker (str): The stock ticker.
        interval (str): Data interval ('1d' or '1wk').

    Returns:
        str | None: The path to the saved file, or None if error.
    """
    if df is None or df.empty:
        logging.warning("Cleansed DataFrame is empty. Nothing to save.")
        return None

    safe_ticker = ticker.replace(".AX", "_AX")
    filename = f"{safe_ticker}_{interval}_prices_cleansed.csv"
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)

    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Successfully saved cleansed data for {ticker} ({interval}) to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving cleansed data to CSV {filepath}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage: Requires data to be present from data_crawler.py
    # Ensure you have run data_crawler.py first to generate sample data.

    test_ticker = "CBA.AX" # Should have been created by data_crawler
    test_interval = "1d"

    print(f"\n--- Testing Data Cleanser for {test_ticker} ({test_interval}) ---")
    raw_df = load_price_data_from_csv(test_ticker, test_interval)

    if raw_df is not None:
        print(f"Raw data loaded. Shape: {raw_df.shape}")
        print(raw_df.head())
        print("\nApplying cleansing...")
        cleansed_df, report = cleanse_price_data(raw_df, test_ticker, test_interval)

        print(f"\nCleansing Report for {test_ticker} ({test_interval}):")
        for k, v in report.items():
            print(f"  {k}: {v}")

        if cleansed_df is not None:
            print(f"\nCleansed data shape: {cleansed_df.shape}")
            print("Cleansed data types:\n", cleansed_df.dtypes)
            print("Cleansed data head:\n", cleansed_df.head())
            print("Cleansed data tail (to check last NaNs if any):\n", cleansed_df.tail())

            # Check for NaNs specifically
            print("\nNaN counts in cleansed data:")
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in cleansed_df.columns:
                    print(f"  {col}: {cleansed_df[col].isnull().sum()}")
                else:
                    print(f"  {col}: Not found")


            save_path = save_cleansed_data(cleansed_df, test_ticker, test_interval)
            if save_path:
                print(f"Cleansed data saved to: {save_path}")
        else:
            print(f"Data cleansing failed or resulted in empty DataFrame for {test_ticker} ({test_interval}).")
    else:
        print(f"Could not load raw data for {test_ticker} ({test_interval}) to test cleanser.")

    # Test with a non-existent file
    print("\n--- Testing with non-existent data ---")
    non_existent_df = load_price_data_from_csv("NONEXISTENT.AX", "1d")
    if non_existent_df is None:
        print("Correctly handled non-existent file.")

    print("\n--- Data Cleanser Module Test Complete ---")
