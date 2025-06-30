import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")
    if len(data) < window:
        # Return a series of NaNs with the same index if data is shorter than window
        logging.warning(f"Data length ({len(data)}) is less than SMA window ({window}). Returning NaNs.")
        return pd.Series(np.nan, index=data.index)
    return data.rolling(window=window, min_periods=max(1, window)).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")
    if len(data) < window + 1: # RSI needs at least window + 1 periods for first calculation
        logging.warning(f"Data length ({len(data)}) is less than RSI window+1 ({window+1}). Returning NaNs.")
        return pd.Series(np.nan, index=data.index)

    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # The first 'window' periods of RSI will be NaN due to the rolling mean of gain/loss.
    # The first delta is also NaN, so effectively first 'window' RSIs are NaN.
    return rsi

def add_technical_indicators(df: pd.DataFrame, short_sma_window: int = 50, long_sma_window: int = 200, rsi_window: int = 14) -> pd.DataFrame:
    """
    Adds SMA and RSI indicators to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with at least a 'Close' column.
        short_sma_window (int): Window for the short-term SMA.
        long_sma_window (int): Window for the long-term SMA.
        rsi_window (int): Window for RSI calculation.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    if df is None or df.empty or 'Close' not in df.columns:
        logging.error("DataFrame is empty or 'Close' column is missing.")
        return pd.DataFrame() # Return empty DataFrame

    data_with_indicators = df.copy()

    # Ensure 'Close' is numeric
    try:
        data_with_indicators['Close'] = pd.to_numeric(data_with_indicators['Close'])
    except Exception as e:
        logging.error(f"Could not convert 'Close' column to numeric: {e}")
        return df # Return original df or an empty one

    sma_short_col = f"SMA_{short_sma_window}"
    sma_long_col = f"SMA_{long_sma_window}"
    rsi_col = f"RSI_{rsi_window}"

    data_with_indicators[sma_short_col] = calculate_sma(data_with_indicators['Close'], short_sma_window)
    data_with_indicators[sma_long_col] = calculate_sma(data_with_indicators['Close'], long_sma_window)
    data_with_indicators[rsi_col] = calculate_rsi(data_with_indicators['Close'], rsi_window)

    logging.info(f"Added indicators: {sma_short_col}, {sma_long_col}, {rsi_col}")
    return data_with_indicators

def simple_forecast_30_day(
    df_with_indicators: pd.DataFrame,
    interval: str = "1d"
) -> Tuple[Optional[float], Optional[float]]:
    """
    Generates a simple 30-day price forecast and expected return.
    This is a very basic heuristic model for Phase 1.

    Args:
        df_with_indicators (pd.DataFrame): DataFrame with 'Close' prices and technical indicators.
                                           Assumed to be sorted by date, ascending.
        interval (str): "1d" for daily, "1wk" for weekly. Affects number of periods for 30 days.


    Returns:
        Tuple[Optional[float], Optional[float]]: (forecasted_price, expected_return_percent)
                                                 Returns (None, None) if forecast cannot be made.
    """
    if df_with_indicators is None or df_with_indicators.empty or 'Close' not in df_with_indicators.columns:
        logging.warning("Cannot forecast: DataFrame is empty or 'Close' column is missing.")
        return None, None

    last_close_price = df_with_indicators['Close'].iloc[-1]
    if pd.isna(last_close_price):
        logging.warning("Cannot forecast: Last close price is NaN.")
        return None, None

    # Determine number of periods for ~30 calendar days
    # Daily: ~20-22 trading days in 30 calendar days. Let's use 21.
    # Weekly: ~4-5 weeks in 30 calendar days. Let's use 4.
    periods_for_30_days = 21 if interval == "1d" else 4

    # Simple momentum: Average price change over the last N periods
    # Use a window similar to the forecast horizon for stability
    momentum_window = periods_for_30_days * 2 # Look back twice the forecast horizon

    if len(df_with_indicators['Close']) < momentum_window + 1:
        logging.warning(f"Not enough data for momentum calculation (need {momentum_window + 1}, have {len(df_with_indicators['Close'])}). Using simpler last change if possible.")
        if len(df_with_indicators['Close']) > 1:
             avg_change = df_with_indicators['Close'].diff().iloc[-1] # Last single period change
        else:
            logging.warning("Not enough data for even a single period change forecast.")
            return None, None # Cannot compute if less than 2 data points
    else:
        avg_change = df_with_indicators['Close'].diff().tail(momentum_window).mean()

    if pd.isna(avg_change):
        logging.warning("Average change calculation resulted in NaN. Cannot forecast.")
        # This can happen if all diffs are NaN (e.g. only one data point after diff from insufficient momentum window)
        # Or if the tail(momentum_window) contains only NaNs from diff()
        if len(df_with_indicators['Close']) > 1 and not pd.isna(df_with_indicators['Close'].diff().iloc[-1]):
             avg_change = df_with_indicators['Close'].diff().iloc[-1] # Fallback to last single period change
             logging.info(f"Fell back to last single period change: {avg_change}")
        else:
            logging.warning("Fallback to single period change also NaN or not possible.")
            return None, None


    forecasted_price = last_close_price + (avg_change * periods_for_30_days)

    # Basic sanity check: floor price at 0 (though unlikely for typical stocks)
    if forecasted_price < 0:
        forecasted_price = 0.0
        logging.info("Forecasted price was negative, floored to 0.")

    expected_return_percent = ((forecasted_price - last_close_price) / last_close_price) * 100 if last_close_price > 0 else 0.0

    logging.info(f"Simple 30-day forecast: Last Close={last_close_price:.2f}, Avg Change (over {momentum_window} periods)={avg_change:.2f}, Forecasted Price={forecasted_price:.2f}, Expected Return={expected_return_percent:.2f}%")

    return forecasted_price, expected_return_percent

def get_analysis_summary(df_with_indicators: pd.DataFrame) -> Dict:
    """
    Provides a summary of the latest indicator values.
    """
    summary = {
        "last_close": None,
        "short_sma": None,
        "long_sma": None,
        "rsi": None,
        "short_sma_col": None,
        "long_sma_col": None,
        "rsi_col": None
    }
    if df_with_indicators is None or df_with_indicators.empty:
        return summary

    summary["last_close"] = df_with_indicators['Close'].iloc[-1] if not df_with_indicators['Close'].empty else None

    # Dynamically find SMA and RSI columns (assuming standard naming from add_technical_indicators)
    sma_short_col = next((col for col in df_with_indicators.columns if col.startswith("SMA_") and int(col.split('_')[1]) < 100), None) # Heuristic for short SMA
    sma_long_col = next((col for col in df_with_indicators.columns if col.startswith("SMA_") and int(col.split('_')[1]) >= 100), None) # Heuristic for long SMA
    rsi_col = next((col for col in df_with_indicators.columns if col.startswith("RSI_")), None)

    if sma_short_col:
        summary["short_sma_col"] = sma_short_col
        summary["short_sma"] = df_with_indicators[sma_short_col].iloc[-1] if not df_with_indicators[sma_short_col].empty else None
    if sma_long_col:
        summary["long_sma_col"] = sma_long_col
        summary["long_sma"] = df_with_indicators[sma_long_col].iloc[-1] if not df_with_indicators[sma_long_col].empty else None
    if rsi_col:
        summary["rsi_col"] = rsi_col
        summary["rsi"] = df_with_indicators[rsi_col].iloc[-1] if not df_with_indicators[rsi_col].empty else None

    return summary


if __name__ == '__main__':
    # Create dummy data for testing
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                            '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                            '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                            '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20'])
    close_prices = pd.Series([10, 12, 11, 13, 15, 14, 16, 17, 18, 20,
                              19, 18, 20, 22, 23, 21, 20, 22, 24, 25], index=dates)
    dummy_df = pd.DataFrame({'Date': dates, 'Close': close_prices})
    dummy_df.set_index('Date', inplace=True) # RSI function expects Series with DateIndex for diff

    print("--- Testing SMA Calculation ---")
    sma_5 = calculate_sma(dummy_df['Close'], window=5)
    print("SMA 5:\n", sma_5)

    print("\n--- Testing RSI Calculation ---")
    rsi_14 = calculate_rsi(dummy_df['Close'], window=5) # Using smaller window for small dataset
    print("RSI 5 (window for test):\n", rsi_14)

    print("\n--- Testing Add Technical Indicators ---")
    # Reset index for add_technical_indicators as it expects 'Date' as a column
    df_for_indicators = dummy_df.reset_index()
    df_with_indicators = add_technical_indicators(df_for_indicators.copy(), short_sma_window=5, long_sma_window=10, rsi_window=5)
    print(df_with_indicators.tail())

    print("\n--- Testing Simple 30-Day Forecast (Daily) ---")
    if not df_with_indicators.empty:
        # Ensure enough data for forecast test
        forecast_dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        forecast_close_prices = np.linspace(20, 40, 60) + np.random.randn(60) * 2
        forecast_df_data = pd.DataFrame({'Date': forecast_dates, 'Close': forecast_close_prices})

        forecast_df_with_indicators = add_technical_indicators(forecast_df_data, short_sma_window=5, long_sma_window=10, rsi_window=5)

        forecasted_price, expected_return = simple_forecast_30_day(forecast_df_with_indicators, interval="1d")
        if forecasted_price is not None:
            print(f"Forecasted Price (1d): {forecasted_price:.2f}")
            print(f"Expected Return (1d): {expected_return:.2f}%")
        else:
            print("Could not generate daily forecast with test data.")

        print("\n--- Testing Simple 30-Day Forecast (Weekly) ---")
        # Create weekly-like data
        weekly_dates = pd.date_range(start='2023-01-01', periods=20, freq='W-MON')
        weekly_close_prices = np.linspace(20, 30, 20) + np.random.randn(20)
        weekly_df_data = pd.DataFrame({'Date': weekly_dates, 'Close': weekly_close_prices})
        weekly_df_with_indicators = add_technical_indicators(weekly_df_data, short_sma_window=4, long_sma_window=8, rsi_window=4)

        forecasted_price_w, expected_return_w = simple_forecast_30_day(weekly_df_with_indicators, interval="1wk")
        if forecasted_price_w is not None:
            print(f"Forecasted Price (1wk): {forecasted_price_w:.2f}")
            print(f"Expected Return (1wk): {expected_return_w:.2f}%")
        else:
            print("Could not generate weekly forecast with test data.")

    print("\n--- Testing Analysis Summary ---")
    if not df_with_indicators.empty:
        summary = get_analysis_summary(forecast_df_with_indicators) # Use the one with more data
        print("Analysis Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    # Test with insufficient data for SMA/RSI
    print("\n--- Testing with insufficient data for indicators ---")
    short_data = pd.DataFrame({'Close': [10,11]})
    short_data_indic = add_technical_indicators(short_data, short_sma_window=5, long_sma_window=10, rsi_window=5)
    print(short_data_indic) # Expect NaNs

    print("\n--- Analysis Engine Module Test Complete ---")
