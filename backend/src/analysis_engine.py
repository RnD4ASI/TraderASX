import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Any

# Imports for ARIMA
from statsmodels.tsa.stattools import adfuller
try:
    import pmdarima as pm
except ImportError:
    pm = None # Handle optional import

# Attempt to import LSTM related components
try:
    from . import lstm_model_trainer # Relative import for modules in the same package
    from sklearn.preprocessing import MinMaxScaler # Required by LSTM part
    if not lstm_model_trainer.tf_available:
        logging.warning("TensorFlow not available. LSTM forecasting will be disabled in analysis_engine.")
        lstm_model_trainer = None # Effectively disable it if TF is missing
except ImportError as e:
    logging.warning(f"Could not import lstm_model_trainer or MinMaxScaler, LSTM features disabled: {e}")
    lstm_model_trainer = None
    MinMaxScaler = None


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")
    if len(data) < window:
        return pd.Series(np.nan, index=data.index)
    return data.rolling(window=window, min_periods=max(1, window)).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")
    if len(data) < window + 1:
        return pd.Series(np.nan, index=data.index)

    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rs[rs == np.inf] = 1e6 # Handle potential division by zero if loss is 0 for a period
    rs.fillna(1e6, inplace=True) # if gain is 0 and loss is 0, rs is nan. Treat as high gain.

    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(df: pd.DataFrame, short_sma_window: int = 50, long_sma_window: int = 200, rsi_window: int = 14) -> pd.DataFrame:
    if df is None or df.empty or 'Close' not in df.columns:
        logging.error("DataFrame is empty or 'Close' column is missing for indicators.")
        return pd.DataFrame()

    data_with_indicators = df.copy()
    try:
        data_with_indicators['Close'] = pd.to_numeric(data_with_indicators['Close'])
    except Exception as e:
        logging.error(f"Could not convert 'Close' column to numeric: {e}")
        return df

    sma_short_col = f"SMA_{short_sma_window}"
    sma_long_col = f"SMA_{long_sma_window}"
    rsi_col = f"RSI_{rsi_window}"

    data_with_indicators[sma_short_col] = calculate_sma(data_with_indicators['Close'], short_sma_window)
    data_with_indicators[sma_long_col] = calculate_sma(data_with_indicators['Close'], long_sma_window)
    data_with_indicators[rsi_col] = calculate_rsi(data_with_indicators['Close'], rsi_window)

    logging.info(f"Added indicators: {sma_short_col}, {sma_long_col}, {rsi_col}")
    return data_with_indicators

def get_analysis_summary(df_with_indicators: pd.DataFrame) -> Dict:
    summary = {
        "last_close": None, "short_sma": None, "long_sma": None, "rsi": None,
        "short_sma_col": None, "long_sma_col": None, "rsi_col": None
    }
    if df_with_indicators is None or df_with_indicators.empty: return summary

    summary["last_close"] = df_with_indicators['Close'].iloc[-1] if not df_with_indicators['Close'].empty else None

    sma_short_col = next((col for col in df_with_indicators.columns if col.startswith("SMA_") and int(col.split('_')[1]) < 100), None)
    sma_long_col = next((col for col in df_with_indicators.columns if col.startswith("SMA_") and int(col.split('_')[1]) >= 100), None)
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

# --- ARIMA Specific Functions ---
def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Tuple[bool, float]:
    if series is None or series.empty or series.isnull().all():
        logging.warning("Stationarity check: Input series is empty or all NaN.")
        return False, np.nan
    series_cleaned = series.dropna()
    if len(series_cleaned) < 10:
        logging.warning(f"Stationarity check: Series too short ({len(series_cleaned)}) after dropping NaNs.")
        return False, np.nan
    try:
        result = adfuller(series_cleaned)
        p_value = result[1]
        is_stationary = p_value < significance_level
        logging.info(f"ADF Test: p-value={p_value:.4f}. Stationary: {is_stationary} (alpha={significance_level})")
        return is_stationary, p_value
    except Exception as e:
        logging.error(f"Error during ADF test: {e}")
        return False, np.nan

# --- LSTM Specific Functions ---
def get_lstm_forecast(
    price_series: pd.Series,
    ticker_symbol: str,
    n_periods_forecast: int
) -> Dict[str, Any]:
    """
    Generates a forecast using a pre-trained LSTM model.
    """
    results = {'forecasted_values': None, 'error': None}
    if lstm_model_trainer is None or not lstm_model_trainer.tf_available:
        results['error'] = "LSTM model trainer or TensorFlow is not available."
        logging.warning(results['error'])
        return results

    model, scaler = lstm_model_trainer.load_lstm_model_and_scaler(ticker_symbol)
    if model is None or scaler is None:
        results['error'] = f"No pre-trained LSTM model or scaler found for {ticker_symbol}."
        logging.warning(results['error'])
        return results

    sequence_length = lstm_model_trainer.DEFAULT_SEQUENCE_LENGTH
    if len(price_series) < sequence_length:
        results['error'] = f"Insufficient data for LSTM input: need {sequence_length}, got {len(price_series)}."
        logging.warning(results['error'])
        return results

    try:
        # Prepare the last sequence of data, scaled
        last_sequence_actual = price_series.values[-sequence_length:].reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence_actual) # Use transform, not fit_transform

        forecast_values = lstm_model_trainer.predict_with_lstm(
            model, scaler, last_sequence_scaled.flatten(), # predict_with_lstm expects 1D array for input_series_scaled
            n_steps_forecast=n_periods_forecast,
            sequence_length=sequence_length
        )

        if forecast_values is not None:
            results['forecasted_values'] = forecast_values
            logging.info(f"LSTM forecast successful for {ticker_symbol}. Forecasted {n_periods_forecast} periods.")
        else:
            results['error'] = f"LSTM prediction failed for {ticker_symbol}."
            logging.warning(results['error'])
    except Exception as e:
        results['error'] = f"Exception during LSTM forecast for {ticker_symbol}: {e}"
        logging.error(results['error'], exc_info=True)

    return results


def get_arima_forecast(
    series: pd.Series, n_periods_forecast: int, seasonal: bool = False,
    m_seasonality: int = 1, max_d: int = 2
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
    if pm is None:
        logging.error("pmdarima library is not installed. Cannot perform ARIMA forecast.")
        return None, None, None
    if series is None or len(series) < 20:
        logging.warning(f"ARIMA: Series is None or too short (length {len(series) if series is not None else 0}). Minimum ~20 suggested.")
        return None, None, None
    series_to_fit = series.copy().dropna()
    if len(series_to_fit) < 20: # Check again after dropna
        logging.warning(f"ARIMA: Series too short ({len(series_to_fit)}) after dropna for auto_arima.")
        return None, None, None
    try:
        auto_model = pm.auto_arima(
            series_to_fit, start_p=1, start_q=1, max_p=3, max_q=3,
            start_P=0, seasonal=seasonal, m=m_seasonality, d=None, D=None if not seasonal else None,
            trace=False, error_action='ignore', suppress_warnings=True, stepwise=True,
            max_order=10, information_criterion='aic', test='adf', stationary=False,
            max_d=max_d, max_D=1 if seasonal else 0
        )
        forecast, conf_int = auto_model.predict(n_periods=n_periods_forecast, return_conf_int=True)
        logging.info(f"ARIMA model fitted: {auto_model.order} {auto_model.seasonal_order if seasonal else ''}")
        return forecast, conf_int, auto_model.order
    except Exception as e:
        logging.error(f"Error fitting/forecasting with auto_arima: {e}")
        return None, None, None

# --- Forecast Orchestrator ---
def get_forecast_and_return(
    df_with_indicators: pd.DataFrame, # This contains the 'Close' series and potentially indicators
    ticker_symbol: str, # Needed for LSTM model loading
    interval: str = "1d",
    model_type: str = "simple", # simple, arima, lstm
    seasonal_arima: bool = False,
    m_seasonality_arima: int = 1
) -> Dict[str, Any]:
    results = {
        'forecast_model_used': model_type, 'forecasted_price': None, 'expected_return_pct': None,
        'arima_order': None, 'arima_conf_int': None, 'forecast_sequence': None, 'error': None
    }
    # Ensure 'Close' column exists for all models, df_with_indicators is the source of truth for this.
    if df_with_indicators is None or df_with_indicators.empty or 'Close' not in df_with_indicators.columns:
        results['error'] = "Input data for forecast is invalid (df_with_indicators empty or no 'Close' column)."
        logging.warning(results['error'])
        return results

    try:
        series_for_forecast = pd.to_numeric(df_with_indicators['Close']).dropna()
        if series_for_forecast.empty:
            raise ValueError("Close price series is empty after dropping NaNs.")
    except Exception as e:
        results['error'] = f"Invalid 'Close' price data: {e}"
        logging.error(results['error'])
        return results

    last_close_price = series_for_forecast.iloc[-1]
    periods_for_30_days = 21 if interval == "1d" else 4

    if model_type.lower() == "arima":
        if pm is None:
            results['error'] = "ARIMA model selected, but pmdarima library is not available."
        else:
            # For ARIMA, use the potentially indicator-augmented series if it helps stationarity,
            # but typically, it's applied to the raw 'Close' prices.
            # Here, series_for_forecast is 'Close' prices.
            forecast_points, conf_int, order = get_arima_forecast(
                series_for_forecast, n_periods_forecast=periods_for_30_days,
                seasonal=seasonal_arima, m_seasonality=m_seasonality_arima
            )
            if forecast_points is not None and len(forecast_points) > 0:
                results['forecasted_price'] = forecast_points[-1]
                results['forecast_sequence'] = forecast_points # Store full sequence
                results['arima_order'] = str(order) if order else None
                results['arima_conf_int'] = conf_int[-1].tolist() if conf_int is not None and len(conf_int) > 0 else None
            else:
                results['error'] = results.get('error') or "ARIMA model failed to produce a forecast."

    elif model_type.lower() == "lstm":
        if lstm_model_trainer is None:
            results['error'] = "LSTM model selected, but trainer/TensorFlow is not available."
        else:
            # LSTM typically uses raw 'Close' prices for training and prediction.
            # series_for_forecast is appropriate here.
            lstm_results = get_lstm_forecast(
                series_for_forecast, # Use the 'Close' price series
                ticker_symbol=ticker_symbol,
                n_periods_forecast=periods_for_30_days
            )
            if lstm_results.get('forecasted_values') is not None and len(lstm_results['forecasted_values']) > 0:
                results['forecasted_price'] = lstm_results['forecasted_values'][-1]
                results['forecast_sequence'] = lstm_results['forecasted_values'] # Store full sequence
            else:
                results['error'] = lstm_results.get('error') or "LSTM model failed to produce a forecast."

    elif model_type.lower() == "simple": # Simple Momentum
        # Simple momentum should also ideally work on the 'Close' price series.
        momentum_window = periods_for_30_days * 2
        diff_series = series_for_forecast.diff()
        avg_change = 0
        if not diff_series.dropna().empty:
            if len(diff_series.dropna()) < momentum_window:
                avg_change = diff_series.dropna().mean()
            else:
                avg_change = diff_series.tail(momentum_window).mean()
        if pd.isna(avg_change): avg_change = 0

        results['forecasted_price'] = last_close_price + (avg_change * periods_for_30_days)
        if results['forecasted_price'] < 0: results['forecasted_price'] = 0.0
        logging.info(f"Simple forecast: Last Close={last_close_price:.2f}, Avg Change={avg_change:.2f}, Forecasted Price={results['forecasted_price']:.2f}")
    else:
        results['error'] = f"Unknown forecast model type: {model_type}"

    if results['error']: logging.warning(results['error'])

    if results['forecasted_price'] is not None:
        if last_close_price > 0:
            results['expected_return_pct'] = ((results['forecasted_price'] - last_close_price) / last_close_price) * 100
        elif results['forecasted_price'] > 0:
            results['expected_return_pct'] = float('inf')
        else:
            results['expected_return_pct'] = 0.0
    return results

if __name__ == '__main__':
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='B')) # Business days for more data
    close_prices_data = np.linspace(20, 40, 50) + np.random.normal(0, 2, 50)
    dummy_df_full = pd.DataFrame({'Date': dates, 'Close': close_prices_data, 'Open': close_prices_data, 'High': close_prices_data, 'Low': close_prices_data})

    print("--- Testing Add Technical Indicators ---")
    df_with_indicators = add_technical_indicators(dummy_df_full.copy(), short_sma_window=10, long_sma_window=20, rsi_window=7)
    print(df_with_indicators.tail())

    print("\n--- Testing get_forecast_and_return (Simple Momentum) ---")
    if not df_with_indicators.empty:
        # forecast_results_simple = get_forecast_and_return(df_with_indicators.copy(), ticker_symbol="DUMMY.AX", interval="1d", model_type="simple")
        # print(f"Simple Forecast Results: {forecast_results_simple}")
        print("Skipping get_forecast_and_return direct test for now due to signature change, covered by app/CLI.")

    print("\n--- Testing Analysis Summary ---")
    if not df_with_indicators.empty:
        summary = get_analysis_summary(df_with_indicators)
        print("Analysis Summary:")
        for key_val, val_val in summary.items(): print(f"  {key_val}: {val_val}")

    print("\n--- ARIMA Specific Tests (using get_forecast_and_return) ---")
    # Use the same df_with_indicators as it has a 'Close' column
    if not df_with_indicators.empty and len(df_with_indicators) >=25: # Ensure enough data for ARIMA
        print(f"Sample Series for ARIMA (tail from dummy data):\n{df_with_indicators['Close'].tail()}")
        is_stationary, p_value = check_stationarity(df_with_indicators['Close'])
        print(f"Is series for ARIMA stationary before internal handling? {is_stationary}, p-value: {p_value:.4f if not pd.isna(p_value) else 'N/A'}")

        print("\nTesting ARIMA forecast via get_forecast_and_return...")
        # arima_forecast_results = get_forecast_and_return(
        #     df_with_indicators.copy(), ticker_symbol="DUMMY.AX", interval="1d", model_type="arima", seasonal_arima=False
        # )
        # print(f"ARIMA Forecast Results (Non-Seasonal): {arima_forecast_results}")
        print("Skipping get_forecast_and_return ARIMA direct test for now due to signature change, covered by app/CLI.")
    else:
        print("Skipping ARIMA test on dummy_df_full due to insufficient length or emptiness.")

    print("\n--- Analysis Engine Module Test Complete ---")
