import numpy as np
import pandas as pd
import os
import logging

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split # Optional: if splitting here
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf_available = True
except ImportError:
    tf_available = False
    logging.error("TensorFlow or scikit-learn not found. LSTM functionality will be disabled.")
    # Define dummy classes/functions if tf is not available to avoid import errors elsewhere if this file is imported
    MinMaxScaler = None
    Sequential = None
    load_model = None
    LSTM = Dense = Dropout = None
    EarlyStopping = None


import joblib # For saving the scaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_SEQUENCE_LENGTH = 60 # Number of past days' data to use for predicting the next day
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32

def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of data for LSTM model.
    Args:
        data (np.ndarray): Scaled time series data.
        sequence_length (int): Length of each input sequence.
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (sequences), y (targets)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length]) # Predict the next value
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length: int, n_features: int = 1) -> Optional[Sequential]:
    """
    Builds a simple LSTM model.
    Args:
        sequence_length (int): Input sequence length.
        n_features (int): Number of features in the input data (usually 1 for price).
    Returns:
        Optional[tf.keras.models.Sequential]: Compiled Keras LSTM model or None if TF not available.
    """
    if not tf_available or Sequential is None:
        return None

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=n_features) # Output layer with n_features (usually 1 for price)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(
    price_series: pd.Series,
    ticker_symbol: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = 0.1 # Use part of training data for validation
) -> bool:
    """
    Trains an LSTM model for the given price series and saves it.
    """
    if not tf_available or MinMaxScaler is None:
        logging.error(f"Cannot train LSTM for {ticker_symbol}: TensorFlow or scikit-learn not available.")
        return False
    if price_series is None or len(price_series) < sequence_length * 2: # Need enough data
        logging.error(f"Cannot train LSTM for {ticker_symbol}: Insufficient data (need at least {sequence_length * 2}, got {len(price_series)}).")
        return False

    logging.info(f"Starting LSTM model training for {ticker_symbol}...")

    # 1. Preprocessing
    data_values = price_series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    # 2. Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    if X.shape[0] == 0:
        logging.error(f"Could not create sequences for {ticker_symbol}. Check data length and sequence length.")
        return False

    # Reshape X for LSTM [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 3. Build Model
    model = build_lstm_model(sequence_length, n_features=1)
    if model is None: return False # Should not happen if tf_available was true

    # 4. Train Model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    try:
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1 # Can be set to 0 for less output
        )
        logging.info(f"LSTM model training completed for {ticker_symbol}.")
    except Exception as e:
        logging.error(f"Error during LSTM model training for {ticker_symbol}: {e}")
        return False

    # 5. Save Model and Scaler
    model_filename = f"lstm_model_{ticker_symbol.replace('.AX', '_AX')}.keras"
    scaler_filename = f"lstm_scaler_{ticker_symbol.replace('.AX', '_AX')}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    scaler_path = os.path.join(MODELS_DIR, scaler_filename)

    try:
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"LSTM Model saved to {model_path}")
        logging.info(f"Scaler saved to {scaler_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving LSTM model or scaler for {ticker_symbol}: {e}")
        return False

def load_lstm_model_and_scaler(ticker_symbol: str) -> Tuple[Optional[Sequential], Optional[MinMaxScaler]]:
    if not tf_available or load_model is None: return None, None

    model_filename = f"lstm_model_{ticker_symbol.replace('.AX', '_AX')}.keras"
    scaler_filename = f"lstm_scaler_{ticker_symbol.replace('.AX', '_AX')}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    scaler_path = os.path.join(MODELS_DIR, scaler_filename)

    model, scaler = None, None
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            logging.info(f"Loaded LSTM model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading LSTM model for {ticker_symbol} from {model_path}: {e}")
            model = None # Ensure it's None if loading failed
    else:
        logging.warning(f"LSTM model file not found for {ticker_symbol} at {model_path}")

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logging.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logging.error(f"Error loading scaler for {ticker_symbol} from {scaler_path}: {e}")
            scaler = None # Ensure it's None
    else:
        logging.warning(f"Scaler file not found for {ticker_symbol} at {scaler_path}")
        if model is not None: # If model exists but scaler doesn't, that's an issue
             logging.error("Model found but scaler is missing. Cannot use LSTM model.")
             model = None # Invalidate model if scaler is missing

    return model, scaler


def predict_with_lstm(
    model: Sequential,
    scaler: MinMaxScaler,
    input_series_scaled: np.ndarray, # Last `sequence_length` of scaled data points
    n_steps_forecast: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
) -> Optional[np.ndarray]:
    """
    Generates future predictions using the trained LSTM model.
    Iteratively predicts one step at a time and uses the prediction as input for the next.
    """
    if not tf_available or model is None or scaler is None:
        logging.error("Cannot predict with LSTM: Model, scaler, or TensorFlow not available.")
        return None
    if len(input_series_scaled) < sequence_length:
        logging.error(f"Input series length ({len(input_series_scaled)}) is less than sequence length ({sequence_length}).")
        return None

    current_sequence = input_series_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    forecast_scaled = []

    try:
        for _ in range(n_steps_forecast):
            next_pred_scaled = model.predict(current_sequence, verbose=0) # Predict next step
            forecast_scaled.append(next_pred_scaled[0,0]) # Append the scalar prediction
            # Update the sequence: remove oldest, append newest prediction
            current_sequence = np.append(current_sequence[:, 1:, :], [[next_pred_scaled]], axis=1)

        # Inverse transform the forecast
        forecast_actual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
        return forecast_actual.flatten()
    except Exception as e:
        logging.error(f"Error during LSTM prediction: {e}")
        return None

if __name__ == '__main__':
    if not tf_available:
        print("TensorFlow/scikit-learn not installed. Skipping LSTM trainer example.")
    else:
        print("--- LSTM Trainer Example ---")
        # This is a placeholder for how you might run training.
        # It requires data to be available (e.g., from data_crawler)
        # For a real run, you'd fetch data first.

        # Example: Create dummy data for training
        sample_ticker = "DUMMY.AX"
        data_points = 200
        dummy_prices = pd.Series(np.cumsum(np.random.randn(data_points)) + 50) # Random walk

        if len(dummy_prices) > DEFAULT_SEQUENCE_LENGTH * 2 :
            print(f"Attempting to train LSTM model for {sample_ticker} with {len(dummy_prices)} data points...")
            success = train_lstm_model(dummy_prices, sample_ticker, epochs=5) # Reduced epochs for quick test
            if success:
                print(f"Training successful for {sample_ticker}. Model and scaler saved.")

                # Test loading and predicting
                model, scaler = load_lstm_model_and_scaler(sample_ticker)
                if model and scaler:
                    print("Model and scaler loaded successfully.")
                    last_sequence_scaled = scaler.transform(dummy_prices.values[-DEFAULT_SEQUENCE_LENGTH:].reshape(-1,1))
                    forecast = predict_with_lstm(model, scaler, last_sequence_scaled, n_steps_forecast=10)
                    if forecast is not None:
                        print(f"Sample forecast for {sample_ticker} (next 10 periods): \n{forecast}")
                    else:
                        print(f"Failed to generate forecast for {sample_ticker}.")
                else:
                    print(f"Failed to load model/scaler for {sample_ticker} for prediction test.")
            else:
                print(f"Training failed for {sample_ticker}.")
        else:
            print(f"Insufficient dummy data to train LSTM for {sample_ticker}.")
    print("--- LSTM Trainer Script Complete ---")
