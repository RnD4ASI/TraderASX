import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from typing import Dict, List, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple strategy type for clarity
# A strategy function will take a row of data (with indicators) and current holdings,
# and return an action: "BUY", "SELL", "HOLD"
StrategyFunction = Callable[[pd.Series, bool], str] # (row, is_holding_stock) -> action

def simple_sma_rsi_strategy(row: pd.Series, is_holding_stock: bool, short_sma_col: str, long_sma_col: str, rsi_col: str) -> str:
    """
    A simple trading strategy based on SMA crossover and RSI.
    Assumes a 30-day holding period is managed by the backtester, not this function.
    This function decides on INITIATING a buy or INITIATING a sell (if holding).
    """
    if pd.isna(row[short_sma_col]) or pd.isna(row[long_sma_col]) or pd.isna(row[rsi_col]):
        return "HOLD" # Not enough data for indicators

    # Buy signal: Short SMA crosses above Long SMA, and RSI is not overbought (e.g., < 70)
    # and not currently holding stock.
    if not is_holding_stock:
        if row[short_sma_col] > row[long_sma_col] and row[rsi_col] < 70:
            # Check if previous period was below crossover to confirm crossover event
            # This requires looking at previous row, which is complex to pass here.
            # For simplicity, we'll assume any point where short > long is a potential buy if not holding.
            # A more robust strategy would check for the actual crossover moment.
            return "BUY"

    # Sell signal: (Only if holding) Short SMA crosses below Long SMA OR RSI is very overbought (e.g. > 80)
    # The actual sale is determined by the 30-day hold rule in the backtester.
    # This signal suggests a desire to sell IF the holding period allows.
    if is_holding_stock:
        if row[short_sma_col] < row[long_sma_col] or row[rsi_col] > 80: # Example sell condition
            return "SELL_SIGNAL" # Not an actual "SELL" action, but a signal that we might want to sell.
                                # The backtester will decide based on holding period.
    return "HOLD"


def run_backtest(
    df_with_indicators: pd.DataFrame,
    initial_capital: float,
    strategy_fn: StrategyFunction, # This will be a partial function with strategy params bound
    short_sma_col: str, # Name of the short SMA column
    long_sma_col: str,  # Name of the long SMA column
    rsi_col: str        # Name of the RSI column
) -> Dict:
    """
    Runs a backtest on the given data using the provided strategy.

    Args:
        df_with_indicators (pd.DataFrame): DataFrame with 'Date', 'Close', and indicator columns.
                                           Must be sorted by Date.
        initial_capital (float): The starting capital for the backtest.
        strategy_fn (StrategyFunction): A function that implements the trading logic.
        short_sma_col, long_sma_col, rsi_col: Column names for indicators.

    Returns:
        Dict: A dictionary containing backtest performance metrics.
    """
    if df_with_indicators is None or df_with_indicators.empty or 'Close' not in df_with_indicators.columns or 'Date' not in df_with_indicators.columns:
        logging.error("Backtest input DataFrame is invalid.")
        return {}

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_with_indicators['Date']):
        try:
            df_with_indicators['Date'] = pd.to_datetime(df_with_indicators['Date'])
        except Exception as e:
            logging.error(f"Could not convert 'Date' column to datetime in backtester: {e}")
            return {}

    df = df_with_indicators.sort_values(by='Date').reset_index(drop=True)

    capital = initial_capital
    shares_held = 0
    buy_price = 0.0
    buy_date = None
    sell_by_date = None # Date by which we must sell due to 30-day rule (or can choose to sell after)

    trades = [] # List to store trade details: (buy_date, buy_price, sell_date, sell_price, shares, profit)
    portfolio_value_over_time = [] # Store (date, portfolio_value)

    HOLDING_PERIOD_DAYS = 30 # Calendar days

    for i, row in df.iterrows():
        current_date = row['Date']
        current_price = row['Close']
        portfolio_value = capital + (shares_held * current_price)
        portfolio_value_over_time.append({'Date': current_date, 'PortfolioValue': portfolio_value})

        if pd.isna(current_price):
            logging.debug(f"Skipping row {i} due to NaN price on {current_date}")
            continue

        action = "HOLD" # Default action

        # Decision to SELL (if holding and conditions met)
        if shares_held > 0:
            # Check if mandatory holding period is over
            if sell_by_date is not None and current_date >= sell_by_date:
                # Holding period is over, now check strategy for sell signal OR just sell if it was a timed exit
                # For this strategy, we use the strategy_fn to see if it's a good time to sell
                strategy_decision = strategy_fn(row, True) # strategy_fn is already bound with col names
                if strategy_decision == "SELL_SIGNAL": # Strategy suggests selling
                    action = "SELL"
                # If strategy doesn't say SELL_SIGNAL, we continue holding if desired,
                # as the 30-day mandatory period is just a minimum.
                # A more complex strategy could have profit targets/stop losses here.
            else:
                # Still within mandatory holding period
                action = "HOLD"

        # Decision to BUY (if not holding and conditions met)
        if shares_held == 0:
            strategy_decision = strategy_fn(row, False) # strategy_fn is already bound with col names
            if strategy_decision == "BUY":
                action = "BUY"

        # Execute actions
        if action == "BUY" and capital > 0:
            shares_to_buy = capital / current_price # Buy with all available capital
            shares_held = shares_to_buy
            buy_price = current_price
            buy_date = current_date
            capital = 0 # All capital used for shares
            sell_by_date = current_date + timedelta(days=HOLDING_PERIOD_DAYS)
            logging.info(f"{current_date}: BUY {shares_held:.2f} shares at {current_price:.2f}")

        elif action == "SELL" and shares_held > 0:
            capital = shares_held * current_price
            profit = (current_price - buy_price) * shares_held
            trades.append({
                "buy_date": buy_date, "buy_price": buy_price,
                "sell_date": current_date, "sell_price": current_price,
                "shares": shares_held, "profit": profit,
                "return_pct": ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            })
            logging.info(f"{current_date}: SELL {shares_held:.2f} shares at {current_price:.2f}, Profit: {profit:.2f}")
            shares_held = 0
            buy_price = 0.0
            buy_date = None
            sell_by_date = None

    # Final portfolio value if still holding shares at the end
    final_portfolio_value = capital
    if shares_held > 0 and not df.empty:
        last_price = df['Close'].iloc[-1]
        if not pd.isna(last_price):
            final_portfolio_value += shares_held * last_price
            # Log a "virtual" final trade if still holding for performance calculation
            profit = (last_price - buy_price) * shares_held
            trades.append({
                "buy_date": buy_date, "buy_price": buy_price,
                "sell_date": df['Date'].iloc[-1], "sell_price": last_price, # Mark as end of period
                "shares": shares_held, "profit": profit, "status": "ended_holding",
                "return_pct": ((last_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            })
            logging.info(f"End of backtest: Still holding {shares_held:.2f} shares. Valued at {last_price:.2f}. Virtual Profit: {profit:.2f}")


    # Calculate metrics
    total_return_abs = final_portfolio_value - initial_capital
    total_return_pct = (total_return_abs / initial_capital) * 100 if initial_capital > 0 else 0.0
    num_trades = len([t for t in trades if t.get("status") != "ended_holding"]) # Count only completed trades

    profitable_trades = [t for t in trades if t["profit"] > 0 and t.get("status") != "ended_holding"]
    num_profitable_trades = len(profitable_trades)
    win_rate = (num_profitable_trades / num_trades) * 100 if num_trades > 0 else 0.0

    avg_gain_pct = np.mean([t['return_pct'] for t in profitable_trades]) if num_profitable_trades > 0 else 0.0
    avg_loss_pct = np.mean([t['return_pct'] for t in trades if t["profit"] <= 0 and t.get("status") != "ended_holding" and num_trades > 0]) if (num_trades - num_profitable_trades) > 0 else 0.0


    return {
        "initial_capital": initial_capital,
        "final_portfolio_value": final_portfolio_value,
        "total_return_absolute": total_return_abs,
        "total_return_percentage": total_return_pct,
        "number_of_trades": num_trades,
        "number_of_profitable_trades": num_profitable_trades,
        "win_rate_percentage": win_rate,
        "average_gain_percentage": avg_gain_pct,
        "average_loss_percentage": avg_loss_pct,
        "trades_history": trades,
        "portfolio_value_over_time": pd.DataFrame(portfolio_value_over_time) if portfolio_value_over_time else pd.DataFrame()
    }

if __name__ == '__main__':
    from analysis_engine import add_technical_indicators # Import for testing
    from functools import partial

    # Create dummy data for testing
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='B')) # Business days
    close_prices = np.linspace(20, 40, 50).tolist() + np.linspace(40, 30, 50).tolist() # Up then down trend
    close_prices = [p + np.random.randn()*0.5 for p in close_prices] # Add some noise

    dummy_df = pd.DataFrame({'Date': dates, 'Close': close_prices, 'Open': close_prices, 'High': close_prices, 'Low': close_prices, 'Volume': 1000})

    print("--- Testing Backtester ---")
    # Add indicators (using smaller windows for this test data)
    short_sma, long_sma, rsi_w = 10, 20, 7
    df_indic = add_technical_indicators(dummy_df.copy(), short_sma_window=short_sma, long_sma_window=long_sma, rsi_window=rsi_w)

    # Define column names based on how add_technical_indicators names them
    short_sma_col_name = f"SMA_{short_sma}"
    long_sma_col_name = f"SMA_{long_sma}"
    rsi_col_name = f"RSI_{rsi_w}"

    # Ensure these columns exist
    if not all(col in df_indic.columns for col in [short_sma_col_name, long_sma_col_name, rsi_col_name]):
        print("Error: Indicator columns not found in DataFrame. Check naming in analysis_engine.")
        print(f"Expected: {short_sma_col_name}, {long_sma_col_name}, {rsi_col_name}")
        print(f"Available: {df_indic.columns}")
    else:
        # Use partial to bind the strategy's indicator column name arguments
        bound_strategy = partial(simple_sma_rsi_strategy,
                                 short_sma_col=short_sma_col_name,
                                 long_sma_col=long_sma_col_name,
                                 rsi_col=rsi_col_name)

        backtest_results = run_backtest(df_indic, initial_capital=10000, strategy_fn=bound_strategy,
                                        short_sma_col=short_sma_col_name, # Pass these again for clarity, though strategy_fn is bound
                                        long_sma_col=long_sma_col_name,
                                        rsi_col=rsi_col_name)

        print("\nBacktest Results:")
        if backtest_results:
            for key, value in backtest_results.items():
                if key == "trades_history":
                    print(f"  {key}: (count: {len(value)})")
                    # for trade in value: print(f"    {trade}")
                elif key == "portfolio_value_over_time":
                     print(f"  {key}: (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
        else:
            print("Backtest did not produce results.")

    print("\n--- Backtester Module Test Complete ---")
