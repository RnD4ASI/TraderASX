import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_recommendation(
    analysis_summary: Dict[str, Any],
    forecast_price: float | None,
    expected_return_pct: float | None,
    backtest_results: Dict[str, Any] | None = None # Optional backtest results
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Generates a trading recommendation based on analysis, forecast, and optional backtest results.

    Args:
        analysis_summary (Dict[str, Any]): From analysis_engine.get_analysis_summary().
            Expected keys: 'last_close', 'short_sma', 'long_sma', 'rsi',
                           'short_sma_col', 'long_sma_col', 'rsi_col'.
        forecast_price (float | None): The 30-day forecasted price.
        expected_return_pct (float | None): The 30-day expected return percentage.
        backtest_results (Dict[str, Any] | None): Optional results from backtester.py.

    Returns:
        Tuple[str, str, Dict[str, Any]]:
            - Recommendation string ("BUY", "HOLD", "SELL")
            - Confidence level string ("High", "Medium", "Low")
            - Supporting statistics dictionary
    """
    recommendation = "HOLD" # Default
    confidence = "Low"    # Default
    stats = {
        "last_close": analysis_summary.get('last_close'),
        "short_sma": analysis_summary.get('short_sma'),
        "long_sma": analysis_summary.get('long_sma'),
        "rsi": analysis_summary.get('rsi'),
        "short_sma_col": analysis_summary.get('short_sma_col'),
        "long_sma_col": analysis_summary.get('long_sma_col'),
        "rsi_col": analysis_summary.get('rsi_col'),
        "forecasted_price_30d": forecast_price,
        "expected_return_30d_pct": expected_return_pct,
        "reasoning": []
    }

    # --- Input Validation & Condition Checks ---
    lc = stats["last_close"]
    ssma = stats["short_sma"]
    lsma = stats["long_sma"]
    rsi = stats["rsi"]
    exp_ret = stats["expected_return_30d_pct"]

    if lc is None or pd.isna(lc):
        stats["reasoning"].append("Critical error: Last close price is missing.")
        return "ERROR", "None", stats # Cannot make recommendation

    # Check if essential indicator values are present
    bullish_signals = 0
    bearish_signals = 0
    neutral_signals = 0

    # --- SMA Analysis ---
    if ssma is not None and lsma is not None and not pd.isna(ssma) and not pd.isna(lsma):
        if ssma > lsma and lc > ssma: # Golden cross confirmation / price above short SMA
            bullish_signals += 1
            stats["reasoning"].append(f"Bullish: {stats['short_sma_col']} ({ssma:.2f}) is above {stats['long_sma_col']} ({lsma:.2f}) and price ({lc:.2f}) is above {stats['short_sma_col']}.")
        elif ssma < lsma and lc < ssma: # Death cross confirmation / price below short SMA
            bearish_signals += 1
            stats["reasoning"].append(f"Bearish: {stats['short_sma_col']} ({ssma:.2f}) is below {stats['long_sma_col']} ({lsma:.2f}) and price ({lc:.2f}) is below {stats['short_sma_col']}.")
        else:
            neutral_signals += 1
            stats["reasoning"].append(f"Neutral SMA: {stats['short_sma_col']} ({ssma:.2f}), {stats['long_sma_col']} ({lsma:.2f}), Price ({lc:.2f}). No clear crossover or price confirmation.")
    else:
        neutral_signals += 1
        stats["reasoning"].append("Neutral SMA: Insufficient data for full SMA analysis.")

    # --- RSI Analysis ---
    if rsi is not None and not pd.isna(rsi):
        if rsi > 70:
            # Overbought, could be bearish for new entry, or signal to take profit if already holding.
            # For a new BUY decision, this is usually a cautious sign.
            bearish_signals += 0.5 # Less strong than SMA death cross for a SELL, but caution for BUY
            stats["reasoning"].append(f"Caution (RSI): RSI ({rsi:.2f}) is overbought (>70).")
        elif rsi < 30:
            # Oversold, could be bullish for contrarian entry.
            bullish_signals += 0.5 # Less strong than SMA golden cross
            stats["reasoning"].append(f"Potential (RSI): RSI ({rsi:.2f}) is oversold (<30).")
        elif rsi > 50 and rsi <=70 : # Momentum is up
             bullish_signals += 0.5
             stats["reasoning"].append(f"Positive Momentum (RSI): RSI ({rsi:.2f}) is between 50 and 70.")
        elif rsi < 50 and rsi >=30: # Momentum is down
             bearish_signals += 0.5
             stats["reasoning"].append(f"Negative Momentum (RSI): RSI ({rsi:.2f}) is between 30 and 50.")
        else: # RSI is neutral (exactly 50)
            neutral_signals += 1
            stats["reasoning"].append(f"Neutral RSI: RSI ({rsi:.2f}) is neutral.")
    else:
        neutral_signals += 1
        stats["reasoning"].append("Neutral RSI: Insufficient data for RSI analysis.")

    # --- Forecast Analysis ---
    if exp_ret is not None and not pd.isna(exp_ret):
        if exp_ret > 5: # Arbitrary threshold for a decent expected return
            bullish_signals += 1
            stats["reasoning"].append(f"Bullish Forecast: Expected 30-day return is {exp_ret:.2f}%.")
        elif exp_ret < -5: # Arbitrary threshold for poor expected return
            bearish_signals += 1
            stats["reasoning"].append(f"Bearish Forecast: Expected 30-day return is {exp_ret:.2f}%.")
        else:
            neutral_signals += 1
            stats["reasoning"].append(f"Neutral Forecast: Expected 30-day return ({exp_ret:.2f}%) is marginal.")
    else:
        neutral_signals += 1
        stats["reasoning"].append("Neutral Forecast: Forecast data unavailable.")

    # --- Backtest Consideration (Simple) ---
    # This is a very basic way to incorporate backtest results.
    # A more advanced system might use specific backtest metrics.
    if backtest_results:
        stats["backtest_overall_return_pct"] = backtest_results.get("total_return_percentage")
        stats["backtest_win_rate_pct"] = backtest_results.get("win_rate_percentage")
        if backtest_results.get("total_return_percentage", 0) > 10 and backtest_results.get("win_rate_percentage", 0) > 50:
            bullish_signals += 0.5 # Small bonus if backtest was generally good
            stats["reasoning"].append("Positive Backtest Indication: Overall strategy performance was good.")
        elif backtest_results.get("total_return_percentage", 0) < -5:
            bearish_signals += 0.5
            stats["reasoning"].append("Negative Backtest Indication: Overall strategy performance was poor.")


    # --- Determine Recommendation ---
    if bullish_signals > bearish_signals + 0.5: # Need a clear margin for BUY
        recommendation = "BUY"
        if bullish_signals >= 2.5: # e.g. SMA good, Forecast good, RSI supportive/good backtest
            confidence = "High"
        elif bullish_signals >= 1.5:
            confidence = "Medium"
        else:
            confidence = "Low"

    elif bearish_signals > bullish_signals + 0.5: # Clear margin for SELL
        # For this tool, "SELL" mainly applies if one were hypothetically holding.
        # Or, as a strong signal not to buy.
        recommendation = "SELL" # Could also be "AVOID"
        if bearish_signals >= 2.5:
            confidence = "High"
        elif bearish_signals >= 1.5:
            confidence = "Medium"
        else:
            confidence = "Low"
    else: # Neutral or mixed signals
        recommendation = "HOLD"
        if neutral_signals >= 2 and bullish_signals < 1.5 and bearish_signals < 1.5 : # Many neutral signals and no strong bias
            confidence = "Medium"
        else: # Generally low confidence if signals are very mixed or weak
            confidence = "Low"

    stats["final_recommendation"] = recommendation
    stats["confidence_level"] = confidence
    logging.info(f"Recommendation for {stats.get('ticker', 'N/A')}: {recommendation} ({confidence}). Bull: {bullish_signals}, Bear: {bearish_signals}, Neut: {neutral_signals}")
    return recommendation, confidence, stats


if __name__ == '__main__':
    print("--- Testing Recommender ---")

    # Test Case 1: Strong Buy
    print("\nTest Case 1: Strong Buy")
    summary1 = {
        'last_close': 100.0, 'short_sma': 105.0, 'long_sma': 100.0, 'rsi': 60.0,
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    } # Price above short SMA, short SMA above long SMA
    forecast_p1 = 115.0
    exp_ret1 = 15.0 # Strong positive return
    backtest_res1 = {"total_return_percentage": 20.0, "win_rate_percentage": 60.0}
    rec1, conf1, stats1 = generate_recommendation(summary1, forecast_p1, exp_ret1, backtest_res1)
    print(f"Rec: {rec1}, Conf: {conf1}")
    # for r in stats1["reasoning"]: print(f"  - {r}")
    # print(f"  Stats: {stats1}")


    # Test Case 2: Strong Sell / Avoid
    print("\nTest Case 2: Strong Sell/Avoid")
    summary2 = {
        'last_close': 90.0, 'short_sma': 88.0, 'long_sma': 92.0, 'rsi': 35.0,
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    } # Price below short SMA, short SMA below long SMA
    forecast_p2 = 80.0
    exp_ret2 = -11.11 # Strong negative return
    rec2, conf2, stats2 = generate_recommendation(summary2, forecast_p2, exp_ret2)
    print(f"Rec: {rec2}, Conf: {conf2}")
    # for r in stats2["reasoning"]: print(f"  - {r}")

    # Test Case 3: Hold / Neutral
    print("\nTest Case 3: Hold / Neutral")
    summary3 = {
        'last_close': 100.0, 'short_sma': 99.0, 'long_sma': 101.0, 'rsi': 52.0,
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    } # Mixed SMA
    forecast_p3 = 101.0
    exp_ret3 = 1.0 # Marginal return
    rec3, conf3, stats3 = generate_recommendation(summary3, forecast_p3, exp_ret3)
    print(f"Rec: {rec3}, Conf: {conf3}")
    # for r in stats3["reasoning"]: print(f"  - {r}")

    # Test Case 4: RSI Overbought, but other signals bullish (Buy with caution)
    print("\nTest Case 4: RSI Overbought, but bullish SMA and Forecast")
    summary4 = {
        'last_close': 110.0, 'short_sma': 108.0, 'long_sma': 100.0, 'rsi': 75.0, # RSI overbought
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    }
    forecast_p4 = 120.0
    exp_ret4 = 9.09 # Good return
    rec4, conf4, stats4 = generate_recommendation(summary4, forecast_p4, exp_ret4)
    print(f"Rec: {rec4}, Conf: {conf4}") # Expect BUY but maybe Medium/Low confidence due to RSI
    # for r in stats4["reasoning"]: print(f"  - {r}")

    # Test Case 5: Missing some indicator data
    print("\nTest Case 5: Missing Long SMA")
    summary5 = {
        'last_close': 100.0, 'short_sma': 102.0, 'long_sma': None, 'rsi': 55.0, # Missing long SMA
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    }
    forecast_p5 = 105.0
    exp_ret5 = 5.0
    rec5, conf5, stats5 = generate_recommendation(summary5, forecast_p5, exp_ret5)
    print(f"Rec: {rec5}, Conf: {conf5}")
    # for r in stats5["reasoning"]: print(f"  - {r}")

    # Test Case 6: Missing forecast data
    print("\nTest Case 6: Missing forecast")
    summary6 = {
        'last_close': 100.0, 'short_sma': 102.0, 'long_sma': 98.0, 'rsi': 60.0,
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    }
    forecast_p6 = None
    exp_ret6 = None
    rec6, conf6, stats6 = generate_recommendation(summary6, forecast_p6, exp_ret6)
    print(f"Rec: {rec6}, Conf: {conf6}")

    # Test Case 7: Critical missing last_close
    print("\nTest Case 7: Missing last_close")
    summary7 = {
        'last_close': None, 'short_sma': 102.0, 'long_sma': 98.0, 'rsi': 60.0,
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    }
    forecast_p7 = 100.0
    exp_ret7 = 0.0
    rec7, conf7, stats7 = generate_recommendation(summary7, forecast_p7, exp_ret7)
    print(f"Rec: {rec7}, Conf: {conf7}")
    print(f"  Reason: {stats7['reasoning']}")


    print("\n--- Recommender Module Test Complete ---")
