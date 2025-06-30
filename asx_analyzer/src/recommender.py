import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_recommendation(
    analysis_summary: Dict[str, Any],
    forecast_details: Dict[str, Any], # Contains price, return, model type, and model-specifics
    garch_vol_forecast: Optional[np.ndarray] = None, # Array of forecasted conditional std devs (not variances)
    backtest_results: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Generates a trading recommendation.
    """
    recommendation = "HOLD"
    confidence = "Low"

    # Extract primary forecast values
    forecast_price = forecast_details.get('forecasted_price')
    expected_return_pct = forecast_details.get('expected_return_pct')
    forecast_model_used = forecast_details.get('forecast_model_used', 'N/A')

    stats = {
        "last_close": analysis_summary.get('last_close'),
        "short_sma": analysis_summary.get('short_sma'),
        "long_sma": analysis_summary.get('long_sma'),
        "rsi": analysis_summary.get('rsi'),
        "short_sma_col": analysis_summary.get('short_sma_col'),
        "long_sma_col": analysis_summary.get('long_sma_col'),
        "rsi_col": analysis_summary.get('rsi_col'),
        "forecast_model_used": forecast_model_used,
        "forecasted_price_30d": forecast_price,
        "expected_return_30d_pct": expected_return_pct,
        "arima_order": forecast_details.get('arima_order') if forecast_model_used == 'arima' else None,
        "arima_conf_int_30d": forecast_details.get('arima_conf_int') if forecast_model_used == 'arima' else None, # CI for the specific forecast point
        "garch_forecasted_avg_std_dev": np.mean(garch_vol_forecast) if garch_vol_forecast is not None else None,
        "reasoning": [],
        "confidence_factors": {"base_bullish": 0, "base_bearish": 0, "base_neutral": 0, "confidence_adjustment": 0}
    }

    lc = stats["last_close"]
    ssma = stats["short_sma"]
    lsma = stats["long_sma"]
    rsi = stats["rsi"]
    exp_ret = stats["expected_return_30d_pct"]

    if lc is None or pd.isna(lc):
        stats["reasoning"].append("Critical error: Last close price is missing.")
        return "ERROR", "None", stats

    bull = stats["confidence_factors"]["base_bullish"]
    bear = stats["confidence_factors"]["base_bearish"]
    neut = stats["confidence_factors"]["base_neutral"]

    # --- SMA Analysis ---
    if ssma is not None and lsma is not None and not pd.isna(ssma) and not pd.isna(lsma):
        price_above_short = lc > ssma if not pd.isna(lc) else False
        price_above_long = lc > lsma if not pd.isna(lc) else False
        short_above_long = ssma > lsma

        if short_above_long and price_above_short: # Golden cross territory & price confirmation
            bull += 1
            stats["reasoning"].append(f"Bullish SMA: {stats['short_sma_col']} ({ssma:.2f}) > {stats['long_sma_col']} ({lsma:.2f}) and Price ({lc:.2f}) > {stats['short_sma_col']}.")
        elif not short_above_long and not price_above_short : # Death cross territory & price confirmation
            bear += 1
            stats["reasoning"].append(f"Bearish SMA: {stats['short_sma_col']} ({ssma:.2f}) < {stats['long_sma_col']} ({lsma:.2f}) and Price ({lc:.2f}) < {stats['short_sma_col']}.")
        else: # Mixed signals from SMAs / price position relative to them
            neut += 1
            stats["reasoning"].append(f"Neutral/Mixed SMA: Short ({ssma:.2f}), Long ({lsma:.2f}), Price ({lc:.2f}).")
    else:
        neut += 1; stats["reasoning"].append("Neutral SMA: Insufficient data for full SMA analysis.")

    # --- RSI Analysis ---
    if rsi is not None and not pd.isna(rsi):
        if rsi > 70: bear += 0.5; stats["reasoning"].append(f"Caution (RSI): Overbought ({rsi:.2f} > 70).")
        elif rsi < 30: bull += 0.5; stats["reasoning"].append(f"Potential (RSI): Oversold ({rsi:.2f} < 30).")
        elif rsi > 55: bull += 0.25; stats["reasoning"].append(f"Positive Momentum (RSI): ({rsi:.2f}).") # Slight bullish tilt above 55
        elif rsi < 45: bear += 0.25; stats["reasoning"].append(f"Negative Momentum (RSI): ({rsi:.2f}).") # Slight bearish tilt below 45
        else: neut += 1; stats["reasoning"].append(f"Neutral RSI: ({rsi:.2f}).")
    else:
        neut += 1; stats["reasoning"].append("Neutral RSI: Insufficient data.")

    # --- Forecast Analysis (Main Price Forecast) ---
    if exp_ret is not None and not pd.isna(exp_ret) and forecast_price is not None:
        if exp_ret > 7.5: # Increased threshold for stronger signal
            bull += 1.5
            stats["reasoning"].append(f"Strong Bullish Forecast ({forecast_model_used}): Exp. return {exp_ret:.2f}%.")
        elif exp_ret > 2.5:
            bull += 0.75
            stats["reasoning"].append(f"Modest Bullish Forecast ({forecast_model_used}): Exp. return {exp_ret:.2f}%.")
        elif exp_ret < -7.5:
            bear += 1.5
            stats["reasoning"].append(f"Strong Bearish Forecast ({forecast_model_used}): Exp. return {exp_ret:.2f}%.")
        elif exp_ret < -2.5:
            bear += 0.75
            stats["reasoning"].append(f"Modest Bearish Forecast ({forecast_model_used}): Exp. return {exp_ret:.2f}%.")
        else:
            neut += 1
            stats["reasoning"].append(f"Neutral Forecast ({forecast_model_used}): Exp. return ({exp_ret:.2f}%) is marginal.")

        # ARIMA Confidence Interval Check (if applicable)
        if forecast_model_used == 'arima' and stats["arima_conf_int_30d"] and forecast_price > 0:
            ci_lower, ci_upper = stats["arima_conf_int_30d"]
            ci_width_pct = ((ci_upper - ci_lower) / forecast_price) * 100 if forecast_price != 0 else float('inf')
            stats["arima_ci_width_pct_30d"] = ci_width_pct
            if ci_width_pct > 30: # If CI is very wide (e.g., >30% of forecast price)
                stats["reasoning"].append(f"High Uncertainty (ARIMA): Wide 95% CI ({ci_width_pct:.1f}%) for 30d forecast.")
                stats["confidence_factors"]["confidence_adjustment"] -= 0.5 # Reduce confidence score
            elif ci_width_pct < 10: # If CI is narrow
                 stats["reasoning"].append(f"Higher Certainty (ARIMA): Narrow 95% CI ({ci_width_pct:.1f}%) for 30d forecast.")
                 stats["confidence_factors"]["confidence_adjustment"] += 0.25


    else: # Forecast data N/A
        neut += 1; stats["reasoning"].append("Neutral Forecast: Forecast data unavailable or invalid.")

    # --- GARCH Volatility Consideration ---
    if stats["garch_forecasted_avg_std_dev"] is not None:
        # Assuming garch_vol_forecast is daily std dev. Annualize for context (approx * sqrt(252))
        # This threshold is arbitrary and needs calibration.
        annualized_garch_std_dev = stats["garch_forecasted_avg_std_dev"] * np.sqrt(252/100) # /100 because returns were scaled by 100
        stats["garch_annualized_std_dev_pct"] = annualized_garch_std_dev
        if annualized_garch_std_dev > 40: # e.g. >40% annualized volatility is high
            stats["reasoning"].append(f"High Volatility Warning (GARCH): Forecasted avg. annualized std dev is {annualized_garch_std_dev:.2f}%.")
            stats["confidence_factors"]["confidence_adjustment"] -= 0.75 # Significant confidence reduction
        elif annualized_garch_std_dev < 15: # Low vol
            stats["reasoning"].append(f"Low Volatility Indication (GARCH): Forecasted avg. annualized std dev is {annualized_garch_std_dev:.2f}%.")
            stats["confidence_factors"]["confidence_adjustment"] += 0.25


    # --- Backtest Consideration ---
    if backtest_results:
        stats["backtest_overall_return_pct"] = backtest_results.get("total_return_percentage")
        stats["backtest_win_rate_pct"] = backtest_results.get("win_rate_percentage")
        if backtest_results.get("total_return_percentage", 0) > 10 and backtest_results.get("win_rate_percentage", 0) > 50:
            bull += 0.5; stats["reasoning"].append("Supportive Backtest: Strategy showed positive historical performance.")
        elif backtest_results.get("total_return_percentage", 0) < -5:
            bear += 0.5; stats["reasoning"].append("Cautionary Backtest: Strategy showed poor historical performance.")

    stats["confidence_factors"]["base_bullish"] = bull
    stats["confidence_factors"]["base_bearish"] = bear
    stats["confidence_factors"]["base_neutral"] = neut

    # --- Determine Recommendation ---
    net_signal_score = bull - bear

    if net_signal_score >= 1.5: recommendation = "BUY"
    elif net_signal_score <= -1.5: recommendation = "SELL"
    else: recommendation = "HOLD"

    # --- Determine Confidence ---
    total_signals_strength = abs(net_signal_score) + stats["confidence_factors"]["confidence_adjustment"]

    if total_signals_strength >= 2.5 : confidence = "High"
    elif total_signals_strength >= 1.25 : confidence = "Medium"
    else: confidence = "Low"

    # If overall signal is weak, but volatility is high, confidence should be low for action signals
    if recommendation != "HOLD" and stats.get("garch_annualized_std_dev_pct", 0) > 40 and confidence == "High":
        confidence = "Medium" # Downgrade due to high vol

    stats["final_recommendation"] = recommendation
    stats["confidence_level"] = confidence
    logging.info(f"Recommender: Final Score {net_signal_score:.2f}, Bull: {bull:.2f}, Bear: {bear:.2f}, Neut: {neut:.2f}, Conf Adj: {stats['confidence_factors']['confidence_adjustment']:.2f} => {recommendation} ({confidence})")
    return recommendation, confidence, stats


if __name__ == '__main__':
    print("--- Testing Recommender with Advanced Model Inputs ---")

    # Base summary
    analysis_summary_base = {
        'last_close': 100.0, 'short_sma': 102.0, 'long_sma': 98.0, 'rsi': 60.0, # Bullish TA
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    }

    # Test Case 1: Strong Buy with ARIMA (narrow CI) and Low GARCH
    print("\nTest Case 1: Strong Buy, ARIMA (Narrow CI), Low GARCH")
    forecast_details_1 = {
        'forecast_model_used': 'arima', 'forecasted_price': 115.0, 'expected_return_pct': 15.0,
        'arima_order': '(1,1,1)', 'arima_conf_int': [112.0, 118.0] # Narrow CI
    }
    garch_low_1 = np.array([0.1, 0.12, 0.11]) * (100/np.sqrt(252)) # Low daily std dev (scaled for internal calc)
    rec1, conf1, stats1 = generate_recommendation(analysis_summary_base, forecast_details_1, garch_vol_forecast=garch_low_1)
    print(f"Rec: {rec1}, Conf: {conf1}")
    print(f"  Reasoning: {stats1['reasoning']}")
    print(f"  Conf Factors: {stats1['confidence_factors']}")


    # Test Case 2: Buy, but ARIMA (wide CI) and High GARCH
    print("\nTest Case 2: Buy, ARIMA (Wide CI), High GARCH")
    forecast_details_2 = {
        'forecast_model_used': 'arima', 'forecasted_price': 110.0, 'expected_return_pct': 10.0,
        'arima_order': '(1,1,1)', 'arima_conf_int': [95.0, 125.0] # Wide CI (30 / 110 ~ 27%)
    }
    # High daily std dev (e.g. 2% daily -> ~31% annualized)
    garch_high_2 = np.array([2.5, 2.6, 2.4]) * (100/np.sqrt(252)) # High daily std dev (scaled for internal calc)
    rec2, conf2, stats2 = generate_recommendation(analysis_summary_base, forecast_details_2, garch_vol_forecast=garch_high_2)
    print(f"Rec: {rec2}, Conf: {conf2}") # Expect confidence to be lowered
    print(f"  Reasoning: {stats2['reasoning']}")
    print(f"  Conf Factors: {stats2['confidence_factors']}")


    # Test Case 3: LSTM forecast, neutral other signals
    print("\nTest Case 3: LSTM forecast, Neutral TA, No GARCH")
    analysis_summary_neutral_ta = {
        'last_close': 100.0, 'short_sma': 100.0, 'long_sma': 100.0, 'rsi': 50.0,
        'short_sma_col': 'SMA_10', 'long_sma_col': 'SMA_20', 'rsi_col': 'RSI_14'
    }
    forecast_details_3 = {
        'forecast_model_used': 'lstm', 'forecasted_price': 108.0, 'expected_return_pct': 8.0,
    }
    rec3, conf3, stats3 = generate_recommendation(analysis_summary_neutral_ta, forecast_details_3, garch_vol_forecast=None)
    print(f"Rec: {rec3}, Conf: {conf3}")
    print(f"  Reasoning: {stats3['reasoning']}")
    print(f"  Conf Factors: {stats3['confidence_factors']}")

    print("\n--- Recommender Module Test Complete ---")
