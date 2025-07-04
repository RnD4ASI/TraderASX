import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from functools import partial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys # Add sys for path manipulation

# Dynamically adjust import paths
# Add the backend directory to sys.path to allow importing modules from backend.src
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    # Assuming backend/src is now accessible via backend_path
    from src import data_crawler, data_cleanser, analysis_engine, backtester, recommender
    from src.main import parse_period_to_dates # This assumes main.py is importable from src
except ImportError as e:
    # Fallback or error if imports still fail
    st.error(f"Failed to import backend modules: {e}. Check backend structure and paths.")
    # Provide a more robust fallback or exit if critical modules fail
    # For now, let's try a direct import assuming backend/src is in PYTHONPATH somehow or streamlit handles it.
    # This part might need refinement based on how Streamlit apps are typically structured with sibling backend folders.
    try:
        # This assumes 'src' (meaning backend.src) is directly on the python path
        # This is less robust than explicitly adding to sys.path.
        from src import data_crawler, data_cleanser, analysis_engine, backtester, recommender
        from src.main import parse_period_to_dates
    except ImportError:
        # If all else fails, this indicates a significant path issue.
        # For development, we might use st.stop() or raise an error.
        # For now, we'll keep the original attempt's structure for the error message.
        import data_crawler, data_cleanser, analysis_engine, backtester, recommender
        from main import parse_period_to_dates


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

dark_theme_css = """
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.css-1d391kg { background-color: #1a1f2b; }
.stTextInput > div > div > input, .stDateInput > div > div > input,
.stSelectbox > div > div, .stNumberInput > div > div > input {
    color: #FAFAFA; background-color: #262730; border-radius: 0.25rem; border: 1px solid #444;
}
.stButton > button { border: 1px solid #FAFAFA; background-color: #262730; color: #FAFAFA; border-radius: 0.25rem; }
.stButton > button:hover { border: 1px solid #0078FF; background-color: #0078FF; color: #FFFFFF; }
.streamlit-expanderHeader { color: #FAFAFA; }
.stMetric > div > div > div { color: #FAFAFA; } .stMetric > label > div > div { color: #FAFAFA; }
table { color: #FAFAFA !important; } th { background-color: #262730 !important; }
.stChatInput > div > div > input { color: #FAFAFA; background-color: #262730; }
</style>
"""

def initialize_session_state():
    defaults = {
        "ticker": "BHP.AX", "period_options": ['1y', '2y', '5y', 'YTD', 'Max (5y)'], "selected_period": "1y",
        "start_date": datetime.now() - timedelta(days=365), "end_date": datetime.now(),
        "interval_options": ['Daily', 'Weekly'], "selected_interval": "Daily",
        "short_sma": 50, "long_sma": 200, "rsi_window": 14,
        "run_backtest": False, "initial_capital": 10000.0,
        "analysis_results": None, "error_message": None, "plot_generated": False,
        "company_info": None, "cleansed_df": None, "df_with_indicators": None,
        "selected_theme": "Light", "chat_messages": [{"role": "assistant", "content": "Hi! How can I help?"}],
        "forecast_model_options": ["Simple Momentum", "ARIMA", "LSTM"], "selected_forecast_model": "Simple Momentum", # Added LSTM
        "arima_seasonal": False,
        "run_garch_forecast": False, "garch_p": 1, "garch_q": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

    if st.session_state.get("short_sma_default_set_for_interval") != st.session_state.selected_interval :
        if st.session_state.selected_interval == "Daily":
            st.session_state.short_sma = 50; st.session_state.long_sma = 200
        else:
            st.session_state.short_sma = 10; st.session_state.long_sma = 40
        st.session_state.rsi_window = 14
        st.session_state.short_sma_default_set_for_interval = st.session_state.selected_interval

def run_analysis_pipeline():
    # ... (clear old results - remains same) ...
    st.session_state.error_message = None; st.session_state.analysis_results = None; st.session_state.plot_generated = False
    st.session_state.company_info = None; st.session_state.cleansed_df = None; st.session_state.df_with_indicators = None
    st.session_state.garch_results = None

    ticker = st.session_state.ticker
    if not ticker or not ticker.endswith(".AX"):
        st.session_state.error_message = "Error: ASX Ticker must be provided and end with '.AX'."
        return

    period_map = {'1y': '1y', '2y': '2y', '5y': '5y', 'YTD': 'ytd', 'Max (5y)': 'max'}
    parsed_dates = parse_period_to_dates(period_map.get(st.session_state.selected_period), None, None) # Simplified date parsing for now

    if not parsed_dates: st.session_state.error_message = "Error: Could not parse date range."; return
    start_date_str, end_date_str = parsed_dates
    interval_code = "1d" if st.session_state.selected_interval == "Daily" else "1wk"
    periods_for_30_days_forecast = 21 if interval_code == "1d" else 4


    with st.spinner(f"Analyzing {ticker}... This may take a moment, especially with advanced models."):
        # --- Data Fetching and Cleansing ---
        st.session_state.company_info = data_crawler.get_company_info(ticker)
        raw_prices_df = data_crawler.get_historical_prices(ticker, start_date_str, end_date_str, interval_code)
        if raw_prices_df is None or raw_prices_df.empty:
            st.session_state.error_message = f"Failed to fetch historical prices for {ticker}."; return

        st.session_state.cleansed_df, cleanse_report = data_cleanser.cleanse_price_data(raw_prices_df, ticker, interval_code)
        if st.session_state.cleansed_df is None or st.session_state.cleansed_df.empty:
            st.session_state.error_message = f"Data cleansing failed. Report: {cleanse_report.get('messages', 'N/A')}"; return

        # --- Technical Indicators ---
        st.session_state.df_with_indicators = analysis_engine.add_technical_indicators(
            st.session_state.cleansed_df.copy(), st.session_state.short_sma, st.session_state.long_sma, st.session_state.rsi_window
        )
        if st.session_state.df_with_indicators.empty:
            st.session_state.error_message = "Failed to add technical indicators."; return

        analysis_summary = analysis_engine.get_analysis_summary(st.session_state.df_with_indicators)

        # --- Price Forecasting ---
        forecast_details = {}
        selected_model_type_lower = st.session_state.selected_forecast_model.lower().replace(" ", "_")

        if selected_model_type_lower == "lstm":
            lstm_forecast_result = analysis_engine.get_lstm_forecast(
                st.session_state.cleansed_df['Close'], # Use cleansed close for LSTM
                n_periods_forecast=periods_for_30_days_forecast,
                ticker_symbol=ticker
            )
            forecast_details['forecast_model_used'] = "LSTM"
            if lstm_forecast_result.get('error'):
                forecast_details['error'] = lstm_forecast_result['error']
                forecast_details['forecasted_price'] = None
                forecast_details['expected_return_pct'] = None
            elif lstm_forecast_result.get('forecasted_values') is not None:
                forecast_details['forecasted_price'] = lstm_forecast_result['forecasted_values'][-1]
                last_close = st.session_state.cleansed_df['Close'].iloc[-1]
                if last_close > 0:
                    forecast_details['expected_return_pct'] = ((forecast_details['forecasted_price'] - last_close) / last_close) * 100
                else:
                    forecast_details['expected_return_pct'] = 0.0
                forecast_details['forecast_sequence'] = lstm_forecast_result['forecasted_values'] # Store full sequence
            else: # Should not happen if error is not set
                forecast_details['error'] = "LSTM forecast returned unexpected empty result."
        else: # Simple Momentum or ARIMA
            forecast_details = analysis_engine.get_forecast_and_return(
                st.session_state.df_with_indicators,
                ticker_symbol=ticker, # Added ticker_symbol argument
                interval=interval_code,
                model_type=selected_model_type_lower,
                seasonal_arima=st.session_state.arima_seasonal if selected_model_type_lower == "arima" else False
            )

        if forecast_details.get('error') and not st.session_state.error_message: # Prioritize other errors first
             st.session_state.error_message = f"Forecast Error ({forecast_details['forecast_model_used']}): {forecast_details['error']}"

        # --- GARCH Volatility Forecast ---
        garch_volatility_forecast = None
        # ... (GARCH logic remains same) ...
        if st.session_state.run_garch_forecast:
            if st.session_state.cleansed_df is not None and not st.session_state.cleansed_df.empty:
                garch_periods = periods_for_30_days_forecast
                garch_volatility_forecast = analysis_engine.get_garch_volatility_forecast(
                    st.session_state.cleansed_df['Close'], n_periods_forecast=garch_periods,
                    p=st.session_state.garch_p, q=st.session_state.garch_q
                ) # ... rest of GARCH error handling
                if garch_volatility_forecast is None and not st.session_state.error_message: # Add warning if no other major error
                    st.warning("GARCH volatility forecast could not be generated.")
            else:
                st.warning("Cannot run GARCH: Cleansed data unavailable.")


        # --- Backtesting ---
        backtest_summary_for_recommender = None
        # ... (Backtesting logic remains same) ...
        if st.session_state.run_backtest:
            short_sma_col_name = f"SMA_{st.session_state.short_sma}"
            long_sma_col_name = f"SMA_{st.session_state.long_sma}"
            rsi_col_name = f"RSI_{st.session_state.rsi_window}"
            if not all(col in st.session_state.df_with_indicators.columns for col in [short_sma_col_name, long_sma_col_name, rsi_col_name]):
                st.warning("Backtest Warning: Indicator columns not found.")
            else:
                bound_strategy = partial(backtester.simple_sma_rsi_strategy, short_sma_col=short_sma_col_name, long_sma_col=long_sma_col_name, rsi_col=rsi_col_name)
                st.session_state.backtest_results_data = backtester.run_backtest(
                    st.session_state.df_with_indicators.copy(), st.session_state.initial_capital, bound_strategy,
                    short_sma_col_name, long_sma_col_name, rsi_col_name
                )
                if st.session_state.backtest_results_data:
                    backtest_summary_for_recommender = {
                        "total_return_percentage": st.session_state.backtest_results_data.get("total_return_percentage"),
                        "win_rate_percentage": st.session_state.backtest_results_data.get("win_rate_percentage")
                    }

        # --- Recommendation ---
        recommendation, confidence, full_stats = recommender.generate_recommendation(
            analysis_summary, forecast_details.get('forecasted_price'), forecast_details.get('expected_return_pct'), backtest_summary_for_recommender
        )

        st.session_state.analysis_results = {
            "company_info": st.session_state.company_info, "analysis_summary": analysis_summary,
            "forecast_details": forecast_details, "garch_volatility_forecast": garch_volatility_forecast,
            "recommendation": recommendation, "confidence": confidence, "supporting_stats": full_stats,
            "full_backtest_results": st.session_state.get("backtest_results_data") # Removed conditional to ensure key exists
        }
        st.session_state.plot_generated = True


def create_interactive_plot(df_indicators, ticker_symbol, analysis_summary, trades_history=None, forecast_sequence=None, forecast_model_name=None):
    # ... (Plotting function largely same, might add forecast sequence display)
    if df_indicators is None or df_indicators.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], specs=[[{}], [{}]])
    fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['Close'], mode='lines', name='Close Price', line=dict(color='blue')), row=1, col=1)
    short_sma_col = analysis_summary.get('short_sma_col')
    if short_sma_col and short_sma_col in df_indicators:
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators[short_sma_col], mode='lines', name=short_sma_col, line=dict(color='orange', dash='dash')), row=1, col=1)
    long_sma_col = analysis_summary.get('long_sma_col')
    if long_sma_col and long_sma_col in df_indicators:
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators[long_sma_col], mode='lines', name=long_sma_col, line=dict(color='purple', dash='dash')), row=1, col=1)

    # Plot forecast sequence if available
    if forecast_sequence is not None and len(forecast_sequence) > 0 and not df_indicators.empty:
        last_date = pd.to_datetime(df_indicators['Date'].iloc[-1])
        # Assuming daily for forecast sequence indexing for now
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast_sequence))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_sequence, mode='lines', name=f'{forecast_model_name} Forecast', line=dict(color='red', dash='dot')), row=1, col=1)

    if trades_history: # Plot trades
        # ... (trade plotting remains same) ...
        buy_dates = [pd.to_datetime(trade['buy_date']) for trade in trades_history if 'buy_date' in trade]
        buy_prices = [trade['buy_price'] for trade in trades_history if 'buy_date' in trade]
        sell_dates = [pd.to_datetime(trade['sell_date']) for trade in trades_history if 'sell_date' in trade and trade.get("status") != "ended_holding"]
        sell_prices = [trade['sell_price'] for trade in trades_history if 'sell_date' in trade and trade.get("status") != "ended_holding"]
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    rsi_col = analysis_summary.get('rsi_col') # RSI plot remains same
    if rsi_col and rsi_col in df_indicators:
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators[rsi_col], mode='lines', name=rsi_col, line=dict(color='green')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="Oversold (30)", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0,100], row=2, col=1)
    fig.update_layout(title_text=f"{ticker_symbol} Analysis", xaxis_rangeslider_visible=True, height=700, legend_title_text="Indicators") # Enabled rangeslider
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig

def main():
    # ... (initial setup and sidebar remains mostly same, ensure LSTM option is in forecast_model_options)
    st.set_page_config(page_title="ASX Stock Analyzer", layout="wide", initial_sidebar_state="expanded")
    initialize_session_state() # Ensure "LSTM" is in forecast_model_options here
    if st.session_state.selected_theme == "Dark": st.markdown(dark_theme_css, unsafe_allow_html=True)
    st.title("ASX Trading Analysis and Recommendation System")

    # --- Sidebar ---
    st.sidebar.header("⚙️ Analysis Configuration")
    st.sidebar.markdown("---"); theme_options = ["Light", "Dark"]; current_theme_idx = theme_options.index(st.session_state.selected_theme)
    st.session_state.selected_theme = st.sidebar.selectbox("Display Theme", theme_options, index=current_theme_idx, help="Select UI theme.")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Stock and Period")
    st.session_state.ticker = st.sidebar.text_input("ASX Ticker", st.session_state.ticker, help="E.g., 'BHP.AX'. Must end with '.AX'.").upper()
    st.session_state.selected_period = st.sidebar.selectbox("Analysis Period", st.session_state.period_options, index=st.session_state.period_options.index(st.session_state.selected_period))
    prev_interval = st.session_state.selected_interval
    st.session_state.selected_interval = st.sidebar.selectbox("Data Interval", st.session_state.interval_options, index=st.session_state.interval_options.index(st.session_state.selected_interval))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Forecast & Volatility Models")
    # Ensure "LSTM" is in st.session_state.forecast_model_options from initialize_session_state
    st.session_state.selected_forecast_model = st.sidebar.selectbox(
        "Price Forecast Model", st.session_state.forecast_model_options,
        index=st.session_state.forecast_model_options.index(st.session_state.selected_forecast_model)
    )
    if st.session_state.selected_forecast_model == "ARIMA":
        st.session_state.arima_seasonal = st.sidebar.checkbox("ARIMA: Use Seasonality", value=st.session_state.arima_seasonal)
    if st.session_state.selected_forecast_model == "LSTM":
        st.sidebar.caption("LSTM uses pre-trained models. Ensure a model for the ticker is available.")

    st.session_state.run_garch_forecast = st.sidebar.checkbox("Run GARCH Volatility Forecast", value=st.session_state.run_garch_forecast)
    # ... (GARCH p,q inputs remain same)
    if st.session_state.run_garch_forecast:
        garch_cols = st.sidebar.columns(2)
        st.session_state.garch_p = garch_cols[0].number_input("GARCH(p)", 0, 5, st.session_state.garch_p, 1)
        st.session_state.garch_q = garch_cols[1].number_input("GARCH(q)", 0, 5, st.session_state.garch_q, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Technical Analysis Parameters")
    # ... (TA params reset logic and inputs remain same)
    if st.session_state.selected_interval != prev_interval or st.sidebar.button("Reset TA Params to Defaults", help="Resets SMA/RSI to interval defaults."):
        if st.session_state.selected_interval == "Daily": st.session_state.short_sma, st.session_state.long_sma = 50, 200
        else: st.session_state.short_sma, st.session_state.long_sma = 10, 40
        st.session_state.rsi_window = 14
        st.session_state.short_sma_default_set_for_interval = st.session_state.selected_interval
        st.experimental_rerun()
    st.session_state.short_sma = st.sidebar.number_input("Short SMA Window", 5, 100, st.session_state.short_sma, 1)
    st.session_state.long_sma = st.sidebar.number_input("Long SMA Window", 10, 300, st.session_state.long_sma, 1)
    st.session_state.rsi_window = st.sidebar.number_input("RSI Window", 5, 30, st.session_state.rsi_window, 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtesting (Optional)")
    # ... (Backtesting inputs remain same)
    st.session_state.run_backtest = st.sidebar.checkbox("Run Backtest", st.session_state.run_backtest)
    if st.session_state.run_backtest:
        st.session_state.initial_capital = st.sidebar.number_input("Initial Capital", 100.0, value=st.session_state.initial_capital, step=100.0)

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Analyze Stock", type="primary", use_container_width=True): run_analysis_pipeline()

    # --- Main Results Area ---
    st.header("Analysis Results")
    if st.session_state.error_message: st.error(st.session_state.error_message)
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        # ... (Company info display) ...
        if res["company_info"]:
            with st.expander("Company Information", expanded=True):
                st.subheader(res["company_info"].get('shortName', st.session_state.ticker))
                st.caption(f"Sector: {res['company_info'].get('sector', 'N/A')} | Exchange: {res['company_info'].get('exchange', 'N/A')} | Currency: {res['company_info'].get('currency', 'N/A')}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Market Cap", f"{res['company_info'].get('marketCap', 0)/1e9:.2f}B" if res['company_info'].get('marketCap') else "N/A")
                col2.metric("Trailing P/E", f"{res['company_info'].get('trailingPE'):.2f}" if res['company_info'].get('trailingPE') else "N/A")
                col3.metric("Forward P/E", f"{res['company_info'].get('forwardPE'):.2f}" if res['company_info'].get('forwardPE') else "N/A")
                st.metric("Dividend Yield", f"{res['company_info'].get('dividendYield', 0)*100:.2f}%" if res['company_info'].get('dividendYield') else "N/A")
                st.markdown("**Business Summary:**"); st.write(res["company_info"].get('longBusinessSummary', 'Not available.'))

        st.subheader("Recommendation & Forecast")
        # ... (Recommendation display) ...
        rec_color = "green" if res["recommendation"] == "BUY" else ("red" if res["recommendation"] == "SELL" else "orange")
        st.markdown(f"""<div style="border:2px solid {rec_color}; padding:10px; border-radius:5px;"><h3 style="color:{rec_color};">{res["recommendation"]} (Confidence: {res["confidence"]})</h3></div>""", unsafe_allow_html=True)

        f_details = res["forecast_details"]
        st.markdown(f"**Price Forecast Model Used:** {f_details['forecast_model_used'].replace('_', ' ').title()}")
        if f_details.get('arima_order'): st.caption(f"ARIMA Order: {f_details['arima_order']}")
        if f_details.get('forecast_model_used', '').lower() == 'lstm' and f_details.get('error'):
             st.warning(f"LSTM Note: {f_details['error']}") # Show LSTM specific errors prominently

        cols_forecast = st.columns(2)
        cols_forecast[0].metric("30-Day Forecasted Price", f"{f_details.get('forecasted_price'):.2f}" if f_details.get('forecasted_price') is not None else "N/A")
        cols_forecast[1].metric("30-Day Expected Return", f"{f_details.get('expected_return_pct'):.2f}%" if f_details.get('expected_return_pct') is not None else "N/A")
        if f_details.get('arima_conf_int') and f_details['forecast_model_used'] == 'arima':
            st.caption(f"ARIMA 95% CI for Forecast Point: [{f_details['arima_conf_int'][0]:.2f} - {f_details['arima_conf_int'][1]:.2f}]")

        # Display GARCH ...
        if res.get("garch_volatility_forecast") is not None:
            with st.expander("GARCH Volatility Forecast", expanded=False):
                st.write("Forecasted Conditional Standard Deviation (next periods):")
                garch_std_dev = np.sqrt(res["garch_volatility_forecast"])
                st.line_chart(garch_std_dev, use_container_width=True)
                st.caption("Chart shows forecasted conditional standard deviation (volatility).")

        st.markdown("---"); st.subheader("Technical Analysis Summary") # TA Summary ...
        cols_metrics = st.columns(4); ana_sum = res["analysis_summary"]
        cols_metrics[0].metric("Last Close", f"{ana_sum.get('last_close'):.2f}" if ana_sum.get('last_close') else "N/A")
        cols_metrics[1].metric(ana_sum.get('short_sma_col','S SMA'), f"{ana_sum.get('short_sma'):.2f}" if ana_sum.get('short_sma') else "N/A")
        cols_metrics[2].metric(ana_sum.get('long_sma_col','L SMA'), f"{ana_sum.get('long_sma'):.2f}" if ana_sum.get('long_sma') else "N/A")
        cols_metrics[3].metric(ana_sum.get('rsi_col','RSI'), f"{ana_sum.get('rsi'):.2f}" if ana_sum.get('rsi') else "N/A")

        with st.expander("Detailed Reasoning for Recommendation"): # Reasoning ...
            if res["supporting_stats"] and "reasoning" in res["supporting_stats"]:
                for reason in res["supporting_stats"]["reasoning"]: st.write(f"- {reason}")

        if st.session_state.plot_generated and st.session_state.df_with_indicators is not None: # Main Plot ...
            st.subheader("Price Chart & Indicators")
            trades_hist = res.get("full_backtest_results", {}).get("trades_history") if st.session_state.run_backtest else None
            forecast_seq_to_plot = res["forecast_details"].get('forecast_sequence') if res["forecast_details"].get('forecast_model_used', '').lower() == 'lstm' else None
            model_name_for_plot = res["forecast_details"].get('forecast_model_used', '').replace('_', ' ').title()

            interactive_fig = create_interactive_plot(
                st.session_state.df_with_indicators, st.session_state.ticker,
                res["analysis_summary"], trades_hist,
                forecast_sequence=forecast_seq_to_plot, forecast_model_name=model_name_for_plot
            )
            st.plotly_chart(interactive_fig, use_container_width=True)

        if st.session_state.run_backtest and res.get("full_backtest_results"): # Backtest details ...
            # ... (backtest display remains same)
            bt_res = res["full_backtest_results"]
            with st.expander("Backtest Results Details", expanded=False):
                st.subheader("Strategy Performance")
                bt_cols = st.columns(3)
                bt_cols[0].metric("Final Portfolio Value", f"${bt_res.get('final_portfolio_value'):,.2f}")
                bt_cols[1].metric("Total Return", f"{bt_res.get('total_return_percentage'):.2f}%")
                bt_cols[2].metric("Win Rate", f"{bt_res.get('win_rate_percentage'):.2f}%")
                st.metric("Number of Trades", f"{bt_res.get('number_of_trades')}")
                if "trades_history" in bt_res and bt_res["trades_history"]:
                    st.markdown("**Trades History:**"); st.dataframe(pd.DataFrame(bt_res["trades_history"]))
    else:
        st.info("Configure analysis in the sidebar and click '🚀 Analyze Stock'.")

    # --- Chatbot Stub ---
    # ... (Chatbot UI remains same) ...
    st.sidebar.markdown("---")
    with st.sidebar.expander("ASX Helper Chatbot (Beta)", expanded=False):
        st.markdown("_Ask about basic trading concepts or ASX terms._")
        if "chat_messages" not in st.session_state: st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! How can I help?"}]
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]): st.write(message["content"])
        predefined_qa = {
            "what is sma": "A Simple Moving Average (SMA) smooths out price data to identify trends by calculating the average price over a specific period.",
            "what is rsi": "The Relative Strength Index (RSI) is a momentum oscillator measuring price movement speed and change (0-100). >70 often means overbought, <30 oversold.",
            "what is asx": "The Australian Securities Exchange (ASX) is Australia's main stock exchange.",
            "what is garch": "GARCH models volatility, suggesting high volatility often follows high, and low follows low.",
            "what is lstm": "LSTM (Long Short-Term Memory) is a type of neural network good for learning from sequences of data, like time series prices. It can capture complex patterns."
        }
        user_chat_input = st.chat_input("Ask a question...")
        if user_chat_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_chat_input});
            with st.chat_message("user"): st.write(user_chat_input)
            response = "Sorry, I have limited knowledge. Try 'What is LSTM?'" # Updated default
            for qk, ans in predefined_qa.items():
                if all(k.lower() in user_chat_input.lower() for k in qk.split()): response = ans; break
                elif any(k.lower() in user_chat_input.lower() for k in qk.split() if len(k)>2):
                    if any(ik in qk.split() for ik in user_chat_input.lower().split()): response = ans; break
            with st.chat_message("assistant"): st.write(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            if len(st.session_state.chat_messages) > 10: st.session_state.chat_messages = st.session_state.chat_messages[-10:]


if __name__ == '__main__':
    main()
