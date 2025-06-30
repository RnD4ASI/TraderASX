import streamlit as st

def main():
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from functools import partial # Correctly import partial

# Dynamically adjust import paths if running as a module or script
try:
    from src import data_crawler, data_cleanser, analysis_engine, backtester, recommender
    from src.main import parse_period_to_dates # Re-use date parsing from CLI
except ImportError:
    # This allows running streamlit run app.py from the asx_analyzer directory
    import data_crawler, data_cleanser, analysis_engine, backtester, recommender
    from main import parse_period_to_dates


# Define output directory for plots (though Streamlit handles plots in-app)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output") # If app.py is in asx_analyzer/
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "ticker": "BHP.AX",
        "period_options": ['1y', '2y', '5y', 'YTD', 'Max (5y)'],
        "selected_period": "1y",
        "start_date": datetime.now() - timedelta(days=365),
        "end_date": datetime.now(),
        "interval_options": ['Daily', 'Weekly'],
        "selected_interval": "Daily",
        "short_sma": None, # Will be set based on interval
        "long_sma": None,  # Will be set based on interval
        "rsi_window": 14,
        "run_backtest": False,
        "initial_capital": 10000.0,
        "analysis_results": None,
        "error_message": None,
        "plot_generated": False,
        "company_info": None,
        "cleansed_df": None,
        "df_with_indicators": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Set dynamic SMA defaults based on interval
    if st.session_state.short_sma is None:
        st.session_state.short_sma = 50 if st.session_state.selected_interval == "Daily" else 10
    if st.session_state.long_sma is None:
        st.session_state.long_sma = 200 if st.session_state.selected_interval == "Daily" else 40


def run_analysis_pipeline():
    """
    Orchestrates the full analysis pipeline based on current session_state inputs.
    Updates session_state with results or error messages.
    """
    st.session_state.error_message = None
    st.session_state.analysis_results = None
    st.session_state.plot_generated = False
    st.session_state.company_info = None
    st.session_state.cleansed_df = None
    st.session_state.df_with_indicators = None


    ticker = st.session_state.ticker
    if not ticker or not ticker.endswith(".AX"):
        st.session_state.error_message = "Error: ASX Ticker must be provided and end with '.AX'."
        return

    # Date parsing
    period_map = { # Map display period to CLI style period for parse_period_to_dates
        '1y': '1y', '2y': '2y', '5y': '5y',
        'YTD': 'ytd', 'Max (5y)': 'max'
    }

    # Use custom date range if 'Custom' is selected or if period is not driving dates
    # For now, period drives dates, custom range selection can be added later
    parsed_dates = parse_period_to_dates(
        period_map.get(st.session_state.selected_period),
        st.session_state.start_date.strftime("%Y-%m-%d") if isinstance(st.session_state.start_date, datetime) else None, # Simplified for now
        st.session_state.end_date.strftime("%Y-%m-%d") if isinstance(st.session_state.end_date, datetime) else None
    )

    if not parsed_dates:
        st.session_state.error_message = "Error: Could not parse date range."
        return

    start_date_str, end_date_str = parsed_dates
    interval_code = "1d" if st.session_state.selected_interval == "Daily" else "1wk"

    short_sma = st.session_state.short_sma
    long_sma = st.session_state.long_sma
    rsi_window = st.session_state.rsi_window

    with st.spinner(f"Analyzing {ticker}..."):
        # 1. Data Crawling
        st.session_state.company_info = data_crawler.get_company_info(ticker)
        raw_prices_df = data_crawler.get_historical_prices(ticker, start_date_str, end_date_str, interval_code)

        if raw_prices_df is None or raw_prices_df.empty:
            st.session_state.error_message = f"Failed to fetch historical prices for {ticker}."
            return
        # Not saving to CSV in web app, directly pass dataframe

        # 2. Data Cleansing
        st.session_state.cleansed_df, cleanse_report = data_cleanser.cleanse_price_data(raw_prices_df, ticker, interval_code)
        if st.session_state.cleansed_df is None or st.session_state.cleansed_df.empty:
            st.session_state.error_message = f"Data cleansing failed for {ticker}. Report: {cleanse_report.get('messages', 'No specific message')}"
            return

        # 3. Analysis and Forecasting
        st.session_state.df_with_indicators = analysis_engine.add_technical_indicators(
            st.session_state.cleansed_df.copy(),
            short_sma_window=short_sma,
            long_sma_window=long_sma,
            rsi_window=rsi_window
        )
        if st.session_state.df_with_indicators.empty:
            st.session_state.error_message = "Failed to add technical indicators."
            return

        analysis_summary = analysis_engine.get_analysis_summary(st.session_state.df_with_indicators)
        forecast_price, expected_return = analysis_engine.simple_forecast_30_day(st.session_state.df_with_indicators, interval_code)

        # 4. Backtesting (Optional)
        backtest_summary_for_recommender = None
        if st.session_state.run_backtest:
            short_sma_col_name = f"SMA_{short_sma}"
            long_sma_col_name = f"SMA_{long_sma}"
            rsi_col_name = f"RSI_{rsi_window}"

            if not all(col in st.session_state.df_with_indicators.columns for col in [short_sma_col_name, long_sma_col_name, rsi_col_name]):
                st.session_state.error_message = "Backtest Error: Expected indicator columns not found. Check SMA/RSI windows."
            else:
                bound_strategy = partial(
                    backtester.simple_sma_rsi_strategy,
                    short_sma_col=short_sma_col_name,
                    long_sma_col=long_sma_col_name,
                    rsi_col=rsi_col_name
                )
                backtest_results_data = backtester.run_backtest(
                    st.session_state.df_with_indicators.copy(),
                    initial_capital=st.session_state.initial_capital,
                    strategy_fn=bound_strategy,
                    short_sma_col=short_sma_col_name,
                    long_sma_col=long_sma_col_name,
                    rsi_col=rsi_col_name
                )
                # Store full backtest results if needed for detailed display later
                st.session_state.backtest_results_data = backtest_results_data
                # For recommender, a smaller summary might be enough
                if backtest_results_data:
                    backtest_summary_for_recommender = {
                        "total_return_percentage": backtest_results_data.get("total_return_percentage"),
                        "win_rate_percentage": backtest_results_data.get("win_rate_percentage")
                    }


        # 5. Recommendation
        recommendation, confidence, full_stats = recommender.generate_recommendation(
            analysis_summary, forecast_price, expected_return, backtest_summary_for_recommender
        )

        st.session_state.analysis_results = {
            "company_info": st.session_state.company_info,
            "analysis_summary": analysis_summary,
            "forecast_price": forecast_price,
            "expected_return": expected_return,
            "recommendation": recommendation,
            "confidence": confidence,
            "supporting_stats": full_stats, # This contains reasoning too
            "backtest_summary": backtest_summary_for_recommender, # Only summary sent to recommender
            "full_backtest_results": st.session_state.get("backtest_results_data") if st.session_state.run_backtest else None
        }
        st.session_state.plot_generated = True # We will generate plot when displaying


def main():
    st.set_page_config(page_title="ASX Stock Analyzer", layout="wide", initial_sidebar_state="expanded")

    initialize_session_state() # Initialize first

    # --- Define Dark Theme CSS ---
    # Basic dark theme. More comprehensive styling can be added.
    dark_theme_css = """
    <style>
    /* General background and text */
    .stApp {
        background-color: #0E1117; /* Streamlit's default dark bg */
        color: #FAFAFA; /* Streamlit's default dark text */
    }
    /* Sidebar */
    .css-1d391kg { /* Specific class for sidebar, might change with versions */
        background-color: #1a1f2b;
    }
    /* Input widgets, buttons in dark mode */
    .stTextInput > div > div > input, .stDateInput > div > div > input, .stSelectbox > div > div {
        color: #FAFAFA;
        background-color: #262730;
    }
    .stButton > button {
        border: 1px solid #FAFAFA;
        background-color: #262730;
        color: #FAFAFA;
    }
    .stButton > button:hover {
        border: 1px solid #0078FF;
        background-color: #0078FF;
        color: #FFFFFF;
    }
    /* Expander headers */
    .streamlit-expanderHeader {
        color: #FAFAFA;
    }
    /* Metrics */
    .stMetric > div > div > div {
         color: #FAFAFA; /* Metric label */
    }
    .stMetric > label > div > div {
        color: #FAFAFA; /* Metric value */
    }
    /* Markdown generated tables, etc. */
    table {
        color: #FAFAFA !important;
    }
    th {
        background-color: #262730 !important;
    }
    </style>
    """

    # Apply theme based on session state
    if st.session_state.get("selected_theme", "Light") == "Dark":
        st.markdown(dark_theme_css, unsafe_allow_html=True)
    # Else, Streamlit's default light theme is used.

    st.title("ASX Trading Analysis and Recommendation System")

    # --- Sidebar for inputs ---
    st.sidebar.header("User Inputs")

    # Theme Selector
    current_theme_index = 0 # Default to Light
    if st.session_state.get("selected_theme") == "Dark":
        current_theme_index = 1

    theme_options = ["Light", "Dark"]
    st.session_state.selected_theme = st.sidebar.selectbox(
        "Select Theme",
        options=theme_options,
        index=current_theme_index,
        key="theme_selector" # Key to ensure it updates session state properly
    )
    # Note: Changing theme via selectbox will trigger a rerun.
    # The CSS will be applied at the top of the script on rerun.

    st.sidebar.markdown("---") # Separator
    st.session_state.ticker = st.sidebar.text_input("ASX Ticker (e.g., BHP.AX)", value=st.session_state.ticker).upper()

    # Date and Interval Selection
    st.session_state.selected_period = st.sidebar.selectbox("Period", options=st.session_state.period_options, index=st.session_state.period_options.index(st.session_state.selected_period))
    # TODO: Add custom date pickers if needed, for now period drives dates
    # st.session_state.start_date = st.sidebar.date_input("Start Date", value=st.session_state.start_date)
    # st.session_state.end_date = st.sidebar.date_input("End Date", value=st.session_state.end_date)

    prev_interval = st.session_state.selected_interval
    st.session_state.selected_interval = st.sidebar.selectbox("Interval", options=st.session_state.interval_options, index=st.session_state.interval_options.index(st.session_state.selected_interval))

    # Update SMA defaults if interval changes
    if st.session_state.selected_interval != prev_interval or st.sidebar.button("Reset SMA/RSI to defaults"):
        if st.session_state.selected_interval == "Daily":
            st.session_state.short_sma = 50
            st.session_state.long_sma = 200
        else: # Weekly
            st.session_state.short_sma = 10
            st.session_state.long_sma = 40
        st.session_state.rsi_window = 14
        # This will trigger a rerun, updating the input fields below.

    st.sidebar.subheader("Technical Analysis Parameters")
    st.session_state.short_sma = st.sidebar.number_input("Short SMA Window", min_value=5, max_value=100, value=st.session_state.short_sma, step=1)
    st.session_state.long_sma = st.sidebar.number_input("Long SMA Window", min_value=10, max_value=300, value=st.session_state.long_sma, step=1)
    st.session_state.rsi_window = st.sidebar.number_input("RSI Window", min_value=5, max_value=30, value=st.session_state.rsi_window, step=1)

    st.sidebar.subheader("Backtesting (Optional)")
    st.session_state.run_backtest = st.sidebar.checkbox("Run Backtest", value=st.session_state.run_backtest)
    if st.session_state.run_backtest:
        st.session_state.initial_capital = st.sidebar.number_input("Initial Capital", min_value=100.0, value=st.session_state.initial_capital, step=100.0)

    if st.sidebar.button("Analyze Stock", type="primary", use_container_width=True):
        run_analysis_pipeline()

    # --- Main area for results ---
    st.header("Analysis Results")

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        # Company Info
        if results["company_info"]:
            with st.expander("Company Information", expanded=True):
                st.subheader(results["company_info"].get('shortName', st.session_state.ticker))
                st.caption(f"Sector: {results['company_info'].get('sector', 'N/A')} | Exchange: {results['company_info'].get('exchange', 'N/A')} | Currency: {results['company_info'].get('currency', 'N/A')}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Market Cap", f"{results['company_info'].get('marketCap', 0)/1e9:.2f}B" if results['company_info'].get('marketCap') else "N/A")
                col2.metric("Trailing P/E", f"{results['company_info'].get('trailingPE'):.2f}" if results['company_info'].get('trailingPE') else "N/A")
                col3.metric("Forward P/E", f"{results['company_info'].get('forwardPE'):.2f}" if results['company_info'].get('forwardPE') else "N/A")
                st.metric("Dividend Yield", f"{results['company_info'].get('dividendYield', 0)*100:.2f}%" if results['company_info'].get('dividendYield') else "N/A")

                st.markdown("**Business Summary:**")
                st.write(results["company_info"].get('longBusinessSummary', 'Not available.'))
        else:
            st.info(f"No detailed company information found for {st.session_state.ticker}.")

        # Recommendation and Key Stats
        st.subheader("Recommendation")
        rec_color = "green" if results["recommendation"] == "BUY" else ("red" if results["recommendation"] == "SELL" else "orange")
        st.markdown(f"""
        <div style="border:2px solid {rec_color}; padding: 10px; border-radius: 5px;">
            <h3 style="color:{rec_color};">{results["recommendation"]} (Confidence: {results["confidence"]})</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        cols_metrics = st.columns(4)
        ana_sum = results["analysis_summary"]
        cols_metrics[0].metric("Last Close", f"{ana_sum.get('last_close'):.2f}" if ana_sum.get('last_close') else "N/A")
        cols_metrics[1].metric(ana_sum.get('short_sma_col', 'Short SMA'), f"{ana_sum.get('short_sma'):.2f}" if ana_sum.get('short_sma') else "N/A")
        cols_metrics[2].metric(ana_sum.get('long_sma_col', 'Long SMA'), f"{ana_sum.get('long_sma'):.2f}" if ana_sum.get('long_sma') else "N/A")
        cols_metrics[3].metric(ana_sum.get('rsi_col', 'RSI'), f"{ana_sum.get('rsi'):.2f}" if ana_sum.get('rsi') else "N/A")

        st.metric("30-Day Forecasted Price", f"{results.get('forecast_price'):.2f}" if results.get('forecast_price') else "N/A")
        st.metric("30-Day Expected Return", f"{results.get('expected_return'):.2f}%" if results.get('expected_return') else "N/A")

        with st.expander("Detailed Reasoning"):
            if results["supporting_stats"] and "reasoning" in results["supporting_stats"]:
                for reason in results["supporting_stats"]["reasoning"]:
                    st.write(f"- {reason}")
            else:
                st.write("No detailed reasoning available.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_plot(df_indicators: pd.DataFrame, ticker_symbol: str, analysis_summary: dict, trades_history: list = None):
    """
    Creates an interactive Plotly chart with price, SMAs, RSI, and optional trade markers.
    """
    if df_indicators is None or df_indicators.empty:
        return go.Figure() # Return empty figure if no data

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.7, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    # Price and SMAs on the first subplot
    fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['Close'], mode='lines', name='Close Price', line=dict(color='blue')), row=1, col=1)

    short_sma_col = analysis_summary.get('short_sma_col')
    if short_sma_col and short_sma_col in df_indicators:
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators[short_sma_col], mode='lines', name=short_sma_col, line=dict(color='orange', dash='dash')), row=1, col=1)

    long_sma_col = analysis_summary.get('long_sma_col')
    if long_sma_col and long_sma_col in df_indicators:
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators[long_sma_col], mode='lines', name=long_sma_col, line=dict(color='purple', dash='dash')), row=1, col=1)

    # Add trade markers if available
    if trades_history:
        buy_dates = [trade['buy_date'] for trade in trades_history if 'buy_date' in trade]
        buy_prices = [trade['buy_price'] for trade in trades_history if 'buy_date' in trade] # Use actual buy price for marker position
        sell_dates = [trade['sell_date'] for trade in trades_history if 'sell_date' in trade and trade.get("status") != "ended_holding"]
        sell_prices = [trade['sell_price'] for trade in trades_history if 'sell_date' in trade and trade.get("status") != "ended_holding"]

        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signal',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signal',
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)

    # RSI on the second subplot
    rsi_col = analysis_summary.get('rsi_col')
    if rsi_col and rsi_col in df_indicators:
        fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators[rsi_col], mode='lines', name=rsi_col, line=dict(color='green')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="bottom right", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="Oversold (30)", annotation_position="bottom right", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0,100], row=2, col=1)

    fig.update_layout(
        title_text=f"{ticker_symbol} Analysis",
        xaxis_rangeslider_visible=False,
        legend_title_text="Indicators",
        height=600 # Adjust height as needed
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


        # Placeholder for plot - will be implemented in next step
        if st.session_state.plot_generated and st.session_state.df_with_indicators is not None:
            st.subheader("Price Chart & Indicators")
            # Generate and display the Plotly chart
            trades_history_for_plot = None
            if st.session_state.run_backtest and results.get("full_backtest_results"):
                trades_history_for_plot = results["full_backtest_results"].get("trades_history")

            interactive_fig = create_interactive_plot(
                st.session_state.df_with_indicators,
                st.session_state.ticker,
                results["analysis_summary"],
                trades_history_for_plot
            )
            st.plotly_chart(interactive_fig, use_container_width=True)


        # Backtest Results Display
        if st.session_state.run_backtest and results.get("full_backtest_results"):
            bt_res = results["full_backtest_results"]
            with st.expander("Backtest Results", expanded=False):
                st.subheader("Strategy Performance")
                bt_cols = st.columns(3)
                bt_cols[0].metric("Final Portfolio Value", f"${bt_res.get('final_portfolio_value'):,.2f}")
                bt_cols[1].metric("Total Return", f"{bt_res.get('total_return_percentage'):.2f}%")
                bt_cols[2].metric("Win Rate", f"{bt_res.get('win_rate_percentage'):.2f}%")
                st.metric("Number of Trades", f"{bt_res.get('number_of_trades')}")

                if "trades_history" in bt_res and bt_res["trades_history"]:
                    st.markdown("**Trades History:**")
                    # Convert trades history to DataFrame for better display
                    trades_df = pd.DataFrame(bt_res["trades_history"])
                    st.dataframe(trades_df)

                # Placeholder for backtest plot (portfolio value over time)
                # if not bt_res.get("portfolio_value_over_time", pd.DataFrame()).empty:
                #     st.line_chart(bt_res["portfolio_value_over_time"].set_index("Date")["PortfolioValue"])


    else:
        st.info("Enter parameters in the sidebar and click 'Analyze Stock'.")

    # --- Chatbot Stub ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("ASX Helper Chatbot (Beta)", expanded=False):
        st.markdown("_Ask about basic trading concepts or ASX terms._")

        # Initialize chat history in session state if it doesn't exist
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! How can I help you with basic ASX or trading questions today?"}]

        # Display prior messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Predefined Q&A
        predefined_qa = {
            "what is sma": "A Simple Moving Average (SMA) is an arithmetic moving average calculated by adding recent closing prices and then dividing that by the number of time periods in the calculation average. It helps smooth out price data to identify trends.",
            "what is rsi": "The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. RSI values range from 0 to 100. Typically, RSI above 70 is considered overbought, and below 30 is considered oversold.",
            "what is asx": "The Australian Securities Exchange (ASX) is Australia's primary securities exchange. It's where shares of Australian public companies, derivatives, and other financial products are traded.",
            "what is a stock": "A stock (also known as equity) represents a share in the ownership of a company, which includes a claim on the company's assets and earnings.",
            "what is a ticker": "A ticker symbol is a unique series of letters assigned to a security for trading purposes. For example, 'CBA.AX' is the ticker for Commonwealth Bank on the ASX.",
            "buy signal": "In this app, a BUY signal generally suggests that technical indicators and forecasts align to indicate a potential upward price movement, making it a candidate for purchase (considering the 30-day hold).",
            "sell signal": "In this app, a SELL signal suggests indicators point towards a potential downward price movement or that an existing position might be optimal to close (after the 30-day hold).",
            "hold signal": "A HOLD signal means the indicators and forecast don't show a strong conviction for either buying or selling at the current moment."
        }

        user_chat_input = st.chat_input("Ask a question...")

        if user_chat_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_chat_input})
            with st.chat_message("user"):
                st.write(user_chat_input)

            response = "Sorry, I can only answer a few predefined questions currently. Try asking 'What is SMA?' or 'What is ASX?'"
            for question_key, answer in predefined_qa.items():
                # Simple keyword matching
                if all(keyword.lower() in user_chat_input.lower() for keyword in question_key.split()):
                    response = answer
                    break
                # Fallback for single keyword from question key
                elif any(keyword.lower() in user_chat_input.lower() for keyword in question_key.split() if len(keyword)>2):
                    # check if any word in user input is in question key
                    input_keywords = user_chat_input.lower().split()
                    if any(input_word in question_key.split() for input_word in input_keywords):
                        response = answer
                        break


            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            # Limit chat history length
            if len(st.session_state.chat_messages) > 20:
                st.session_state.chat_messages = st.session_state.chat_messages[-20:]


if __name__ == '__main__':
    main()
