import click
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import module functions
from .data_crawler import get_historical_prices, get_company_info, save_data_to_csv, save_info_to_json, RAW_DATA_DIR
from .data_cleanser import load_price_data_from_csv, cleanse_price_data, save_cleansed_data, PROCESSED_DATA_DIR
from .analysis_engine import add_technical_indicators, simple_forecast_30_day, get_analysis_summary
from .backtester import run_backtest, simple_sma_rsi_strategy # Assuming default strategy for now
from .recommender import generate_recommendation

from functools import partial # For binding strategy arguments in backtester

# Define output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_period_to_dates(period: str | None, start_date_str: str | None, end_date_str: str | None) -> tuple[str, str] | None:
    """Parses period string or start/end date strings to definitive start and end dates."""
    today = datetime.now()
    if period:
        period = period.lower()
        end_date = today
        if period == "1y":
            start_date = today - timedelta(days=365)
        elif period == "2y":
            start_date = today - timedelta(days=365 * 2)
        elif period == "5y":
            start_date = today - timedelta(days=365 * 5)
        elif period == "ytd":
            start_date = datetime(today.year, 1, 1)
        elif period == "max": # data_crawler handles the 5-year actual limit from yfinance perspective
            start_date = today - timedelta(days=365 * 5) # Max as per project spec
        else:
            click.echo(f"Error: Invalid period string '{period}'. Use '1y', '2y', '5y', 'ytd', 'max'.")
            return None
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif start_date_str and end_date_str:
        try:
            # Validate date formats
            datetime.strptime(start_date_str, "%Y-%m-%d")
            datetime.strptime(end_date_str, "%Y-%m-%d")
            return start_date_str, end_date_str
        except ValueError:
            click.echo("Error: Invalid date format. Please use YYYY-MM-DD.")
            return None
    else: # Default to 1 year if nothing is specified
        click.echo("No period or specific dates provided. Defaulting to 1 year ('1y').")
        end_date = today
        start_date = today - timedelta(days=365)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


@click.command()
@click.option('--ticker', '-t', required=True, type=str, help="Stock ticker (e.g., 'BHP.AX'). Must end with .AX.")
@click.option('--period', '-p', type=str, help="Predefined period (e.g., '1y', '5y', 'ytd', 'max'). Overrides start/end dates if provided.")
@click.option('--start-date', '-s', type=str, help="Start date in YYYY-MM-DD format. Used if --period is not set.")
@click.option('--end-date', '-e', type=str, help="End date in YYYY-MM-DD format. Used if --period is not set.")
@click.option('--interval', '-i', type=click.Choice(['daily', 'weekly']), required=True, help="Data interval: 'daily' or 'weekly'.")
@click.option('--run-backtest', '-b', 'run_backtest_flag', is_flag=True, help="Run backtesting with a default strategy.") # Explicit dest
@click.option('--short-sma', type=int, default=None, help="Short SMA window (default: 50d/10w).")
@click.option('--long-sma', type=int, default=None, help="Long SMA window (default: 200d/40w).")
@click.option('--rsi-window', type=int, default=14, help="RSI window (default: 14).")
@click.option('--initial-capital', type=float, default=10000, help="Initial capital for backtesting.")
def main_cli(ticker: str, period: str | None, start_date: str | None, end_date: str | None,
             interval: str, run_backtest_flag: bool, # Parameter name matches dest
             short_sma: int | None, long_sma: int | None, rsi_window: int, initial_capital: float):
    """
    ASX Trading Analysis and Recommendation System CLI.
    """
    click.echo(f"--- Starting Analysis for {ticker} ---")

    if not ticker.endswith(".AX"):
        click.secho("Error: Ticker must end with '.AX'. Example: 'CBA.AX'", fg="red")
        return

    # Determine interval code for yfinance and filenames
    interval_code = "1d" if interval == "daily" else "1wk"

    # Set default SMA windows based on interval if not provided
    if short_sma is None:
        short_sma = 50 if interval == "daily" else 10
    if long_sma is None:
        long_sma = 200 if interval == "daily" else 40

    click.echo(f"Using SMA windows: Short={short_sma}, Long={long_sma}. RSI window: {rsi_window}")


    # 1. Parse Dates
    date_range = parse_period_to_dates(period, start_date, end_date)
    if not date_range:
        return
    parsed_start_date, parsed_end_date = date_range
    click.echo(f"Data period: {parsed_start_date} to {parsed_end_date}, Interval: {interval}")

    # 2. Data Crawling
    click.echo("\n--- Step 1: Data Crawling ---")
    company_info = get_company_info(ticker)
    if company_info:
        click.echo(f"Successfully fetched company info for {company_info.get('shortName', ticker)}.")
        # Consider saving info if it's a new fetch or significantly different
        # save_info_to_json(company_info, ticker) # Can be made conditional
    else:
        click.secho(f"Could not fetch company info for {ticker}. Proceeding with price data if possible.", fg="yellow")

    historical_prices_df = get_historical_prices(ticker, parsed_start_date, parsed_end_date, interval_code)
    if historical_prices_df is None or historical_prices_df.empty:
        click.secho(f"Failed to fetch historical prices for {ticker}. Exiting.", fg="red")
        return
    raw_file_path = save_data_to_csv(historical_prices_df, ticker, interval_code, "prices")
    if raw_file_path:
        click.echo(f"Raw price data saved to: {raw_file_path}")
    else:
        click.secho("Failed to save raw price data.", fg="red")
        # Decide if we should exit or try to proceed if df is in memory

    # 3. Data Cleansing
    click.echo("\n--- Step 2: Data Cleansing ---")
    # Load from the saved raw file to simulate a full pipeline step
    # Alternatively, could pass historical_prices_df directly if saving is just for record
    loaded_raw_df = load_price_data_from_csv(ticker, interval_code)
    if loaded_raw_df is None:
        click.secho(f"Failed to load raw price data for cleansing for {ticker}. Exiting.", fg="red")
        return

    cleansed_df, cleanse_report = cleanse_price_data(loaded_raw_df, ticker, interval_code)
    if cleansed_df is None or cleansed_df.empty:
        click.secho(f"Data cleansing failed or resulted in empty DataFrame for {ticker}.", fg="red")
        click.echo(f"Cleansing report: {cleanse_report}")
        return
    click.echo(f"Data cleansing successful. Report: {cleanse_report['status']}")
    cleansed_file_path = save_cleansed_data(cleansed_df, ticker, interval_code)
    if cleansed_file_path:
        click.echo(f"Cleansed price data saved to: {cleansed_file_path}")

    # 4. Analysis and Forecasting
    click.echo("\n--- Step 3: Analysis and Forecasting ---")
    df_with_indicators = add_technical_indicators(cleansed_df.copy(), short_sma_window=short_sma, long_sma_window=long_sma, rsi_window=rsi_window)
    if df_with_indicators.empty:
        click.secho("Failed to add technical indicators.", fg="red")
        return

    analysis_summary = get_analysis_summary(df_with_indicators)
    click.echo("Latest Analysis Data:")
    click.echo(f"  Last Close Price: {analysis_summary.get('last_close'):.2f}" if analysis_summary.get('last_close') is not None else "  Last Close Price: N/A")
    click.echo(f"  {analysis_summary.get('short_sma_col', 'Short SMA')}: {analysis_summary.get('short_sma'):.2f}" if analysis_summary.get('short_sma') is not None else f"  {analysis_summary.get('short_sma_col', 'Short SMA')}: N/A")
    click.echo(f"  {analysis_summary.get('long_sma_col', 'Long SMA')}: {analysis_summary.get('long_sma'):.2f}" if analysis_summary.get('long_sma') is not None else f"  {analysis_summary.get('long_sma_col', 'Long SMA')}: N/A")
    click.echo(f"  {analysis_summary.get('rsi_col', 'RSI')}: {analysis_summary.get('rsi'):.2f}" if analysis_summary.get('rsi') is not None else f"  {analysis_summary.get('rsi_col', 'RSI')}: N/A")

    forecast_price, expected_return = simple_forecast_30_day(df_with_indicators, interval_code)
    if forecast_price is not None and expected_return is not None:
        click.echo(f"  30-Day Forecasted Price: {forecast_price:.2f}")
        click.echo(f"  30-Day Expected Return: {expected_return:.2f}%")
    else:
        click.echo("  30-Day Forecast: Could not be generated (likely insufficient data).")

    # 5. Backtesting (Optional)
    backtest_results_data = None
    if run_backtest_flag:
        click.echo("\n--- Step 4: Backtesting ---")
        # Ensure indicator column names used by strategy match those added
        short_sma_col_name = f"SMA_{short_sma}"
        long_sma_col_name = f"SMA_{long_sma}"
        rsi_col_name = f"RSI_{rsi_window}"

        if not all(col in df_with_indicators.columns for col in [short_sma_col_name, long_sma_col_name, rsi_col_name]):
            click.secho("Error: Expected indicator columns for backtesting not found. Check SMA/RSI window parameters.", fg="red")
        else:
            bound_strategy = partial(simple_sma_rsi_strategy,
                                     short_sma_col=short_sma_col_name,
                                     long_sma_col=long_sma_col_name,
                                     rsi_col=rsi_col_name)

            backtest_results_data = run_backtest(
                df_with_indicators.copy(),
                initial_capital=initial_capital,
                strategy_fn=bound_strategy,
                short_sma_col=short_sma_col_name, # These are passed for run_backtest to know, strategy_fn is bound
                long_sma_col=long_sma_col_name,
                rsi_col=rsi_col_name
            )
            if backtest_results_data:
                click.echo("Backtest Results:")
                click.echo(f"  Final Portfolio Value: {backtest_results_data.get('final_portfolio_value'):.2f}")
                click.echo(f"  Total Return: {backtest_results_data.get('total_return_percentage'):.2f}%")
                click.echo(f"  Number of Trades: {backtest_results_data.get('number_of_trades')}")
                click.echo(f"  Win Rate: {backtest_results_data.get('win_rate_percentage'):.2f}%")
            else:
                click.secho("Backtesting did not produce results or failed.", fg="yellow")

    # 6. Recommendation
    click.echo("\n--- Step 5: Recommendation ---")
    recommendation, confidence, full_stats = generate_recommendation(
        analysis_summary, forecast_price, expected_return, backtest_results_data
    )
    click.secho(f"Recommendation: {recommendation}", fg="green" if recommendation == "BUY" else ("red" if recommendation == "SELL" else "yellow"))
    click.secho(f"Confidence: {confidence}", fg="green" if confidence == "High" else ("red" if confidence == "Low" else "yellow"))

    click.echo("\nSupporting Details:")
    for key, val in full_stats.items():
        if key == "reasoning":
            click.echo("  Reasoning:")
            for reason in val: click.echo(f"    - {reason}")
        elif isinstance(val, float):
            click.echo(f"  {key.replace('_', ' ').title()}: {val:.2f}")
        else:
            click.echo(f"  {key.replace('_', ' ').title()}: {val}")


    # 7. Visualization
    click.echo("\n--- Step 6: Generating Visualization ---")
    plot_filename = f"{ticker.replace('.AX', '_AX')}_{interval_code}_analysis_plot.png"
    plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)

    plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Price
    ax1.plot(df_with_indicators['Date'], df_with_indicators['Close'], label='Close Price', color='blue', alpha=0.8)
    if analysis_summary.get('short_sma_col') and analysis_summary.get('short_sma_col') in df_with_indicators:
        ax1.plot(df_with_indicators['Date'], df_with_indicators[analysis_summary['short_sma_col']], label=analysis_summary['short_sma_col'], color='orange', linestyle='--')
    if analysis_summary.get('long_sma_col') and analysis_summary.get('long_sma_col') in df_with_indicators:
        ax1.plot(df_with_indicators['Date'], df_with_indicators[analysis_summary['long_sma_col']], label=analysis_summary['long_sma_col'], color='purple', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    plt.title(f"Price and Indicator Analysis for {ticker} ({interval})")

    # RSI on a separate y-axis if available
    if analysis_summary.get('rsi_col') and analysis_summary.get('rsi_col') in df_with_indicators:
        ax2 = ax1.twinx()
        ax2.plot(df_with_indicators['Date'], df_with_indicators[analysis_summary['rsi_col']], label=analysis_summary['rsi_col'], color='green', alpha=0.5)
        ax2.set_ylabel('RSI', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.axhline(70, color='red', linestyle=':', alpha=0.5, label='RSI Overbought (70)')
        ax2.axhline(30, color='red', linestyle=':', alpha=0.5, label='RSI Oversold (30)')
        ax2.legend(loc='lower left')

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate() # Auto-formats the x-axis labels for dates

    try:
        plt.savefig(plot_filepath)
        click.echo(f"Analysis plot saved to: {plot_filepath}")
    except Exception as e:
        click.secho(f"Error saving plot: {e}", fg="red")
    plt.close()

    click.echo(f"\n--- Analysis for {ticker} Complete ---")


if __name__ == '__main__':
    main_cli()
