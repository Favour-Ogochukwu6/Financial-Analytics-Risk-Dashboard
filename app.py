import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import norm


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Financial Analytics & Risk Dashboard",
    layout="wide"
)

st.title("Financial Analytics & Risk Dashboard")
st.write(
    "Monitor stock performance, portfolio returns "
    "and financial risk metrics."
)
st.divider()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_close_prices(data):
    """Extract closing prices from yfinance data."""
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data.xs("Close", axis=1, level=0)
        elif "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        close = data[["Close"]].copy()
        close.columns = ["Stock"]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.apply(pd.to_numeric, errors="coerce")
    return close.dropna(axis=1, how="all")


def calculate_var_and_es(returns):
    """Calculate daily VaR and Expected Shortfall using three methods."""
    returns = returns.dropna()

    if returns.empty:
        return None

    mean_return = returns.mean()
    daily_std = returns.std()

    # Historical VaR and Expected Shortfall
    historical_var_95 = -returns.quantile(0.05)
    historical_var_99 = -returns.quantile(0.01)

    tail_95 = returns[returns <= returns.quantile(0.05)]
    tail_99 = returns[returns <= returns.quantile(0.01)]

    historical_es_95 = (
        -tail_95.mean() if not tail_95.empty else historical_var_95
    )
    historical_es_99 = (
        -tail_99.mean() if not tail_99.empty else historical_var_99
    )

    # Parametric method (normal distribution)
    if pd.notna(daily_std) and daily_std > 0:
        parametric_var_95 = -(mean_return - norm.ppf(0.95) * daily_std)
        parametric_var_99 = -(mean_return - norm.ppf(0.99) * daily_std)

        parametric_es_95 = (
            -mean_return
            + daily_std * norm.pdf(norm.ppf(0.05)) / 0.05
        )
        parametric_es_99 = (
            -mean_return
            + daily_std * norm.pdf(norm.ppf(0.01)) / 0.01
        )
    else:
        parametric_var_95 = -mean_return
        parametric_var_99 = -mean_return
        parametric_es_95 = -mean_return
        parametric_es_99 = -mean_return

    # Monte Carlo method
    rng = np.random.default_rng(42)
    simulated_returns = rng.normal(
        loc=mean_return,
        scale=0 if pd.isna(daily_std) else daily_std,
        size=10000
    )

    monte_carlo_var_95 = -np.percentile(simulated_returns, 5)
    monte_carlo_var_99 = -np.percentile(simulated_returns, 1)

    simulated_tail_95 = simulated_returns[
        simulated_returns <= np.percentile(simulated_returns, 5)
    ]
    simulated_tail_99 = simulated_returns[
        simulated_returns <= np.percentile(simulated_returns, 1)
    ]

    monte_carlo_es_95 = -simulated_tail_95.mean()
    monte_carlo_es_99 = -simulated_tail_99.mean()

    comparison = pd.DataFrame({
        "Risk Method": ["Historical", "Parametric", "Monte Carlo"],
        "VaR 95%": [
            historical_var_95, parametric_var_95, monte_carlo_var_95
        ],
        "VaR 99%": [
            historical_var_99, parametric_var_99, monte_carlo_var_99
        ],
        "Expected Shortfall 95%": [
            historical_es_95, parametric_es_95, monte_carlo_es_95
        ],
        "Expected Shortfall 99%": [
            historical_es_99, parametric_es_99, monte_carlo_es_99
        ]
    })

    return comparison, historical_var_95


# ==========================================
# STOCK SELECTION
# ==========================================

st.header("Individual Stock Analysis")

stock_options = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
    "Tesla": "TSLA"
}

selected_company = st.selectbox(
    "Select a company:",
    list(stock_options.keys())
)
selected_ticker = stock_options[selected_company]

period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

selected_period = st.selectbox(
    "Select analysis period:",
    list(period_options.keys()),
    index=3
)
download_period = period_options[selected_period]


# ==========================================
# INDIVIDUAL STOCK DATA AND ANALYSIS
# ==========================================

with st.spinner(f"Downloading {selected_company} market data..."):
    stock_data = yf.download(
        selected_ticker,
        period=download_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

if stock_data is not None and not stock_data.empty:
    stock_close_frame = get_close_prices(stock_data)

    if not stock_close_frame.empty:
        close_prices = stock_close_frame.iloc[:, 0].dropna()
        daily_returns = close_prices.pct_change(fill_method=None).dropna()

        if len(close_prices) >= 2 and not daily_returns.empty:
            total_return = close_prices.iloc[-1] / close_prices.iloc[0] - 1
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            stock_var_95 = -daily_returns.quantile(0.05)

            drawdown = close_prices / close_prices.cummax() - 1
            max_drawdown = drawdown.min()

            if daily_returns.std() > 0:
                sharpe_ratio = (
                    daily_returns.mean() / daily_returns.std()
                ) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            st.success(f"{selected_company} data downloaded successfully!")
            st.subheader("Individual Stock Risk Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{total_return:.2%}")
            col2.metric("Annualised Volatility", f"{annualized_volatility:.2%}")
            col3.metric("Historical VaR (95%)", f"{stock_var_95:.2%}")

            col4, col5 = st.columns(2)
            col4.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
            col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            st.subheader("Stock Price History")
            st.line_chart(close_prices.rename("Closing Price"))

            st.subheader("Daily Stock Returns")
            st.line_chart(daily_returns.rename("Daily Return"))

            st.subheader("Latest Stock Market Records")
            st.dataframe(stock_data.tail(10), use_container_width=True)
        else:
            st.warning(
                "There is not enough stock-price history to calculate the metrics."
            )
    else:
        st.warning("The downloaded data does not contain closing prices.")
else:
    st.warning("Unable to download stock data. Please try again later.")


# ==========================================
# MULTI-STOCK PORTFOLIO ANALYSIS
# ==========================================

st.divider()
st.header("Multi-Stock Portfolio Analysis")

portfolio_companies = st.multiselect(
    "Select companies for your portfolio:",
    list(stock_options.keys()),
    default=["Apple", "Microsoft", "NVIDIA"]
)

if len(portfolio_companies) < 2:
    st.info("Please select at least two companies.")
else:
    portfolio_tickers = [
        stock_options[company] for company in portfolio_companies
    ]

    with st.spinner("Downloading portfolio market data..."):
        portfolio_data = yf.download(
            portfolio_tickers,
            period=download_period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

    if portfolio_data is None or portfolio_data.empty:
        st.warning("Unable to download portfolio data.")
    else:
        portfolio_prices = get_close_prices(portfolio_data)
        portfolio_prices = portfolio_prices.dropna(axis=1, how="all")

        matching_tickers = [
            ticker for ticker in portfolio_tickers
            if ticker in portfolio_prices.columns
        ]
        if matching_tickers:
            portfolio_prices = portfolio_prices[matching_tickers]

        available_tickers = list(portfolio_prices.columns)

        if len(available_tickers) < 2:
            st.warning(
                "At least two selected stocks must have valid market data."
            )
        else:
            st.success("Portfolio data downloaded successfully!")

            # ==========================================
            # INVESTMENT AMOUNT AND STOCK WEIGHTS
            # ==========================================

            st.subheader("Investment Allocation")
            investment_amount = st.number_input(
                "Initial investment (€)",
                min_value=100.0,
                value=10000.0,
                step=500.0
            )

            st.write("Set your investment weights (%).")
            weights = {}
            weight_columns = st.columns(len(available_tickers))

            for i, ticker in enumerate(available_tickers):
                with weight_columns[i]:
                    weights[ticker] = st.number_input(
                        ticker,
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0 / len(available_tickers),
                        step=5.0,
                        key=f"weight_{ticker}"
                    )

            total_weight = sum(weights.values())

            if total_weight <= 0:
                st.warning(
                    "Please allocate a weight greater than 0% to at least one stock."
                )
            else:
                normalized_weights = {
                    ticker: weight / total_weight
                    for ticker, weight in weights.items()
                }
                weight_series = pd.Series(normalized_weights)

                st.write(f"Total entered allocation: {total_weight:.2f}%")
                st.write("Actual portfolio allocation:")

                for ticker, weight in normalized_weights.items():
                    st.write(
                        f"{ticker}: {weight:.2%} "
                        f"= €{weight * investment_amount:,.2f}"
                    )

                # ==========================================
                # PORTFOLIO ALLOCATION PIE CHART
                # ==========================================

                st.subheader("Portfolio Allocation")
                allocation_data = pd.DataFrame({
                    "Stock": list(normalized_weights.keys()),
                    "Weight": list(normalized_weights.values())
                })
                allocation_data["Investment (€)"] = (
                    allocation_data["Weight"] * investment_amount
                )

                fig_allocation = px.pie(
                    allocation_data,
                    names="Stock",
                    values="Investment (€)",
                    title="Investment Distribution",
                    hole=0.4
                )
                fig_allocation.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )
                st.plotly_chart(
                    fig_allocation,
                    use_container_width=True
                )

                # ==========================================
                # PORTFOLIO RETURNS
                # ==========================================

                portfolio_returns = portfolio_prices.pct_change(
                    fill_method=None
                ).dropna(how="any")

                portfolio_returns = portfolio_returns[weight_series.index]

                # Assumes daily rebalancing to the target weights.
                daily_portfolio_returns = (
                    portfolio_returns.mul(weight_series, axis=1).sum(axis=1)
                ).dropna()

                if daily_portfolio_returns.empty:
                    st.warning(
                        "Not enough overlapping price data to calculate portfolio returns."
                    )
                else:
                    # ==========================================
                    # STOCK CORRELATION HEATMAP
                    # ==========================================

                    st.subheader("Stock Correlation Heatmap")
                    correlation_matrix = portfolio_returns.corr()

                    fig_correlation = px.imshow(
                        correlation_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        aspect="auto",
                        title="Correlation Between Selected Stocks"
                    )
                    st.plotly_chart(
                        fig_correlation,
                        use_container_width=True
                    )
                    st.write(
                        "Values closer to +1 indicate stocks that move together. "
                        "Values closer to -1 indicate stocks that tend to move in "
                        "opposite directions. Values around 0 indicate a weak "
                        "linear relationship."
                    )

                    # ==========================================
                    # VALUE AT RISK AND EXPECTED SHORTFALL
                    # ==========================================

                    risk_results = calculate_var_and_es(
                        daily_portfolio_returns
                    )

                    # ==========================================
                    # PORTFOLIO PERFORMANCE AND RISK METRICS
                    # ==========================================

                    portfolio_value = (
                        investment_amount
                        * (1 + daily_portfolio_returns).cumprod()
                    )

                    total_portfolio_return = (
                        portfolio_value.iloc[-1] / investment_amount
                    ) - 1

                    portfolio_volatility = (
                        daily_portfolio_returns.std() * np.sqrt(252)
                    )

                    portfolio_drawdown = (
                        portfolio_value / portfolio_value.cummax()
                    ) - 1
                    portfolio_max_drawdown = portfolio_drawdown.min()

                    if daily_portfolio_returns.std() > 0:
                        portfolio_sharpe = (
                            daily_portfolio_returns.mean()
                            / daily_portfolio_returns.std()
                        ) * np.sqrt(252)
                    else:
                        portfolio_sharpe = 0.0

                    st.divider()
                    st.subheader("Portfolio Risk Metrics")

                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Portfolio Value",
                        f"€{portfolio_value.iloc[-1]:,.2f}"
                    )
                    col2.metric(
                        "Total Return",
                        f"{total_portfolio_return:.2%}"
                    )
                    col3.metric(
                        "Annualised Volatility",
                        f"{portfolio_volatility:.2%}"
                    )

                    col4, col5, col6 = st.columns(3)
                    if risk_results is not None:
                        var_comparison, portfolio_historical_var_95 = risk_results
                        col4.metric(
                            "Historical VaR (95%)",
                            f"{portfolio_historical_var_95:.2%}"
                        )
                    else:
                        col4.metric("Historical VaR (95%)", "N/A")

                    col5.metric(
                        "Maximum Drawdown",
                        f"{portfolio_max_drawdown:.2%}"
                    )
                    col6.metric(
                        "Sharpe Ratio",
                        f"{portfolio_sharpe:.2f}"
                    )

                    # ==========================================
                    # VaR AND EXPECTED SHORTFALL COMPARISON
                    # ==========================================

                    if risk_results is not None:
                        st.subheader(
                            "Value at Risk (VaR) and Expected Shortfall (ES)"
                        )
                        st.caption(
                            "These are estimated daily portfolio loss measures. "
                            "VaR estimates a loss threshold; Expected Shortfall "
                            "estimates the average loss beyond that threshold."
                        )
                        st.dataframe(
                            var_comparison.style.format({
                                "VaR 95%": "{:.2%}",
                                "VaR 99%": "{:.2%}",
                                "Expected Shortfall 95%": "{:.2%}",
                                "Expected Shortfall 99%": "{:.2%}"
                            }),
                            use_container_width=True
                        )

                    # ==========================================
                    # PORTFOLIO PERFORMANCE CHART
                    # ==========================================

                    st.subheader("Portfolio Performance")
                    st.line_chart(
                        portfolio_value.rename("Portfolio Value (€)")
                    )

                    # ==========================================
                    # PORTFOLIO VS S&P 500 BENCHMARK
                    # ==========================================

                    st.divider()
                    st.subheader("Portfolio vs S&P 500 Benchmark")

                    with st.spinner("Downloading S&P 500 benchmark data..."):
                        benchmark_data = yf.download(
                            "^GSPC",
                            period=download_period,
                            interval="1d",
                            auto_adjust=True,
                            progress=False
                        )

                    benchmark_prices = get_close_prices(benchmark_data)

                    if not benchmark_prices.empty:
                        benchmark_close = (
                            benchmark_prices.iloc[:, 0].dropna()
                        )

                        benchmark_returns = benchmark_close.pct_change(
                            fill_method=None
                        ).dropna()

                        comparison_returns = pd.concat(
                            [
                                daily_portfolio_returns.rename("Portfolio"),
                                benchmark_returns.rename("S&P 500")
                            ],
                            axis=1,
                            join="inner"
                        ).dropna()

                        if not comparison_returns.empty:
                            comparison_values = (
                                investment_amount
                                * (1 + comparison_returns).cumprod()
                            )

                            st.line_chart(comparison_values)

                            portfolio_benchmark_return = (
                                comparison_values["Portfolio"].iloc[-1]
                                / investment_amount
                            ) - 1

                            sp500_return = (
                                comparison_values["S&P 500"].iloc[-1]
                                / investment_amount
                            ) - 1

                            difference = (
                                portfolio_benchmark_return - sp500_return
                            )

                            col1, col2, col3 = st.columns(3)
                            col1.metric(
                                "Your Portfolio Return",
                                f"{portfolio_benchmark_return:.2%}"
                            )
                            col2.metric(
                                "S&P 500 Return",
                                f"{sp500_return:.2%}"
                            )
                            col3.metric(
                                "Outperformance",
                                f"{difference:.2%}"
                            )
                        else:
                            st.warning(
                                "Not enough overlapping dates to compare "
                                "your portfolio with the S&P 500."
                            )
                    else:
                        st.warning(
                            "Unable to download S&P 500 benchmark data."
                        )

                    # ==========================================
                    # INDIVIDUAL STOCK PRICES
                    # ==========================================

                    st.subheader("Individual Stock Prices")
                    st.line_chart(portfolio_prices)

                    # ==========================================
                    # DOWNLOAD PORTFOLIO REPORTS
                    # ==========================================

                    st.divider()
                    st.subheader("Download Portfolio Analysis")

                    performance_export = pd.concat(
                        [
                            daily_portfolio_returns.rename(
                                "Daily Portfolio Return"
                            ),
                            portfolio_value.rename("Portfolio Value (€)")
                        ],
                        axis=1
                    )

                    performance_csv = performance_export.to_csv(
                        index_label="Date"
                    ).encode("utf-8")

                    allocation_csv = allocation_data.to_csv(
                        index=False
                    ).encode("utf-8")

                    download_col1, download_col2, download_col3 = st.columns(3)

                    download_col1.download_button(
                        label="Download Performance CSV",
                        data=performance_csv,
                        file_name="portfolio_performance.csv",
                        mime="text/csv"
                    )

                    download_col2.download_button(
                        label="Download Allocation CSV",
                        data=allocation_csv,
                        file_name="portfolio_allocation.csv",
                        mime="text/csv"
                    )

                    if risk_results is not None:
                        risk_csv = var_comparison.to_csv(
                            index=False
                        ).encode("utf-8")

                        download_col3.download_button(
                            label="Download VaR and ES CSV",
                            data=risk_csv,
                            file_name="portfolio_var_es_analysis.csv",
                            mime="text/csv"
                        )
                        
