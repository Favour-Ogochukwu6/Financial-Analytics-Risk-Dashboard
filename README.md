# Financial Analytics & Risk Dashboard

## Project Overview

The Financial Analytics & Risk Dashboard is a Python-based financial analysis application developed with Streamlit. It allows users to analyse individual stocks, build investment portfolios, evaluate portfolio performance and measure financial risk.

The dashboard also compares portfolio performance against the S&P 500 benchmark index.

## Project Objectives

* Analyse historical stock price movements.
* Evaluate investment returns and portfolio performance.
* Measure investment risk using different financial risk metrics.
* Compare different Value at Risk (VaR) calculation methods.
* Examine relationships between selected stocks.
* Compare portfolio performance against the S&P 500.

## Key Features

### 1. Individual Stock Analysis

* Historical stock price analysis.
* Daily stock returns.
* Total return.
* Annualised volatility.
* Historical Value at Risk (VaR).
* Maximum drawdown.
* Sharpe ratio.

### 2. Portfolio Analysis

* Select multiple stocks for an investment portfolio.
* Configure investment weights.
* Calculate portfolio value and total return.
* Visualise portfolio allocation.
* Analyse portfolio performance and risk.

### 3. Risk Analysis

The dashboard calculates Value at Risk and Expected Shortfall using three methods:

* Historical Simulation
* Parametric Method
* Monte Carlo Simulation

Risk calculations are available at 95% and 99% confidence levels.

### 4. Correlation Analysis

A correlation heatmap visualises the relationships between selected stocks and helps assess potential diversification benefits.

### 5. S&P 500 Benchmark Comparison

The dashboard compares the performance of the selected investment portfolio against the S&P 500 index.

### 6. Data Export

Users can download:

* Portfolio performance data.
* Portfolio allocation data.
* VaR and Expected Shortfall results.

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* yfinance
* Plotly
* SciPy

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Financial-Analytics-Risk-Dashboard
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

## Financial Methodology and Assumptions

* Historical stock prices are retrieved using Yahoo Finance through yfinance.
* Portfolio weights are applied to daily stock returns.
* Annualised volatility is calculated from daily return volatility.
* The Sharpe ratio assumes a risk-free rate of zero.
* VaR and Expected Shortfall are estimated using historical, parametric and Monte Carlo approaches.
* Monte Carlo simulation uses 10,000 simulated daily returns.
* Benchmark comparisons use overlapping trading dates.

## Disclaimer

This project is developed for educational and analytical purposes. It does not constitute financial advice or a recommendation to buy or sell securities.

## Author
Ogbaka Favour
MSc Business Information Systems 
Specialisation: Advanced Analytics

**Project:** Financial Analytics & Risk Dashboard
