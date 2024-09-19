# Install yfinance if you don't have it
# pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np

# Define the energy commodities to analyze
commodities = ['CL=F', 'BZ=F', 'NG=F']  # WTI Crude Oil, Brent Crude Oil, Natural Gas

# Download historical data (adjusted close prices)
data = yf.download(commodities, start='2010-01-01', end='2023-09-01')['Adj Close']

# Calculate daily log returns
log_returns = np.log(data / data.shift(1))

# Calculate volatility (30-day rolling window standard deviation)
volatility = log_returns.rolling(window=30).std()

# Calculate Sharpe Ratio (annualized using 252 trading days)
risk_free_rate = 0.01  # Assuming 1% annual risk-free rate (adjust if needed)
sharpe_ratio = (log_returns.mean() * 252 - risk_free_rate) / (log_returns.std() * np.sqrt(252))

# Print a sample of the calculated data
print(f"Volatility for commodities:\n{volatility.head()}")
print(f"Sharpe Ratios:\n{sharpe_ratio}")

# Optionally save the data to a CSV for further analysis
volatility.to_csv('commodities_volatility.csv')
sharpe_ratio.to_csv('commodities_sharpe_ratio.csv')

