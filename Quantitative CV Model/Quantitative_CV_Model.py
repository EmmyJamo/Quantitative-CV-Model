# Install yfinance if you don't have it
# pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np

# Define the energy commodities to analyze
commodities = ['CL=F', 'BZ=F', 'NG=F']  # WTI Crude Oil, Brent Crude Oil, Natural Gas

# Download historical data (adjusted close prices)
data = yf.download(commodities, start='2010-01-01', end='2023-09-01')['Adj Close']

print(data) #debug

# Calculate daily log returns
log_returns = np.log(data / data.shift(1))

# Calculate volatility (30-day rolling window standard deviation)
volatility = log_returns.rolling(window=30).std()

# Calculate Sharpe Ratio (annualized using 252 trading days)
risk_free_rate = 0.01  # Assuming 1% annual risk-free rate - is this a good rate, to review!
sharpe_ratio = (log_returns.mean() * 252 - risk_free_rate) / (log_returns.std() * np.sqrt(252))

# Now saving the data to an Excel file with two sheets
file_name = 'commodities_risk_data.xlsx'

# Creating an Excel writer
with pd.ExcelWriter(file_name) as writer:
    # Write Volatility data to the first sheet
    volatility.to_excel(writer, sheet_name='Volatility')
    
    # Write Sharpe Ratio data to the second sheet
    sharpe_ratio.to_excel(writer, sheet_name='Sharpe Ratio')

# File saved
print(f"Data saved to {file_name}")

