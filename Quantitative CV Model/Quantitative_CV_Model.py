import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats

# Define commodities and benchmarks
commodities = ['CL=F', 'BZ=F', 'NG=F', 'GC=F', 'SI=F', 'HG=F', 'ZC=F', 'ZW=F', 'ZS=F']
benchmarks = ['^GSPC', 'URTH', 'SPN']  # S&P 500, MSCI World Index, S&P 500 Energy

# Download historical data
data = yf.download(commodities + benchmarks, start='2010-01-01', end='2023-09-01')['Adj Close']

# Calculate daily log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Calculate Beta for each commodity relative to each benchmark
def calculate_beta(asset_returns, benchmark_returns):
    cov = np.cov(asset_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    return cov / var

betas = {}
for commodity in commodities:
    betas[commodity] = {}
    for benchmark in benchmarks:
        betas[commodity][benchmark] = calculate_beta(log_returns[commodity], log_returns[benchmark])

# Calculate VaR (95% confidence) and Expected Shortfall (ES)
confidence_level = 0.95
z_score = stats.norm.ppf(1 - confidence_level)

VaRs = {}
ESs = {}

for commodity in commodities:
    returns = log_returns[commodity]
    mean = returns.mean()
    std_dev = returns.std()
    
    # Parametric VaR
    VaR = mean + z_score * std_dev
    VaRs[commodity] = VaR
    
    # Expected Shortfall (ES)
    ES = returns[returns < VaR].mean()
    ESs[commodity] = ES

# Save results to a dataframe for easy access
beta_df = pd.DataFrame(betas)
VaR_df = pd.Series(VaRs, name='VaR')
ES_df = pd.Series(ESs, name='Expected Shortfall')

# Display results
print("Betas:")
print(beta_df)
print("\nValue-at-Risk (VaR):")
print(VaR_df)
print("\nExpected Shortfall (ES):")
print(ES_df)

# Save to Excel
with pd.ExcelWriter('commodities_risk_metrics.xlsx') as writer:
    beta_df.to_excel(writer, sheet_name='Beta')
    VaR_df.to_excel(writer, sheet_name='VaR')
    ES_df.to_excel(writer, sheet_name='Expected Shortfall')
