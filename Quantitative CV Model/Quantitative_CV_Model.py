import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats

# Define commodities and benchmarks
commodities = ['CL=F', 'BZ=F', 'NG=F', 'GC=F', 'SI=F', 'HG=F', 'ZC=F', 'ZW=F', 'ZS=F']
benchmarks = ['^GSPC', 'URTH', 'SPN']  # S&P 500, MSCI World Index, S&P 500 Energy

# Download historical data
print("Downloading data...")
data = yf.download(commodities + benchmarks, start='2010-01-01', end='2023-09-01')['Adj Close']

# Debug: Check for missing data
print("\nInitial missing data in the downloaded dataset:")
print(data.isna().sum())

# Handle missing data
data = data.fillna(method='ffill').dropna()

# Debug: Check data after handling missing values
print("\nMissing data after forward-fill and dropping remaining NaNs:")
print(data.isna().sum())

# Calculate daily log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Debug: Check for missing or inf values in log returns
print("\nChecking for NaN or inf in log returns:")
print(log_returns.isna().sum())
print(log_returns[log_returns.isin([np.nan, np.inf, -np.inf])])

# Replace inf values with NaN, and drop rows with NaN values
log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
log_returns = log_returns.dropna()

# Debug: Check log returns after cleaning
print("\nLog returns after cleaning NaN or inf values:")
print(log_returns.isna().sum())

# Calculate Beta for each commodity relative to each benchmark
def calculate_beta(asset_returns, benchmark_returns):
    if asset_returns.isna().sum() > 0 or benchmark_returns.isna().sum() > 0:
        print(f"NaN detected in asset or benchmark returns for Beta calculation.")
        return np.nan
    cov = np.cov(asset_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    if var == 0:
        print(f"Variance is zero for benchmark, unable to calculate Beta.")
        return np.nan
    return cov / var

betas = {}
for commodity in commodities:
    betas[commodity] = {}
    for benchmark in benchmarks:
        print(f"Calculating Beta for {commodity} vs {benchmark}...")
        betas[commodity][benchmark] = calculate_beta(log_returns[commodity], log_returns[benchmark])

# Debug: Check Betas
print("\nBeta values:")
print(pd.DataFrame(betas))

# Calculate VaR (95% confidence) and Expected Shortfall (ES)
confidence_level = 0.95
z_score = stats.norm.ppf(1 - confidence_level)

VaRs = {}
ESs = {}

for commodity in commodities:
    print(f"\nCalculating VaR and ES for {commodity}...")
    returns = log_returns[commodity]
    if returns.isna().sum() > 0:
        print(f"NaN detected in returns for {commodity} during VaR/ES calculation.")
        VaRs[commodity], ESs[commodity] = np.nan, np.nan
        continue
    
    mean = returns.mean()
    std_dev = returns.std()

    # Check if standard deviation is zero
    if std_dev == 0:
        print(f"Standard deviation is zero for {commodity}, unable to calculate VaR/ES.")
        VaRs[commodity], ESs[commodity] = np.nan, np.nan
        continue

    # Parametric VaR
    VaR = mean + z_score * std_dev
    VaRs[commodity] = VaR
    
    # Expected Shortfall (ES)
    ES = returns[returns < VaR].mean() if returns[returns < VaR].size > 0 else np.nan
    ESs[commodity] = ES

# Debug: Check VaR and ES results
print("\nValue-at-Risk (VaR):")
print(pd.Series(VaRs, name='VaR'))

print("\nExpected Shortfall (ES):")
print(pd.Series(ESs, name='Expected Shortfall'))

# Save results to Excel for further inspection
with pd.ExcelWriter('commodities_risk_metrics_debugged.xlsx') as writer:
    pd.DataFrame(betas).to_excel(writer, sheet_name='Beta')
    pd.Series(VaRs, name='VaR').to_excel(writer, sheet_name='VaR')
    pd.Series(ESs, name='Expected Shortfall').to_excel(writer, sheet_name='Expected Shortfall')

print("\nResults saved to 'commodities_risk_metrics_debugged.xlsx'")
