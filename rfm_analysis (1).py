"""
Stock Market EDA & Anomaly Detection
Author: Ruchika Kumari
Tools: Python, Pandas, Statsmodels, Seaborn, Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC STOCK DATA (5 years, NSE-style)
# ─────────────────────────────────────────────
def generate_stock_data(ticker, start_price, n_days=1260):
    dates = pd.bdate_range(start='2019-01-01', periods=n_days)
    returns = np.random.normal(0.0003, 0.015, n_days)

    # Inject market events (crashes/rallies)
    returns[260:280] -= 0.04   # COVID crash
    returns[280:300] += 0.02   # Recovery
    returns[600:610] -= 0.025  # Correction
    returns[900:910] += 0.02   # Rally

    prices = start_price * np.cumprod(1 + returns)
    volumes = np.random.randint(500000, 5000000, n_days)

    df = pd.DataFrame({
        'Date': dates,
        'Open': (prices * np.random.uniform(0.99, 1.01, n_days)).round(2),
        'High': (prices * np.random.uniform(1.00, 1.03, n_days)).round(2),
        'Low': (prices * np.random.uniform(0.97, 1.00, n_days)).round(2),
        'Close': prices.round(2),
        'Volume': volumes
    })
    df['Ticker'] = ticker
    return df.set_index('Date')

print("=" * 60)
print("STOCK MARKET EDA & ANOMALY DETECTION")
print("=" * 60)

# Generate data for 3 stocks
tickers = {
    'RELIANCE': generate_stock_data('RELIANCE', 1200),
    'INFY': generate_stock_data('INFY', 900),
    'TCS': generate_stock_data('TCS', 2500),
}

stock = tickers['RELIANCE'].copy()
print(f"\nReliance Industries — {len(stock)} trading days")
print(stock[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2))

# ─────────────────────────────────────────────
# 2. TECHNICAL INDICATORS
# ─────────────────────────────────────────────
# Moving Averages
stock['MA_20'] = stock['Close'].rolling(20).mean()
stock['MA_50'] = stock['Close'].rolling(50).mean()
stock['MA_200'] = stock['Close'].rolling(200).mean()

# Bollinger Bands (20-day, 2 std)
stock['BB_Upper'] = stock['MA_20'] + 2 * stock['Close'].rolling(20).std()
stock['BB_Lower'] = stock['MA_20'] - 2 * stock['Close'].rolling(20).std()

# Daily Returns
stock['Daily_Return'] = stock['Close'].pct_change()
stock['Log_Return'] = np.log(stock['Close'] / stock['Close'].shift(1))

# Volatility (21-day rolling)
stock['Volatility'] = stock['Daily_Return'].rolling(21).std() * np.sqrt(252)

print(f"\nAnnualised Volatility (mean): {stock['Volatility'].mean():.1%}")
print(f"Max Daily Return: {stock['Daily_Return'].max():.2%}")
print(f"Min Daily Return: {stock['Daily_Return'].min():.2%}")

# ─────────────────────────────────────────────
# 3. ANOMALY DETECTION
# ─────────────────────────────────────────────
# Z-Score method
stock['Return_ZScore'] = (stock['Daily_Return'] - stock['Daily_Return'].mean()) \
                          / stock['Daily_Return'].std()
stock['Anomaly_ZScore'] = stock['Return_ZScore'].abs() > 2.5

# IQR method
Q1 = stock['Daily_Return'].quantile(0.25)
Q3 = stock['Daily_Return'].quantile(0.75)
IQR = Q3 - Q1
stock['Anomaly_IQR'] = (stock['Daily_Return'] < Q1 - 2.5 * IQR) | \
                        (stock['Daily_Return'] > Q3 + 2.5 * IQR)

# Combined anomaly flag
stock['Anomaly'] = stock['Anomaly_ZScore'] | stock['Anomaly_IQR']
anomalies = stock[stock['Anomaly']].copy()
print(f"\nAnomalies detected: {len(anomalies)} days ({len(anomalies)/len(stock):.1%} of trading days)")

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle('Reliance Industries — Stock Market EDA & Anomaly Detection\n(2019-2024)',
             fontsize=15, fontweight='bold')

# Panel 1: Price + Bollinger Bands
ax1 = fig.add_subplot(3, 1, 1)
recent = stock.iloc[-756:]  # Last 3 years
ax1.plot(recent.index, recent['Close'], color='#1F4E79', lw=1.5, label='Close Price')
ax1.plot(recent.index, recent['MA_20'], color='orange', lw=1, label='20-day MA', linestyle='--')
ax1.plot(recent.index, recent['MA_50'], color='green', lw=1, label='50-day MA', linestyle='--')
ax1.fill_between(recent.index, recent['BB_Upper'], recent['BB_Lower'],
                 alpha=0.15, color='blue', label='Bollinger Bands')
ax1.scatter(recent[recent['Anomaly']].index,
            recent[recent['Anomaly']]['Close'],
            color='red', s=40, zorder=5, label='Anomaly', marker='^')
ax1.set_title('Price Chart with Bollinger Bands & Anomalies')
ax1.set_ylabel('Price (INR)')
ax1.legend(loc='upper left', fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Panel 2: Daily Returns
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(stock.index, stock['Daily_Return'], color='#555555', lw=0.8, alpha=0.8)
ax2.scatter(anomalies.index, anomalies['Daily_Return'],
            color='red', s=25, zorder=5, label='Anomaly')
ax2.axhline(0, color='black', lw=0.8, linestyle='-')
ax2.axhline(stock['Daily_Return'].mean() + 2.5 * stock['Daily_Return'].std(),
            color='orange', lw=1, linestyle='--', label='+2.5σ')
ax2.axhline(stock['Daily_Return'].mean() - 2.5 * stock['Daily_Return'].std(),
            color='orange', lw=1, linestyle='--', label='-2.5σ')
ax2.set_title('Daily Returns with Z-Score Anomaly Boundaries')
ax2.set_ylabel('Daily Return')
ax2.legend(fontsize=8)

# Panel 3: Rolling Volatility
ax3 = fig.add_subplot(3, 1, 3)
ax3.fill_between(stock.index, stock['Volatility'], alpha=0.6, color='#FF9800')
ax3.plot(stock.index, stock['Volatility'], color='#FF6500', lw=1)
ax3.set_title('Annualised 21-Day Rolling Volatility')
ax3.set_ylabel('Volatility')
ax3.set_xlabel('Date')

plt.tight_layout()
plt.savefig('outputs/stock_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Return Distribution
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Return Distribution Analysis', fontsize=13, fontweight='bold')

axes[0].hist(stock['Daily_Return'].dropna(), bins=60, color='#1F4E79',
             edgecolor='white', alpha=0.8)
axes[0].set_title('Distribution of Daily Returns')
axes[0].set_xlabel('Daily Return')
axes[0].set_ylabel('Frequency')

# Correlation between stocks
all_close = pd.DataFrame({t: d['Close'] for t, d in tickers.items()})
corr = all_close.pct_change().corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Stock Return Correlation Matrix')

plt.tight_layout()
plt.savefig('outputs/return_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Save anomaly report
anomalies[['Close', 'Daily_Return', 'Return_ZScore', 'Anomaly_ZScore', 'Anomaly_IQR']].to_csv(
    'outputs/anomaly_report.csv')
print("\nOutputs saved: stock_analysis.png, return_distribution.png, anomaly_report.csv")
print("✅ Stock Market Analysis complete.")