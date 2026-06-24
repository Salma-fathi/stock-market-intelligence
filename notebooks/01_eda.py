import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & Fix Multi-level Columns ──────────────────────────
df = pd.read_csv("data/stocks_raw.csv", header=[0,1])

# Flatten multi-level columns
df.columns = ['_'.join(col).strip('_') for col in df.columns]
print("Columns after fix:", df.columns.tolist())

# ── 2. Reshape to Long Format ───────────────────────────────────
import yfinance as yf

STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

all_data = []
for ticker in STOCKS:
    stock = yf.download(ticker, start="2020-01-01", end="2024-12-31", auto_adjust=True)
    stock = stock[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
    stock.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    stock['Ticker'] = ticker
    stock.reset_index(inplace=True)
    all_data.append(stock)

df = pd.concat(all_data, ignore_index=True)
df.rename(columns={'Date': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Save clean version
df.to_csv("data/stocks_clean.csv", index=False)
print(f"✅ Clean data shape: {df.shape}")
print(df.head(10))
print(df.dtypes)

# ── 3. Basic Stats ──────────────────────────────────────────────
print("\n📊 Basic Statistics:")
print(df.groupby('Ticker')['Close'].describe().round(2))

# ── 4. Add Features ────────────────────────────────────────────
df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
df['MA_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
df['MA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())
df['Volatility'] = df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(20).std())

df.to_csv("data/stocks_features.csv", index=False)
print("\n✅ Features added and saved!")

# ── 5. Visualizations ──────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Stock Market Intelligence - EDA', fontsize=16, fontweight='bold')

# Plot 1: Closing Price Over Time
ax1 = axes[0, 0]
for ticker in STOCKS:
    data = df[df['Ticker'] == ticker]
    ax1.plot(data['Date'], data['Close'], label=ticker, linewidth=1.5)
ax1.set_title('Closing Price Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Daily Returns Distribution
ax2 = axes[0, 1]
for ticker in STOCKS:
    data = df[df['Ticker'] == ticker]['Daily_Return'].dropna()
    ax2.hist(data, bins=50, alpha=0.5, label=ticker)
ax2.set_title('Daily Returns Distribution')
ax2.set_xlabel('Daily Return')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Volatility Over Time
ax3 = axes[1, 0]
for ticker in STOCKS:
    data = df[df['Ticker'] == ticker]
    ax3.plot(data['Date'], data['Volatility'], label=ticker, linewidth=1)
ax3.set_title('20-Day Rolling Volatility')
ax3.set_xlabel('Date')
ax3.set_ylabel('Volatility')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Volume Analysis
ax4 = axes[1, 1]
avg_volume = df.groupby('Ticker')['Volume'].mean() / 1e6
avg_volume.plot(kind='bar', ax=ax4, color=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0'])
ax4.set_title('Average Daily Volume (Millions)')
ax4.set_xlabel('Stock')
ax4.set_ylabel('Volume (M)')
ax4.tick_params(axis='x', rotation=0)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Correlation Heatmap
ax5 = axes[2, 0]
pivot = df.pivot_table(values='Close', index='Date', columns='Ticker')
corr = pivot.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax5, 
            vmin=-1, vmax=1, center=0)
ax5.set_title('Stock Price Correlation')

# Plot 6: Cumulative Returns
ax6 = axes[2, 1]
for ticker in STOCKS:
    data = df[df['Ticker'] == ticker].copy()
    data = data.sort_values('Date')
    cumulative = (1 + data['Daily_Return'].fillna(0)).cumprod() - 1
    ax6.plot(data['Date'], cumulative * 100, label=ticker, linewidth=1.5)
ax6.set_title('Cumulative Returns (%)')
ax6.set_xlabel('Date')
ax6.set_ylabel('Return (%)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/eda_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Charts saved to data/eda_charts.png")