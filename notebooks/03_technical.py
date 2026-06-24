import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/stocks_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])
print(f"✅ Loaded: {df.shape}")

# ── Technical Indicators ────────────────────────────────────────
def add_indicators(stock_df):
    d = stock_df.copy().sort_values('Date')
    
    # Moving Averages
    d['MA_20']  = d['Close'].rolling(20).mean()
    d['MA_50']  = d['Close'].rolling(50).mean()
    d['MA_200'] = d['Close'].rolling(200).mean()
    
    # Bollinger Bands
    d['BB_mid']   = d['Close'].rolling(20).mean()
    d['BB_std']   = d['Close'].rolling(20).std()
    d['BB_upper'] = d['BB_mid'] + 2 * d['BB_std']
    d['BB_lower'] = d['BB_mid'] - 2 * d['BB_std']
    
    # RSI
    delta     = d['Close'].diff()
    gain      = delta.where(delta > 0, 0).rolling(14).mean()
    loss      = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs        = gain / loss
    d['RSI']  = 100 - (100 / (1 + rs))
    
    # MACD
    ema12      = d['Close'].ewm(span=12).mean()
    ema26      = d['Close'].ewm(span=26).mean()
    d['MACD']  = ema12 - ema26
    d['Signal']= d['MACD'].ewm(span=9).mean()
    d['Hist']  = d['MACD'] - d['Signal']
    
    # Buy/Sell Signals
    d['Buy_Signal']  = (
        (d['MA_20'] > d['MA_50']) & 
        (d['MA_20'].shift(1) <= d['MA_50'].shift(1)) &
        (d['RSI'] < 70)
    )
    d['Sell_Signal'] = (
        (d['MA_20'] < d['MA_50']) & 
        (d['MA_20'].shift(1) >= d['MA_50'].shift(1)) &
        (d['RSI'] > 30)
    )
    
    return d

# ── Plot Function ───────────────────────────────────────────────
def plot_stock(ticker, stock_df):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{ticker} - Price & Bollinger Bands',
            'RSI (Relative Strength Index)',
            'MACD'
        ),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # ── Row 1: Candlestick + Indicators ────────────────────────
    fig.add_trace(go.Candlestick(
        x=stock_df['Date'],
        open=stock_df['Open'],
        high=stock_df['High'],
        low=stock_df['Low'],
        close=stock_df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    for ma, color in [('MA_20','blue'), ('MA_50','orange'), ('MA_200','purple')]:
        fig.add_trace(go.Scatter(
            x=stock_df['Date'], y=stock_df[ma],
            name=ma, line=dict(color=color, width=1.5)
        ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=stock_df['Date'], y=stock_df['BB_upper'],
        name='BB Upper', line=dict(color='gray', dash='dash', width=1),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=stock_df['Date'], y=stock_df['BB_lower'],
        name='BB Lower', line=dict(color='gray', dash='dash', width=1),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ), row=1, col=1)
    
    # Buy/Sell signals
    buys  = stock_df[stock_df['Buy_Signal']]
    sells = stock_df[stock_df['Sell_Signal']]
    
    fig.add_trace(go.Scatter(
        x=buys['Date'], y=buys['Close'],
        mode='markers', name='Buy Signal',
        marker=dict(symbol='triangle-up', size=12, color='green')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells['Date'], y=sells['Close'],
        mode='markers', name='Sell Signal',
        marker=dict(symbol='triangle-down', size=12, color='red')
    ), row=1, col=1)
    
    # ── Row 2: RSI ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=stock_df['Date'], y=stock_df['RSI'],
        name='RSI', line=dict(color='purple', width=1.5)
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, row=2, col=1)
    
    # ── Row 3: MACD ─────────────────────────────────────────────
    colors = ['green' if v >= 0 else 'red' for v in stock_df['Hist']]
    fig.add_trace(go.Bar(
        x=stock_df['Date'], y=stock_df['Hist'],
        name='MACD Hist', marker_color=colors, opacity=0.7
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=stock_df['Date'], y=stock_df['MACD'],
        name='MACD', line=dict(color='blue', width=1.5)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=stock_df['Date'], y=stock_df['Signal'],
        name='Signal', line=dict(color='orange', width=1.5)
    ), row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} Technical Analysis Dashboard',
        height=900,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02)
    )
    
    return fig

# ── Generate All Charts ─────────────────────────────────────────
STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

for ticker in STOCKS:
    print(f"📊 Generating chart for {ticker}...")
    stock = df[df['Ticker'] == ticker].copy()
    stock = add_indicators(stock)
    fig   = plot_stock(ticker, stock)
    fig.write_html(f"data/{ticker}_dashboard.html")
    print(f"  ✅ Saved: data/{ticker}_dashboard.html")

print("\n🎉 All dashboards generated!")
print("Open any HTML file in your browser to view!")