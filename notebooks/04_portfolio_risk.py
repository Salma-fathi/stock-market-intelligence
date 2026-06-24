import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/stocks_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ── 1. Pivot to Wide Format ─────────────────────────────────────
prices = df.pivot_table(values='Close', index='Date', columns='Ticker')
prices = prices.sort_index().dropna()
print(f"✅ Price matrix: {prices.shape}")

# ── 2. Daily Returns ────────────────────────────────────────────
returns = prices.pct_change().dropna()
print(f"✅ Returns matrix: {returns.shape}")

# ── 3. Key Risk Metrics ─────────────────────────────────────────
TRADING_DAYS = 252

ann_return = returns.mean() * TRADING_DAYS
ann_vol    = returns.std() * np.sqrt(TRADING_DAYS)
sharpe     = ann_return / ann_vol
skewness   = returns.skew()
kurt       = returns.kurt()

# Value at Risk (95% & 99%)
VaR_95 = returns.quantile(0.05)
VaR_99 = returns.quantile(0.01)

# Max Drawdown
def max_drawdown(return_series):
    cum = (1 + return_series).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return dd.min()

drawdowns = {ticker: max_drawdown(returns[ticker]) 
             for ticker in returns.columns}

# Summary Table
summary = pd.DataFrame({
    'Annual Return %' : (ann_return * 100).round(2),
    'Annual Volatility%': (ann_vol * 100).round(2),
    'Sharpe Ratio'    : sharpe.round(3),
    'VaR 95%'         : (VaR_95 * 100).round(3),
    'VaR 99%'         : (VaR_99 * 100).round(3),
    'Max Drawdown%'   : pd.Series(drawdowns).multiply(100).round(2),
    'Skewness'        : skewness.round(3),
    'Kurtosis'        : kurt.round(3)
})

print("\n📊 RISK METRICS SUMMARY:")
print(summary.to_string())
summary.to_csv("data/risk_metrics.csv")

# ── 4. Portfolio Optimization ───────────────────────────────────
mean_returns = returns.mean()
cov_matrix   = returns.cov()
n_assets     = len(returns.columns)

def portfolio_stats(weights):
    w   = np.array(weights)
    ret = np.dot(w, mean_returns) * TRADING_DAYS
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * TRADING_DAYS, w)))
    sr  = ret / vol
    return ret, vol, sr

def neg_sharpe(weights):
    return -portfolio_stats(weights)[2]

# Constraints & Bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds      = tuple((0.05, 0.40) for _ in range(n_assets))
init_weights = [1/n_assets] * n_assets

# Optimize
result = minimize(neg_sharpe, init_weights,
                  method='SLSQP',
                  bounds=bounds,
                  constraints=constraints)

opt_weights = result.x
opt_ret, opt_vol, opt_sr = portfolio_stats(opt_weights)

print("\n🎯 OPTIMAL PORTFOLIO (Max Sharpe Ratio):")
for ticker, w in zip(returns.columns, opt_weights):
    print(f"  {ticker}: {w*100:.1f}%")
print(f"  Expected Annual Return: {opt_ret*100:.2f}%")
print(f"  Expected Volatility:    {opt_vol*100:.2f}%")
print(f"  Sharpe Ratio:           {opt_sr:.3f}")

# ── 5. Monte Carlo Simulation ───────────────────────────────────
print("\n🎲 Running Monte Carlo (5000 portfolios)...")
mc_returns, mc_vols, mc_sharpes, mc_weights = [], [], [], []

for _ in range(5000):
    w = np.random.random(n_assets)
    w /= w.sum()
    r, v, s = portfolio_stats(w)
    mc_returns.append(r)
    mc_vols.append(v)
    mc_sharpes.append(s)
    mc_weights.append(w)

mc_returns  = np.array(mc_returns)
mc_vols     = np.array(mc_vols)
mc_sharpes  = np.array(mc_sharpes)

# ── 6. Visualizations ──────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "xy"}, {"type": "domain"}],
        [{"type": "xy"}, {"type": "xy"}]
    ],
    subplot_titles=(
        'Efficient Frontier (Monte Carlo)',
        'Optimal Portfolio Allocation',
        'Cumulative Returns by Stock',
        'Risk Metrics Comparison'
    )
)

# Plot 1: Efficient Frontier
fig.add_trace(go.Scatter(
    x=mc_vols * 100, y=mc_returns * 100,
    mode='markers',
    marker=dict(
        color=mc_sharpes, colorscale='Viridis',
        size=3, opacity=0.5,
        colorbar=dict(title='Sharpe', x=0.45)
    ),
    name='Portfolios'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[opt_vol * 100], y=[opt_ret * 100],
    mode='markers',
    marker=dict(color='red', size=15, symbol='star'),
    name='Optimal Portfolio'
), row=1, col=1)

fig.update_xaxes(title_text='Volatility %', row=1, col=1)
fig.update_yaxes(title_text='Return %',     row=1, col=1)

# Plot 2: Optimal Weights Pie
fig.add_trace(go.Pie(
    labels=list(returns.columns),
    values=opt_weights,
    hole=0.4,
    marker_colors=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0']
), row=1, col=2)

# Plot 3: Cumulative Returns
cum_returns = (1 + returns).cumprod()
for ticker in returns.columns:
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns[ticker],
        name=ticker, mode='lines'
    ), row=2, col=1)

# Plot 4: Risk Metrics Bar
fig.add_trace(go.Bar(
    x=list(returns.columns),
    y=summary['Sharpe Ratio'].values,
    name='Sharpe Ratio',
    marker_color=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0']
), row=2, col=2)

fig.update_layout(
    title='Portfolio Risk Analysis Dashboard',
    height=900,
    template='plotly_dark',
    showlegend=True
)

fig.write_html("data/portfolio_risk_dashboard.html")
print("\n✅ Saved: data/portfolio_risk_dashboard.html")
print("🎉 Portfolio Risk Analysis Complete!")