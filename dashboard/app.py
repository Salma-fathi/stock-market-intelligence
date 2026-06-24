import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import dash
from dash import dcc, html, Input, Output, callback
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ───────────────────────────────────────────────────
df       = pd.read_csv("data/stocks_clean.csv")
df['Date'] = pd.to_datetime(df['Date'])
risk     = pd.read_csv("data/risk_metrics.csv", index_col=0)
STOCKS   = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
COLORS   = {'AAPL':'#2196F3','GOOGL':'#4CAF50',
            'MSFT':'#FF9800','AMZN':'#E91E63','TSLA':'#9C27B0'}

# ── Technical Indicators ────────────────────────────────────────
def add_indicators(d):
    d = d.copy().sort_values('Date')
    d['MA_20']    = d['Close'].rolling(20).mean()
    d['MA_50']    = d['Close'].rolling(50).mean()
    d['MA_200']   = d['Close'].rolling(200).mean()
    d['BB_mid']   = d['Close'].rolling(20).mean()
    d['BB_upper'] = d['BB_mid'] + 2 * d['Close'].rolling(20).std()
    d['BB_lower'] = d['BB_mid'] - 2 * d['Close'].rolling(20).std()
    delta         = d['Close'].diff()
    gain          = delta.where(delta > 0, 0).rolling(14).mean()
    loss          = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI']      = 100 - (100 / (1 + gain / loss))
    ema12         = d['Close'].ewm(span=12).mean()
    ema26         = d['Close'].ewm(span=26).mean()
    d['MACD']     = ema12 - ema26
    d['Signal']   = d['MACD'].ewm(span=9).mean()
    d['Hist']     = d['MACD'] - d['Signal']
    d['Buy']      = ((d['MA_20'] > d['MA_50']) & 
                     (d['MA_20'].shift(1) <= d['MA_50'].shift(1)))
    d['Sell']     = ((d['MA_20'] < d['MA_50']) & 
                     (d['MA_20'].shift(1) >= d['MA_50'].shift(1)))
    return d

# ── App Layout ──────────────────────────────────────────────────
app = dash.Dash(__name__, title="Stock Market Intelligence")

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("📈 Stock Market Intelligence Dashboard",
                style={'color':'#00BCD4','margin':'0','fontSize':'28px'}),
        html.P("Technical Analysis • Portfolio Risk • Market Insights",
               style={'color':'#888','margin':'5px 0 0 0'})
    ], style={
        'background':'#1a1a2e','padding':'20px 30px',
        'borderBottom':'2px solid #00BCD4'
    }),

    # Tabs
    dcc.Tabs(id='tabs', value='tab-overview', children=[
        dcc.Tab(label='📊 Market Overview',  value='tab-overview'),
        dcc.Tab(label='📉 Technical Analysis', value='tab-technical'),
        dcc.Tab(label='💼 Portfolio Risk',   value='tab-portfolio'),
        dcc.Tab(label='📋 Risk Metrics',     value='tab-metrics'),
    ], style={'background':'#16213e'},
       colors={'border':'#00BCD4','primary':'#00BCD4','background':'#16213e'}),

    html.Div(id='tab-content',
             style={'background':'#0f3460','minHeight':'100vh','padding':'20px'})

], style={'background':'#0f3460','fontFamily':'Arial, sans-serif'})

# ── Tab 1: Market Overview ──────────────────────────────────────
def render_overview():
    prices = df.pivot_table(values='Close', index='Date', columns='Ticker')
    cum    = (prices.pct_change().dropna() + 1).cumprod()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Stock Prices Over Time',
            'Cumulative Returns',
            'Correlation Heatmap',
            'Annual Return vs Volatility'
        ),
        specs=[[{"type":"xy"},{"type":"xy"}],
               [{"type":"xy"},{"type":"xy"}]]
    )

    # Price chart
    for t in STOCKS:
        s = df[df['Ticker']==t]
        fig.add_trace(go.Scatter(
            x=s['Date'], y=s['Close'], name=t,
            line=dict(color=COLORS[t], width=2)
        ), row=1, col=1)

    # Cumulative returns
    for t in STOCKS:
        fig.add_trace(go.Scatter(
            x=cum.index, y=(cum[t]-1)*100, name=t,
            line=dict(color=COLORS[t], width=2), showlegend=False
        ), row=1, col=2)

    # Correlation heatmap
    corr = prices.corr().round(2)
    fig.add_trace(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdBu', zmin=-1, zmax=1,
        text=corr.values, texttemplate='%{text}',
        showscale=False
    ), row=2, col=1)

    # Return vs Volatility scatter
    fig.add_trace(go.Scatter(
        x=risk['Annual Volatility%'],
        y=risk['Annual Return %'],
        mode='markers+text',
        text=risk.index,
        textposition='top center',
        marker=dict(
            size=risk['Sharpe Ratio']*20,
            color=list(COLORS.values()),
            opacity=0.8
        ),
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=750, template='plotly_dark',
        paper_bgcolor='#0f3460', plot_bgcolor='#16213e',
        legend=dict(orientation='h', y=1.08)
    )
    fig.update_xaxes(gridcolor='#1a1a2e')
    fig.update_yaxes(gridcolor='#1a1a2e')

    return html.Div([dcc.Graph(figure=fig, style={'height':'750px'})])

# ── Tab 2: Technical Analysis ───────────────────────────────────
def render_technical():
    return html.Div([
        html.Div([
            html.Label("Select Stock:", style={'color':'white','fontWeight':'bold'}),
            dcc.Dropdown(
                id='stock-selector',
                options=[{'label':t,'value':t} for t in STOCKS],
                value='AAPL',
                style={'background':'#16213e','color':'black','width':'200px'}
            )
        ], style={'marginBottom':'15px'}),
        dcc.Graph(id='technical-chart', style={'height':'850px'})
    ])

# ── Tab 3: Portfolio Risk ───────────────────────────────────────
def render_portfolio():
    prices  = df.pivot_table(values='Close', index='Date', columns='Ticker')
    returns = prices.pct_change().dropna()
    mean_r  = returns.mean()
    cov_m   = returns.cov()
    n       = len(STOCKS)
    TD      = 252

    def stats(w):
        w   = np.array(w)
        ret = np.dot(w, mean_r) * TD
        vol = np.sqrt(np.dot(w.T, np.dot(cov_m * TD, w)))
        return ret, vol, ret/vol

    result = minimize(
        lambda w: -stats(w)[2],
        [1/n]*n, method='SLSQP',
        bounds=tuple((0.05,0.40) for _ in range(n)),
        constraints={'type':'eq','fun':lambda x: np.sum(x)-1}
    )
    ow = result.x
    or_, ov, os_ = stats(ow)

    # Monte Carlo
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(3000):
        w = np.random.random(n); w /= w.sum()
        r, v, s = stats(w)
        mc_r.append(r); mc_v.append(v); mc_s.append(s)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Efficient Frontier',
            'Optimal Allocation',
            'Cumulative Returns',
            'Max Drawdown Comparison'
        ),
        specs=[[{"type":"xy"},{"type":"domain"}],
               [{"type":"xy"},{"type":"xy"}]]
    )

    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=np.array(mc_v)*100, y=np.array(mc_r)*100,
        mode='markers',
        marker=dict(color=mc_s, colorscale='Viridis', size=3, opacity=0.5),
        name='Portfolios'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[ov*100], y=[or_*100], mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name=f'Optimal (SR={os_:.2f})'
    ), row=1, col=1)
    fig.update_xaxes(title_text='Volatility %', row=1, col=1)
    fig.update_yaxes(title_text='Return %',     row=1, col=1)

    # Pie
    fig.add_trace(go.Pie(
        labels=STOCKS, values=ow, hole=0.4,
        marker_colors=list(COLORS.values()),
        textinfo='label+percent'
    ), row=1, col=2)

    # Cumulative returns
    cum = (1 + returns).cumprod()
    for t in STOCKS:
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[t], name=t,
            line=dict(color=COLORS[t], width=2), showlegend=False
        ), row=2, col=1)

    # Drawdown
    fig.add_trace(go.Bar(
        x=risk.index,
        y=risk['Max Drawdown%'].abs(),
        marker_color=list(COLORS.values()),
        text=risk['Max Drawdown%'].round(1),
        textposition='outside',
        showlegend=False
    ), row=2, col=2)
    fig.update_yaxes(title_text='Max Drawdown %', row=2, col=2)

    fig.update_layout(
        height=800, template='plotly_dark',
        paper_bgcolor='#0f3460', plot_bgcolor='#16213e'
    )

    return html.Div([dcc.Graph(figure=fig)])

# ── Tab 4: Risk Metrics Table ───────────────────────────────────
def render_metrics():
    rows = []
    for ticker in risk.index:
        rows.append(html.Tr([
            html.Td(ticker,                                     style={'color':COLORS[ticker],'fontWeight':'bold'}),
            html.Td(f"{risk.loc[ticker,'Annual Return %']:.2f}%"),
            html.Td(f"{risk.loc[ticker,'Annual Volatility%']:.2f}%"),
            html.Td(f"{risk.loc[ticker,'Sharpe Ratio']:.3f}"),
            html.Td(f"{risk.loc[ticker,'VaR 95%']:.3f}%"),
            html.Td(f"{risk.loc[ticker,'Max Drawdown%']:.2f}%"),
        ], style={'borderBottom':'1px solid #333','textAlign':'center'}))

    return html.Div([
        html.H3("📋 Risk Metrics Summary", style={'color':'#00BCD4'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(c, style={'color':'#00BCD4','padding':'10px','background':'#16213e'})
                for c in ['Stock','Annual Return','Volatility',
                          'Sharpe Ratio','VaR 95%','Max Drawdown']
            ])),
            html.Tbody(rows)
        ], style={
            'width':'100%','borderCollapse':'collapse',
            'color':'white','marginTop':'20px',
            'background':'#1a1a2e','borderRadius':'8px'
        })
    ], style={'padding':'20px'})

# ── Callbacks ───────────────────────────────────────────────────
@app.callback(Output('tab-content','children'), Input('tabs','value'))
def render_tab(tab):
    if tab == 'tab-overview':   return render_overview()
    if tab == 'tab-technical':  return render_technical()
    if tab == 'tab-portfolio':  return render_portfolio()
    if tab == 'tab-metrics':    return render_metrics()

@app.callback(
    Output('technical-chart','figure'),
    Input('stock-selector','value')
)
def update_technical(ticker):
    stock = df[df['Ticker']==ticker].copy()
    stock = add_indicators(stock)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=(
            f'{ticker} — Price, MAs & Bollinger Bands',
            'RSI', 'MACD'
        )
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=stock['Date'],
        open=stock['Open'], high=stock['High'],
        low=stock['Low'],   close=stock['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    for ma, color in [('MA_20','#2196F3'),('MA_50','#FF9800'),('MA_200','#9C27B0')]:
        fig.add_trace(go.Scatter(
            x=stock['Date'], y=stock[ma], name=ma,
            line=dict(color=color, width=1.5)
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock['Date'], y=stock['BB_upper'],
        line=dict(color='gray', dash='dash', width=1),
        showlegend=False, name='BB Upper'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=stock['Date'], y=stock['BB_lower'],
        line=dict(color='gray', dash='dash', width=1),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        showlegend=False, name='BB Lower'
    ), row=1, col=1)

    buys  = stock[stock['Buy']]
    sells = stock[stock['Sell']]
    fig.add_trace(go.Scatter(
        x=buys['Date'], y=buys['Close'], mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='#00FF00'),
        name='Buy Signal'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells['Date'], y=sells['Close'], mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='#FF0000'),
        name='Sell Signal'
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=stock['Date'], y=stock['RSI'],
        line=dict(color='#9C27B0', width=1.5), name='RSI'
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red',   row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)

    # MACD
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in stock['Hist']]
    fig.add_trace(go.Bar(
        x=stock['Date'], y=stock['Hist'],
        marker_color=colors, name='MACD Hist', opacity=0.7
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=stock['Date'], y=stock['MACD'],
        line=dict(color='#2196F3', width=1.5), name='MACD'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=stock['Date'], y=stock['Signal'],
        line=dict(color='#FF9800', width=1.5), name='Signal'
    ), row=3, col=1)

    fig.update_layout(
        height=850, template='plotly_dark',
        paper_bgcolor='#0f3460', plot_bgcolor='#16213e',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)