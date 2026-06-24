# 📈 Stock Market Intelligence Dashboard

> An end-to-end Data Science & Business Intelligence project analyzing 5 major tech stocks (AAPL, GOOGL, MSFT, AMZN, TSLA) using Python, Machine Learning, and interactive dashboards.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Plotly](https://img.shields.io/badge/Plotly-Dash-purple)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

---

## 🎯 Project Overview

This project delivers a full Business Intelligence solution for stock market analysis, covering:

- **Data Collection** — Real-time & historical stock data via `yfinance` API
- **Exploratory Data Analysis** — Trends, correlations, volatility patterns
- **Machine Learning** — Price direction prediction using Gradient Boosting
- **Technical Analysis** — RSI, MACD, Bollinger Bands, Buy/Sell signals
- **Portfolio Optimization** — Sharpe Ratio maximization via Monte Carlo simulation
- **Interactive Dashboard** — Full Plotly Dash web application

---

## 📊 Key Results

| Stock | Annual Return | Volatility | Sharpe Ratio | Max Drawdown |
|-------|-------------|------------|--------------|--------------|
| AAPL  | 29.95%      | 31.69%     | 0.945        | -31.43%      |
| GOOGL | 25.98%      | 32.51%     | 0.799        | -44.32%      |
| MSFT  | 25.07%      | 30.51%     | 0.822        | -37.15%      |
| AMZN  | 23.46%      | 35.98%     | 0.652        | -56.15%      |
| TSLA  | 76.25%      | 67.19%     | 1.135        | -73.63%      |

### 🎯 Optimal Portfolio (Max Sharpe Ratio = 1.225)
- **TSLA** 40.0% — Highest return driver
- **AAPL** 36.2% — Best risk-adjusted anchor
- **GOOGL** 13.8% — Diversification
- **AMZN** 5.0% — Minimum allocation
- **MSFT** 5.0% — Minimum allocation
- **Expected Annual Return: 47.35% | Volatility: 38.65%**

---

## 🗂️ Project Structure

```
stock-market-intelligence/
│
├── data/                          # Data files
│   ├── stocks_raw.csv             # Raw fetched data
│   ├── stocks_clean.csv           # Cleaned data
│   ├── stocks_features.csv        # Engineered features
│   ├── risk_metrics.csv           # Risk analysis results
│   ├── model_results_v2.csv       # ML model results
│   ├── eda_charts.png             # EDA visualizations
│   ├── predictions_v2.png         # ML prediction charts
│   ├── AAPL_dashboard.html        # Individual stock dashboards
│   └── portfolio_risk_dashboard.html
│
├── notebooks/
│   ├── 01_eda.py                  # Exploratory Data Analysis
│   ├── 03_technical.py            # Technical indicators
│   └── 04_portfolio_risk.py       # Portfolio optimization
│
├── models/
│   └── 02_prediction.py           # ML price prediction
│
├── dashboard/
│   └── app.py                     # Main Dash web application
│
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Data Collection** | yfinance, pandas |
| **Data Analysis** | numpy, pandas, scipy |
| **Machine Learning** | scikit-learn (Gradient Boosting, Random Forest) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Plotly Dash |
| **Statistics** | scipy.optimize (Portfolio Optimization) |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Salma-fathi/stock-market-intelligence.git
cd stock-market-intelligence
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install yfinance pandas numpy matplotlib seaborn plotly dash scikit-learn scipy
```

### 4. Fetch data
```bash
python data/fetch_data.py
```

### 5. Run analysis pipeline
```bash
python notebooks/01_eda.py
python models/02_prediction.py
python notebooks/03_technical.py
python notebooks/04_portfolio_risk.py
```

### 6. Launch dashboard
```bash
python dashboard/app.py
```
Open **http://127.0.0.1:8050** in your browser.

---

## 📱 Dashboard Features

### 📊 Tab 1 — Market Overview
- Stock price history for all 5 companies (2020–2024)
- Cumulative returns comparison
- Correlation heatmap between stocks
- Risk vs Return scatter plot (bubble size = Sharpe Ratio)

### 📉 Tab 2 — Technical Analysis
- Interactive stock selector
- Candlestick chart with Moving Averages (MA20, MA50, MA200)
- Bollinger Bands
- RSI with overbought/oversold zones
- MACD with histogram
- Automated Buy/Sell signal detection

### 💼 Tab 3 — Portfolio Risk
- Efficient Frontier (3,000 Monte Carlo simulations)
- Optimal portfolio allocation pie chart
- Cumulative returns by stock
- Maximum drawdown comparison

### 📋 Tab 4 — Risk Metrics
- Full risk metrics table per stock
- Annual Return, Volatility, Sharpe Ratio, VaR 95%, Max Drawdown

---

## 🧠 Methodology

### Machine Learning
- **Model**: Gradient Boosting Regressor
- **Target**: Next-day price direction (Up/Down)
- **Features**: Lag returns, rolling volatility, momentum, MA ratios
- **Validation**: Time-series split (no data leakage)
- **Key insight**: AMZN & TSLA achieved >52% direction accuracy (beats random)

### Portfolio Optimization
- **Method**: Markowitz Mean-Variance Optimization
- **Objective**: Maximize Sharpe Ratio
- **Constraints**: Min 5% / Max 40% per asset, weights sum to 100%
- **Simulation**: 5,000 random portfolios (Monte Carlo)

### Technical Indicators
- **RSI** (14-period) — Momentum oscillator
- **MACD** (12/26/9) — Trend following
- **Bollinger Bands** (20-period, 2σ) — Volatility bands
- **Moving Averages** — MA20, MA50, MA200 crossover signals

---

## 👩‍💻 Author

**Salma Mohammed**
- 📧 salmafathelrhman@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/salma-mohammed-3155a61a4/)
- 🐙 [GitHub](https://github.com/Salma-fathi)
- 🌐 [Portfolio](https://salma-fathi.github.io)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).