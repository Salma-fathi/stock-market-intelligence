import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/stocks_features.csv")
df['Date'] = pd.to_datetime(df['Date'])
print(f"✅ Loaded data: {df.shape}")

def create_features(stock_df):
    d = stock_df.copy().sort_values('Date')
    
    # Predict NEXT day return (not price!)
    d['Target'] = d['Close'].pct_change(1).shift(-1)
    
    # Lag returns
    for lag in [1, 2, 3, 5, 10]:
        d[f'Return_lag{lag}'] = d['Close'].pct_change(1).shift(lag)
        d[f'Volume_lag{lag}'] = d['Volume'].pct_change(1).shift(lag)
    
    # Rolling stats (no leakage)
    d['Volatility_5']  = d['Close'].pct_change().shift(1).rolling(5).std()
    d['Volatility_20'] = d['Close'].pct_change().shift(1).rolling(20).std()
    d['Momentum_5']    = d['Close'].shift(1) / d['Close'].shift(6) - 1
    d['Momentum_20']   = d['Close'].shift(1) / d['Close'].shift(21) - 1
    d['MA_ratio']      = d['Close'].shift(1).rolling(5).mean() / \
                         d['Close'].shift(1).rolling(20).mean() - 1
    d['Range_pct']     = (d['High'].shift(1) - d['Low'].shift(1)) / \
                          d['Close'].shift(1)
    
    return d.dropna()

STOCKS   = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
FEATURES = [
    'Return_lag1','Return_lag2','Return_lag3','Return_lag5','Return_lag10',
    'Volume_lag1','Volume_lag5',
    'Volatility_5','Volatility_20',
    'Momentum_5','Momentum_20',
    'MA_ratio','Range_pct'
]

results = {}
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Stock Return Prediction - Gradient Boosting', 
             fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, ticker in enumerate(STOCKS):
    print(f"\n📈 Training model for {ticker}...")
    
    stock = df[df['Ticker'] == ticker].copy()
    stock = create_features(stock)
    
    X = stock[FEATURES]
    y = stock['Target']
    
    # Time-based split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    # Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    # Metrics
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Direction accuracy (up/down prediction)
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100
    
    results[ticker] = {
        'R²': round(r2, 4),
        'MAE': round(mae, 4),
        'Direction Accuracy%': round(direction_acc, 2)
    }
    print(f"  R²: {r2:.4f} | MAE: {mae:.4f} | Direction: {direction_acc:.1f}%")
    
    # Reconstruct price from predicted returns
    last_prices  = stock['Close'].iloc[split:].values
    pred_prices  = last_prices * (1 + y_pred)
    actual_prices = last_prices * (1 + y_test.values)
    
    ax = axes[i]
    ax.plot(actual_prices[-100:], label='Actual', color='blue', linewidth=2)
    ax.plot(pred_prices[-100:],  label='Predicted', color='red', 
            linestyle='--', linewidth=2)
    ax.set_title(f'{ticker} | Dir.Acc: {direction_acc:.1f}% | R²: {r2:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('data/predictions_v2.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*55)
print("📊 FINAL MODEL RESULTS")
print("="*55)
results_df = pd.DataFrame(results).T
print(results_df)
results_df.to_csv("data/model_results_v2.csv")
print("\n✅ Done! Key metric is Direction Accuracy (>50% beats random!)")