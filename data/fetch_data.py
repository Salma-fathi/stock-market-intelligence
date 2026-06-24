import yfinance as yf
import pandas as pd
import os

# Define stocks to analyze
STOCKS = {
    "AAPL": "Apple",
    "GOOGL": "Google", 
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "TSLA": "Tesla"
}

def fetch_stock_data(ticker, start="2020-01-01", end="2024-12-31"):
    """Fetch historical stock data"""
    print(f"Fetching {ticker} data...")
    df = yf.download(ticker, start=start, end=end)
    df["Ticker"] = ticker
    df["Company"] = STOCKS[ticker]
    return df

def fetch_all_stocks():
    """Fetch all stocks and save to CSV"""
    all_data = []
    
    for ticker, company in STOCKS.items():
        df = fetch_stock_data(ticker)
        all_data.append(df)
    
    # Combine all stocks
    combined = pd.concat(all_data)
    combined.reset_index(inplace=True)
    
    # Save to CSV
    combined.to_csv("data/stocks_raw.csv", index=False)
    print(f"✅ Data saved! Shape: {combined.shape}")
    print(combined.head())
    return combined

if __name__ == "__main__":
    df = fetch_all_stocks()