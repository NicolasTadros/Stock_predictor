import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty:
        return None  

    hist.to_csv(f"data/{ticker}_data.csv")
    return hist
