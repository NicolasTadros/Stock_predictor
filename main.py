from data_fetcher import fetch_stock_data
from predictor import predict

def run_prediction(ticker):
    data = fetch_stock_data(ticker)

    if data is None:
        return {"error": f"'{ticker}' is not a valid stock ticker."}

    predicted, actual = predict(ticker)
    return {
        "predicted": predicted,
        "actual": actual,
        "ticker": ticker.upper()
    }
