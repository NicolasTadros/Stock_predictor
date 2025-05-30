from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  


@app.route('/api/test')
def test_api():
    return jsonify({"message": "Backend is working!", "status": "success"})

def fetch_stock_data(ticker, period='5y', interval='1d'):
    try:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            print(f"No data returned for {ticker}")
            return None, None
        print(f"Successfully fetched {len(data)} data points for {ticker}")
        
        return data['Close'].values.reshape(-1,1), data.index
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None

def prepare_data(data, n_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def build_and_train_model(X, y, epochs=50):
    try:
        print(f"Building model with {len(X)} training samples...")
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        print("Model training completed")
        return model
    except Exception as e:
        print(f"Error building/training model: {e}")
        return None

def predict_future_prices(model, data, n_steps=10, n_preds=15):
    try:
        print("Generating predictions...")
        preds = []
        input_seq = data[-n_steps:].copy()
        for _ in range(n_preds):
            pred = model.predict(input_seq.reshape(1, n_steps, 1), verbose=0)
            preds.append(pred[0,0])
            input_seq = np.append(input_seq[1:], pred, axis=0)
        print(f"Generated {len(preds)} predictions")
        return np.array(preds)
    except Exception as e:
        print(f"Error predicting prices: {e}")
        return None

def get_next_business_days(start_date, num_days):
    """Generate next business days (excluding weekends)"""
    business_days = []
    current_date = start_date
    while len(business_days) < num_days:
        current_date += timedelta(days=1)
        
        if current_date.weekday() < 5:
            business_days.append(current_date)
    return business_days

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Predictor</title>
    </head>
    <body>
        <h1>Stock Price Predictor Backend</h1>
        <p>Backend is running successfully!</p>
        <p>Test the API: <a href="/api/test">/api/test</a></p>
        <p>Get stock data: <a href="/api/data?ticker=AAPL">/api/data?ticker=AAPL</a></p>
        <p>Serve your HTML frontend from the /static folder</p>
    </body>
    </html>
    """

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return app.send_static_file(filename)

@app.route('/api/data')
def api_data():
    print(f"\n=== API REQUEST RECEIVED ===")
    print(f"Request args: {request.args}")
    
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            print("ERROR: No ticker provided")
            return jsonify({"error": "Ticker parameter is required"}), 400
        
        ticker = ticker.upper().strip()
        print(f"Processing ticker: {ticker}")

        
        close_prices, actual_dates = fetch_stock_data(ticker, period='2y')
        if close_prices is None:
            print(f"ERROR: No data found for {ticker}")
            return jsonify({"error": f"No data found for ticker '{ticker}'. Please verify the ticker symbol."}), 404
        
        print(f"Data points available: {len(close_prices)}")
        
        if len(close_prices) < 50:
            print(f"ERROR: Insufficient data for {ticker}")
            return jsonify({"error": f"Insufficient data for ticker '{ticker}'. Need at least 50 data points."}), 400
        
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        
        n_steps = 10  
        X, y = prepare_data(scaled_data, n_steps)
        
        if len(X) == 0:
            print("ERROR: Unable to prepare training data")
            return jsonify({"error": f"Unable to prepare training data for ticker '{ticker}'"}), 400
        
        
        model = build_and_train_model(X, y, epochs=50)
        if model is None:
            print("ERROR: Model training failed")
            return jsonify({"error": "Failed to build or train the model"}), 500
        
        
        predicted_scaled = predict_future_prices(model, scaled_data, n_steps, n_preds=5)
        if predicted_scaled is None:
            print("ERROR: Prediction generation failed")
            return jsonify({"error": "Failed to generate predictions"}), 500
        
        predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1,1)).flatten()
        
        
        last_date = actual_dates[-1].to_pydatetime()
        predicted_dates = get_next_business_days(last_date, 5)
        
        
        num_actual_points = min(84, len(close_prices))
        recent_actual_dates = actual_dates[-num_actual_points:]
        recent_actual_prices = close_prices[-num_actual_points:]
        
        actual_data = [
            {
                "date": date.strftime("%Y-%m-%d"), 
                "price": float(price[0])
            } 
            for date, price in zip(recent_actual_dates, recent_actual_prices)
        ]
        
        predicted_data = [
            {
                "date": date.strftime("%Y-%m-%d"), 
                "price": float(price)
            } 
            for date, price in zip(predicted_dates, predicted_prices)
        ]
        
        response = {
            "ticker": ticker,
            "actual": actual_data,
            "predicted": predicted_data
        }
        
        print(f"SUCCESS: Returning data for {ticker}")
        print(f"Actual data points: {len(actual_data)}")
        print(f"Predicted data points: {len(predicted_data)}")
        return jsonify(response)
        
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Backend will be available at: http://localhost:5000")
    print("Test endpoint: http://localhost:5000/api/test")
    print("API endpoint: http://localhost:5000/api/data?ticker=AAPL")
    print("Frontend should be served from: http://localhost:5000/static/index.html")
    app.run(debug=True, host='0.0.0.0', port=5000)