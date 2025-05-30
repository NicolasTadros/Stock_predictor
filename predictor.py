import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

def predict(ticker):
    ticker = ticker.upper()
    path = f"data/{ticker}_data.csv"

    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)
    data = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_test = []
    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i - 60:i])

    x_test = np.array(x_test)

    model = load_model('lstm_model.h5')
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    actual = scaler.inverse_transform(scaled_data[60:])

    return predictions.flatten().tolist(), actual.flatten().tolist()
