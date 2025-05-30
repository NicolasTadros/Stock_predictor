import matplotlib.pyplot as plt
import matplotlib.dates as mdates  

def plot_closing_price(data, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price')
    plt.title(f"{ticker} Stock Price (Closing)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)  

    plt.tight_layout()
    plt.show()

def plot_predictions(data, predictions, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Actual Close Price')
    plt.plot(data.index[1:], predictions, label='Predicted Close Price', linestyle='--')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
