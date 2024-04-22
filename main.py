# Create a virtual environment
# python -m venv myenv

# Activate the virtual environment
# On Windows
# myenv\Scripts\activate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to fetch cryptocurrency data
def fetch_crypto_data(fsym, tsym, limit=500):
    api_key = 'c1ceb6987448327a75b5e515e45e1b1c355951bd0386ce3f64979426071d3f70'
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    parameters = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'api_key': api_key
    }
    response = requests.get(url, params=parameters)
    data = response.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

# Function to create sequences needed for LSTM input
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to validate cryptocurrency symbol
def validate_crypto_symbol(symbol):
    # List of known cryptocurrency symbols
    known_crypto_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'USDT', 'BNB', 'SOL', 'XRP', 'USDC', 'ADA', 'AVAX', 'DOGE', 'SHIB']  
    return symbol.upper() in known_crypto_symbols

# Function to validate target currency symbol
def validate_currency_symbol(symbol):
    # List of known currency symbols
    known_currency_symbols = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF', 'AUD', 'NZD', 'CNY', 'HKD',
                              'SGD', 'KRW', 'INR', 'BRL', 'RUB', 'ZAR', 'MXN', 'TRY', 'NOK', 'SEK'] 
    return symbol.upper() in known_currency_symbols

# Main script
if __name__ == "__main__":
    # User input for the cryptocurrency and target currency
    while True:
        fsym = input("Enter the symbol of the base cryptocurrency (e.g., 'BTC' for Bitcoin, 'ETH' for Ethereum, 'XRP' for Ripple, 'LTC' for Litecoin): ").strip().upper()
        if validate_crypto_symbol(fsym):
            break
        else:
            print("Invalid cryptocurrency symbol. Please enter a valid symbol.")

    while True:
        tsym = input("Enter the symbol of the target currency (e.g., 'USD'): ").strip().upper()
        if validate_currency_symbol(tsym):
            break
        else:
            print("Invalid currency symbol. Please enter a valid symbol.")

    while True:
        try:
            days_into_future = int(input("Enter how many days into the future you want to predict: ").strip())
            if days_into_future > 0:
                break
            else:
                print("Please enter a positive integer for the number of days into the future.")
        except ValueError:
            print("Please enter a valid integer.")

    days = 500

    # Fetch and prepare data
    data = fetch_crypto_data(fsym, tsym, days)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['scaled'] = scaler.fit_transform(data['close'].values.reshape(-1,1))
    
    sequence_length = 60
    x, y = create_sequences(data['scaled'], sequence_length)

    # Reshape input for LSTM
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Split data into train and test sets
    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Prepare data for future prediction
    last_sequence = data['scaled'].values[-sequence_length:].reshape(1, sequence_length, 1)
    future_dates = pd.date_range(data['time'].iloc[-1], periods=days_into_future+1, freq='D')[1:]
    future_predictions = []
    current_sequence = last_sequence.copy()

    for i in range(days_into_future):
        next_step = model.predict(current_sequence)
        future_predictions.append(next_step[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_step[0, 0]

    # Inverse transform to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plotting the results
    plt.figure(figsize=(14, 5))
    plt.plot(data['time'], data['close'], label='Historical Prices')
    plt.plot(future_dates, future_predictions, color='red', label='Predicted Future Prices')
    plt.title(f'{fsym} Price Prediction for the Next {days_into_future} Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
