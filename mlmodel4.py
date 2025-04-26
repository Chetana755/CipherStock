# Install required packages
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fetch stock data - using yfinance directly (simpler approach)
def get_stock_data(ticker, start_date, end_date=None):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Add technical indicators
def add_technical_indicators(df):
    # Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length, 0])  # Predicting Close price
    return np.array(X), np.array(y)

# Get latest data including today
ticker = 'TSLA'
start_date = datetime.datetime(2015, 1, 1)  # Using more recent data for better results
print(f"Fetching {ticker} stock data from {start_date} to today...")
stock_data = get_stock_data(ticker, start_date)

# Add technical indicators
print("Adding technical indicators...")
stock_data = add_technical_indicators(stock_data)

# Normalize data
print("Normalizing data...")
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(stock_data)

# Create sequences
SEQ_LENGTH = 90  # Using 90 days of history
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Use all data for training to maximize prediction accuracy
X_train, y_train = X, y

# Build and train model
print("Building and training model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.legend()
plt.title(f'{ticker} Model Training Loss')
plt.show()

# Prepare data for future predictions
num_prediction_days = 30  # Predict 30 days into the future

# Get the last sequence from our data
last_sequence = scaled_data[-SEQ_LENGTH:]
last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

# Make future predictions
future_predictions = []
current_sequence = last_sequence.copy()

for _ in range(num_prediction_days):
    # Get prediction for next day
    next_day_price = model.predict(current_sequence)[0][0]
    future_predictions.append(next_day_price)
    
    # Update the sequence by removing first element and adding the prediction
    pred_full_features = np.zeros(scaled_data.shape[1])
    pred_full_features[0] = next_day_price  # Set the predicted close price
    
    # Roll the sequence forward (remove first element, add new prediction)
    current_sequence = np.append(current_sequence[:,1:,:], 
                                np.reshape(pred_full_features, (1, 1, scaled_data.shape[1])), 
                                axis=1)

# Create date range for future predictions
last_date = stock_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_prediction_days, freq='B')

# Convert predictions back to original scale
future_predictions_array = np.array(future_predictions).reshape(-1, 1)
# Create array with zeros for the other features
future_zeros = np.zeros((len(future_predictions), scaled_data.shape[1]-1))
# Combine to create the right shape
future_full_features = np.concatenate((future_predictions_array, future_zeros), axis=1)
# Inverse transform
future_prices = scaler.inverse_transform(future_full_features)[:,0]

# Create DataFrame for future predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': future_prices
})
future_df.set_index('Date', inplace=True)

# Plot historical data and future predictions
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index[-90:], stock_data['Close'][-90:], label='Historical Close Price', color='blue')
plt.plot(future_df.index, future_df['Predicted_Close'], label='Predicted Close Price', color='red')
plt.title(f'{ticker} Stock Price Prediction for the Next {num_prediction_days} Days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Display the predictions
print("\n === Future Price Predictions ===")
print(future_df)

# Calculate last known price and predicted future prices
latest_price = stock_data['Close'][-1]
future_price_end = future_df['Predicted_Close'][-1]
price_change = ((future_price_end - latest_price) / latest_price) * 100

print(f"\nLatest {ticker} closing price (as of {stock_data.index[-1].date()}): ${latest_price:.2f}")
print(f"Predicted {ticker} price on {future_df.index[-1].date()}: ${future_price_end:.2f}")
print(f"Predicted change: {price_change:.2f}%")
print("\nNOTE: These predictions are for educational purposes only. Stock market prediction is inherently uncertain.")