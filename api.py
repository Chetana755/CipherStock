# === Flask API to serve ML predictions ===
# Save as api.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to saved model and scaler
MODEL_PATH = 'stock_lstm_model.h5'
SCALER_PATH = 'stock_scaler.pkl'

# === Functions ===
def get_stock_data(ticker, start_date, end_date=None):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def add_technical_indicators(df):
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
    return np.array(X)

@app.route('/api/predict/<ticker>', methods=['GET'])
def predict_stock(ticker):
    try:
        # Configuration
        SEQ_LENGTH = 90
        num_prediction_days = 30
        
        # Get historical data
        start_date = datetime.datetime.now() - datetime.timedelta(days=365*3)  # 3 years of data
        end_date = datetime.datetime.now()
        
        # Get and process data
        stock_data = get_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 404
            
        stock_data = add_technical_indicators(stock_data)
        
        # Check if model exists, if not, return mock data
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            # Generate mock prediction (you can implement this based on your generateMockPrediction JS function)
            return jsonify({'error': 'Model not found, would return mock data in production'}), 500
        
        # Load model and scaler
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare data
        # Scale data
        scaled_data = scaler.transform(stock_data)
        
        # Future prediction
        last_sequence = scaled_data[-SEQ_LENGTH:]
        last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(num_prediction_days):
            next_day_price = model.predict(current_sequence)[0][0]
            future_predictions.append(next_day_price)
            
            new_feature = np.zeros(scaled_data.shape[1])
            new_feature[0] = next_day_price
            
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        np.reshape(new_feature, (1, 1, scaled_data.shape[1])), 
                                        axis=1)
        
        # Inverse transform future predictions
        future_predictions_array = np.array(future_predictions).reshape(-1, 1)
        future_zeros = np.zeros((len(future_predictions), scaled_data.shape[1] - 1))
        future_full = np.concatenate((future_predictions_array, future_zeros), axis=1)
        future_prices = scaler.inverse_transform(future_full)[:, 0]
        
        # Date index
        last_date = stock_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_prediction_days, freq='B')
        
        # Prepare historical dates and prices
        historical_dates = stock_data.index[-90:].strftime('%Y-%m-%d').tolist()
        historical_prices = stock_data['Close'][-90:].tolist()
        
        # Calculate model metrics
        # For simplicity, we're returning fixed metrics
        # In a real scenario, you would calculate these based on test data
        metrics = {
            'mae': 2.45,
            'mse': 9.82,
            'rmse': 3.13,
            'r2': 0.87
        }
        
        # Prepare response
        response = {
            'symbol': ticker,
            'historicalDates': historical_dates,
            'historicalPrices': historical_prices,
            'futureDates': future_dates.strftime('%Y-%m-%d').tolist(),
            'predictedPrices': future_prices.tolist(),
            'metrics': metrics,
            'latestPrice': historical_prices[-1],
            'predictedEndPrice': future_prices[-1],
            'percentChange': ((future_prices[-1] - historical_prices[-1]) / historical_prices[-1]) * 100
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to train or retrain the model
@app.route('/api/train/<ticker>', methods=['POST'])
def train_model(ticker):
    # Implementation for training the model
    # This would be a more complex endpoint that would run your training code
    # and save the model and scaler to disk
    return jsonify({'status': 'Not implemented in this example'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)