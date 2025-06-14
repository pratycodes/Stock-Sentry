import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
SYMBOL = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2025-01-19'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 50
STRIDE = 5
THRESHOLD_PERCENTILE = 99

# --- Data Download & Preprocessing ---
def download_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

def add_technical_indicators(data):
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values

    data['MACD'], data['MACD_signal'], _ = ta.MACD(close)
    data['RSI'] = ta.RSI(close)
    data['SMA_20'] = ta.SMA(close, timeperiod=20)
    data['EMA_20'] = ta.EMA(close, timeperiod=20)
    data['ADX'] = ta.ADX(high, low, close)
    
    return data.dropna()

def preprocess_data(data):
    df = data.drop(['Open', 'High', 'Low'], axis=1)
    train_raw, test_raw = train_test_split(df, test_size=0.2, shuffle=False)
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    
    train_df = pd.DataFrame(train_scaled, columns=df.columns, index=train_raw.index)
    test_df = pd.DataFrame(test_scaled, columns=df.columns, index=test_raw.index)
    
    return train_df, test_df, scaler

def create_sequences(data, sequence_length, stride=5):
    sequences = []
    for i in range(0, len(data) - sequence_length, stride):
        seq = data.iloc[i : i + sequence_length].values
        sequences.append(seq)
    return np.array(sequences)

# --- Model Building ---
def build_lstm_autoencoder(sequence_length, num_features):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.RepeatVector(sequence_length),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(num_features))
    ])
    return model

# --- Training ---
def train_model(model, X_train, X_test, epochs=50, batch_size=64):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test),
        callbacks=[early_stopping]
    )
    return history

# --- Evaluation ---
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def detect_anomalies(model, X_train, X_test, test_df, sequence_length, threshold_percentile=99):
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_error = np.mean(np.abs(train_pred - X_train), axis=1)
    test_error = np.mean(np.abs(test_pred - X_test), axis=1)

    threshold = np.percentile(train_error, threshold_percentile)
    anomalies = test_error > threshold
    anomaly_indices = np.where(anomalies)[0]
    anomaly_dates = test_df.index[anomaly_indices + sequence_length]
    
    return test_error, threshold, anomaly_dates

def plot_anomalies(test_error, threshold, anomaly_dates, stock_data):
    plt.figure(figsize=(14, 6))
    plt.plot(test_error, label='Reconstruction Error')
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error for Test Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.scatter(anomaly_dates, stock_data.loc[anomaly_dates]['Close'], color='red', label='Anomalies')
    plt.title("Stock Price with Detected Anomalies")
    plt.legend()
    plt.show()

# --- Main ---
def main():
    stock_data = download_stock_data(SYMBOL, START_DATE, END_DATE)
    stock_data = add_technical_indicators(stock_data)

    train_df, test_df, _ = preprocess_data(stock_data)

    X_train = create_sequences(train_df, SEQUENCE_LENGTH, stride=STRIDE)
    X_test = create_sequences(test_df, SEQUENCE_LENGTH, stride=STRIDE)

    num_features = X_train.shape[2]
    model = build_lstm_autoencoder(SEQUENCE_LENGTH, num_features)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = train_model(model, X_train, X_test, EPOCHS, BATCH_SIZE)

    plot_loss(history)

    test_error, threshold, anomaly_dates = detect_anomalies(
        model, X_train, X_test, test_df, SEQUENCE_LENGTH, THRESHOLD_PERCENTILE)

    print(f"Threshold: {threshold}")
    print(f"Number of anomalies detected: {len(anomaly_dates)}")
    print("Sample anomaly dates:", anomaly_dates[:10])

    plot_anomalies(test_error, threshold, anomaly_dates, stock_data)

if __name__ == "__main__":
    main()
