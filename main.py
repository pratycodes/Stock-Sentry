import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Config ---
SYMBOL = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2025-01-19'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 50
STRIDE = 5
THRESHOLD_PERCENTILE = 99


# --- Device ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)


# --- Data Download & Preprocessing ---
def download_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data


def add_technical_indicators(data,symbol):
    close_prices = data[('Close', symbol)].astype(np.float64).to_numpy()
    high_prices = data[('High', symbol)].astype(np.float64).to_numpy()
    low_prices = data[('Low', symbol)].astype(np.float64).to_numpy()

    data['MACD'], data['MACD_signal'], _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    data['RSI'] = ta.RSI(close_prices, timeperiod=14)
    data['SMA_20'] = ta.SMA(close_prices, timeperiod=20)
    data['EMA_20'] = ta.EMA(close_prices, timeperiod=20)
    data['ADX'] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)

    data.dropna(inplace=True)
    return data


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


# --- PyTorch Model ---
class BiLSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, num_feat):
        super().__init__()
        self.seq_len = seq_len
        self.num_feat = num_feat

        self.encoder_lstm1 = nn.LSTM(num_feat, 64, batch_first=True, bidirectional=True)
        self.encoder_ln1 = nn.LayerNorm(64*2)
        self.encoder_dropout1 = nn.Dropout(0.3)

        self.encoder_lstm2 = nn.LSTM(64*2, 32, batch_first=True, bidirectional=True)
        self.encoder_ln2 = nn.LayerNorm(32*2)

        self.decoder_lstm1 = nn.LSTM(32*2, 32, batch_first=True)
        self.decoder_ln1 = nn.LayerNorm(32)
        self.decoder_dropout1 = nn.Dropout(0.3)

        self.decoder_lstm2 = nn.LSTM(32, 64, batch_first=True)
        self.decoder_ln2 = nn.LayerNorm(64)

        self.output_layer = nn.Linear(64, num_feat)

    def forward(self, x):
        batch_size = x.size(0)
        x, _ = self.encoder_lstm1(x)
        x = self.encoder_ln1(x)
        x = self.encoder_dropout1(x)

        x, (h_n, _) = self.encoder_lstm2(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat((h_forward, h_backward), dim=1)
        x = self.encoder_ln2(h)

        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        x, _ = self.decoder_lstm1(x)
        x = self.decoder_ln1(x)
        x = self.decoder_dropout1(x)

        x, _ = self.decoder_lstm2(x)
        x = self.decoder_ln2(x)

        x = self.output_layer(x)
        return x


def add_noise(x, noise_std=0.01):
    noise = torch.randn_like(x) * noise_std
    return x + noise


# --- Training Loop ---
def train(autoencoder, train_loader, val_loader, epochs=50, patience=5, lr=1e-4):
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses_history = []
    val_losses_history = []

    for epoch in range(epochs):
        autoencoder.train()
        train_losses = []
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            noisy_x = add_noise(batch_x)

            optimizer.zero_grad()
            outputs = autoencoder(noisy_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        autoencoder.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                outputs = autoencoder(batch_x)
                loss = criterion(outputs, batch_x)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)

        lr_scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = autoencoder.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                autoencoder.load_state_dict(best_weights)
                break

    return train_losses_history, val_losses_history


# --- Anomaly Detection ---
def detect_anomalies(autoencoder, X_train, X_test, test_df, threshold_percentile=99):
    autoencoder.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        train_reconstructed = autoencoder(X_train_tensor).cpu().numpy()
        test_reconstructed = autoencoder(X_test_tensor).cpu().numpy()

    train_error = np.mean(np.abs(train_reconstructed - X_train), axis=(1, 2))
    test_error = np.mean(np.abs(test_reconstructed - X_test), axis=(1, 2))

    threshold = np.percentile(train_error, threshold_percentile)
    anomalies = test_error > threshold

    anomaly_indices = np.where(anomalies)[0]
    anomaly_dates = test_df.index[anomaly_indices + SEQUENCE_LENGTH]

    return test_error, threshold, anomaly_dates


# --- Plotting ---
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_anomalies(test_error, threshold, anomaly_dates, data):
    plt.figure(figsize=(14, 6))
    plt.plot(test_error, label='Reconstruction Error')
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error for Test Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(data['Close'], label='Price')
    plt.scatter(anomaly_dates, data.loc[anomaly_dates]['Close'], color='red', s=50, label='Anomalies')
    plt.title('Price Chart with Detected Anomalies')
    plt.legend()
    plt.show()


# --- Main ---
def main():
    data = download_stock_data(SYMBOL, START_DATE, END_DATE)
    data = add_technical_indicators(data,SYMBOL)

    train_df, test_df, scaler = preprocess_data(data)

    X_train = create_sequences(train_df, SEQUENCE_LENGTH, stride=STRIDE)
    X_test = create_sequences(test_df, SEQUENCE_LENGTH, stride=STRIDE)

    num_features = X_train.shape[2]
    autoencoder = BiLSTMAutoencoder(SEQUENCE_LENGTH, num_features).to(device)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_losses, val_losses = train(autoencoder, train_loader, val_loader, epochs=EPOCHS, patience=5)

    plot_loss(train_losses, val_losses)

    test_error, threshold, anomaly_dates = detect_anomalies(autoencoder, X_train, X_test, test_df, THRESHOLD_PERCENTILE)

    print(f"Anomaly detection threshold: {threshold}")
    print(f"Number of anomalies detected: {len(anomaly_dates)}")
    print("Sample anomaly dates:", anomaly_dates[:10])

    plot_anomalies(test_error, threshold, anomaly_dates, data)


if __name__ == "__main__":
    main()
