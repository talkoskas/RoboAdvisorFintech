import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas_ta as ta
import pandas as pd
from StockPredictionModel import StockPredictionModel
import sqlite3
import os
import joblib


# --- Configuration ---
DATABASE_PATH = "stocks.db"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

START_DATE = "2015-01-01"
END_DATE = "2023-12-01"


#  --- Functions ---
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError(f"No data fetched for ticker {ticker}.")

    # In case it has multiindex, flatten it (safety net)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Now rename if the ticker name appears in column names
    data.columns = [col.replace(f"{ticker}_", "") for col in data.columns]

    return data


def add_indicators(df):
    print("Adding technical indicators...")

    df['Return'] = df['Close'].pct_change()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add technical indicators
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)

    # Handle Bollinger Bands separately
    bbands = df.ta.bbands(length=20, std=2)

    if bbands is not None:
        # bbands will have BBL, BBM, BBU, BBB
        bbands = bbands.rename(columns={
            'BBL_20_2.0': 'BBL_20_2_0',
            'BBM_20_2.0': 'BBM_20_2_0',
            'BBU_20_2.0': 'BBU_20_2_0',
            'BBB_20_2.0': 'BBB_20_2_0'
        })
        df = pd.concat([df, bbands], axis=1)

    df = df.dropna()
    print("Indicators added successfully.")
    return df


def prepare_features(data):
    features = data[['Open', 'High', 'Low', 'Volume', 'Return',
                     'SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2_0']]
    target = data['Close']
    return features, target


def load_or_initialize_model(ticker, input_size):
    model = StockPredictionModel(input_size=input_size)
    model_path = f"models/{ticker}_model.pth"
    already_trained = False

    if os.path.exists(model_path):
        print(f"Loading existing model for {ticker}...")
        model.load_state_dict(torch.load(model_path))
        already_trained = True
    else:
        print(f"No existing model for {ticker}, initializing new one.")

    return model, already_trained


def train_model(X_train, y_train, X_test, y_test, ticker, already_trained):
    model = StockPredictionModel(input_size=X_train.shape[1])

    model_path = f"{MODEL_DIR}/{ticker}_model.pth"

    # Load previous model if exists
    if already_trained:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model for {ticker}...")

    criterion = nn.HuberLoss()

    if already_trained:
        learning_rate = 0.0001
        num_epochs = 10
    else:
        learning_rate = 0.001
        num_epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 5), gamma=0.5)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_loss = sum(criterion(model(X_batch), y_batch).item() for X_batch, y_batch in test_loader) / len(test_loader)

        train_losses.append(total_loss / len(train_loader))
        test_losses.append(test_loss)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Test Loss={test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model for {ticker}.")

    return model


def evaluate_test_loss(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        test_loss = sum(criterion(model(X_batch), y_batch).item() for X_batch, y_batch in test_loader) / len(test_loader)
    return test_loss


def save_to_db(df, ticker):
    conn = sqlite3.connect("stocks.db")
    df = df.copy()
    df['ticker'] = ticker
    df['date'] = df.index

    # Only keep the necessary columns
    expected_cols = [
        'ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Return', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26_9',
        'BBL_20_2_0', 'BBM_20_2_0', 'BBU_20_2_0', 'BBB_20_2_0'
    ]

    df = df[[col for col in expected_cols if col in df.columns]]  # Safe filtering

    df.to_sql("stock_data", conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f"Saved {len(df)} records for {ticker} to the database.")


def predict_custom_window(ticker, scaler, model):
    print("\n✅ Now you can make custom predictions!")

    custom_start = input("Enter custom start date (YYYY-MM-DD): ").strip()
    custom_end = input("Enter custom end date (YYYY-MM-DD): ").strip()

    df_small = fetch_stock_data(ticker, custom_start, custom_end)
    df_small = add_indicators(df_small)
    X_small, y_small = prepare_features(df_small)

    # Scale using the same scaler
    X_small_scaled = scaler.transform(X_small)

    X_small_tensor = torch.tensor(X_small_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = model(X_small_tensor).squeeze().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_small.values, label="Actual")
    plt.plot(preds, label="Predicted", alpha=0.7)
    plt.title(f"{ticker} Prediction for Custom Window")
    plt.legend()
    plt.show()


# --- MAIN ---

def main():
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT): ").upper()

    # Fetch full training data
    data = fetch_stock_data(ticker, START_DATE, END_DATE)
    print("Fetched columns:", data.columns.tolist())

    data = add_indicators(data)

    save_to_db(data, ticker)

    X, y = prepare_features(data)

    if X.empty:
        raise ValueError("No training data available.")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    already_trained = os.path.exists(f"{MODEL_DIR}/{ticker}_model.pth")

    model = train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, ticker, already_trained)

    # Save the scaler
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    print(f"✅ Saved scaler for {ticker}.")

    # Predict short window
    predict_custom_window(ticker, scaler, model)


if __name__ == '__main__':
    main()
