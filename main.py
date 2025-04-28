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


# Step 1: Data Collection
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


# Add indicators to the dataset
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



def main():
    ticker = input("Enter stock ticker symbol (e.g., AAPL, MSFT, TSLA): ").upper()
    start_date = "2015-01-01"
    end_date = "2023-12-01"

    # Fetch stock data
    data = fetch_stock_data(ticker, start_date, end_date)
    print("Fetched columns:", data.columns.tolist())

    # Add indicators
    data = add_indicators(data)

    # Save to database before preparing features and scaling
    save_to_db(data, ticker)

    # Prepare features and target
    X, y = prepare_features(data)

    if X.empty:
        raise ValueError("No training data available after preprocessing.")

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

    # Model
    model = StockPredictionModel(input_size=X.shape[1])
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Training
    train_losses, test_losses = [], []
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_loss = sum(criterion(model(X_batch), y_batch).item() for X_batch, y_batch in test_loader) / len(test_loader)
            test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Test Loss={test_loss:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title(f"{ticker} Training/Test Loss")
    plt.show()

    # Prediction vs Actual
    with torch.no_grad():
        preds = model(X_test_tensor).squeeze().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(preds, label='Predicted', alpha=0.7)
    plt.title(f"{ticker} Stock Price Prediction")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
