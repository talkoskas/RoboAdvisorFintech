import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas_ta as ta
from StockPredictionModel import StockPredictionModel


# Step 1: Data Collection
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data fetched for ticker {ticker}. Check the ticker symbol or date range.")
    return data


# Add indicators to the dataset
def add_indicators(df):
    print("Adding technical indicators...")
    if df.empty:
        raise ValueError("Error: Dataframe is empty. Cannot calculate indicators.")

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("Error: 'Close' column is missing in the dataset. Cannot calculate indicators.")

    # Add technical indicators
    df['Return'] = df['Close'].pct_change()  # Daily return
    print("\nReturn - \n", df.head(50)['Return'])
    sma10 = ta.sma(df["Close"], length=10)
    print("\nSMA10 - \n", sma10)
    # df['RSI'] = ta.rsi(df['Close'], length=14)  # Relative Strength Index (14-day)
    # df['EMA20'] = ta.ema(df['Close'], length=20)  # Exponential Moving Average (20-day)

    # Drop rows with NaN values caused by indicator calculation
    df = df.dropna()
    print("Indicators added successfully. Dataset preview:")
    print(df.head(50))
    return df


# Step 2: Feature Engineering
def prepare_features(data):
    # Select features including indicators
    features = data[['Open', 'High', 'Low', 'Volume', 'Return', 'SMA20']]
    target = data['Close']  # Predict the 'Close' price

    return features, target


def main():
    ticker = "AAPL"  # Example: Apple Inc.
    start_date = "2015-01-01"
    end_date = "2023-12-01"

    # Step 1: Fetch stock data
    data = fetch_stock_data(ticker, start_date, end_date)

    print(data.head(50))
    # Step 2: Add indicators
    data = add_indicators(data)

    # Step 3: Prepare features and target
    X, y = prepare_features(data)

    if X.shape[0] == 0:
        raise ValueError("Input data X is empty. Check the data pipeline.")
    print(f"Shape of X: {X.shape}")
    print(f"Content of X:\n{X.head()}")

    # Step 4: Normalize features
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Add dimension for output
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Use DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Step 5: Define the Neural Network
    input_size = X_train.shape[1]
    model = StockPredictionModel(input_size=input_size)
    criterion = nn.HuberLoss()  # Huber loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Training loop
    num_epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_losses.append(running_loss / len(train_loader))

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_loss = sum(criterion(model(X_batch), y_batch).item() for X_batch, y_batch in test_loader) / len(
                test_loader)
            test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Test Loss: {test_loss:.4f}")

    # Step 6: Plot Training and Test Losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.show()

    # Step 7: Evaluate Model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze()
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(y_test)), y_test.values, label='Actual Prices')
        plt.plot(range(len(predictions)), predictions.numpy(), label='Predicted Prices', alpha=0.7)
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
