import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def get_data_frame():
    """
    Load the Bitcoin dataset from a CSV file named 'bitcoin_dataset.csv'.

    This function reads the dataset into a Pandas DataFrame, which is a
    two-dimensional size-mutable, potentially heterogeneous tabular data structure
    with labeled axes.

    Returns:
        pd.DataFrame: A DataFrame containing the Bitcoin dataset.
    """
    df = pd.read_csv("bitcoin_dataset.csv")
    return df

def print_head():
    """
    Print the first few rows of the Bitcoin dataset to inspect its structure.

    This function is useful for quickly checking the initial entries of the
    dataset to understand its columns and data types. It calls the `get_data_frame`
    function to retrieve the DataFrame and then prints the first five rows using
    the `head()` method.
    """
    data_frame = get_data_frame()
    print(data_frame.head())

def convert_df_date_time():
    """
    Convert the 'Date' column in the DataFrame to datetime format and set it as the index.

    This function ensures that the 'Date' column is properly formatted for time series
    analysis. It converts the 'Date' column to datetime objects and sets it as the
    DataFrame's index, enabling time-based indexing and operations.

    Returns:
        pd.DataFrame: DataFrame with 'Date' column converted to datetime and set as index.
    """
    df = get_data_frame()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def plot_scaled():
    """
    Plot the scaled features of the Bitcoin dataset over time.

    This function scales the feature columns (excluding 'Trend' if present) using
    Min-Max scaling to normalize the data between 0 and 1. It then plots these
    scaled values over time to visualize trends and patterns in the data.

    The Min-Max scaler transforms features by scaling each feature to a given range,
    typically between zero and one. This is particularly useful when features have
    varying scales and need to be normalized for analysis.
    """
    df = convert_df_date_time()
    ft_columns = df.columns.difference(['Trend']) if 'Trend' in df.columns else df.columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_frame_scaled = df.copy()
    data_frame_scaled[ft_columns] = scaler.fit_transform(data_frame_scaled[ft_columns])

    plt.figure(figsize=(12, 6))
    for col in ft_columns:
        plt.plot(data_frame_scaled.index, data_frame_scaled[col], label=col, alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Scaled Value (0-1)")
    plt.title("Normalized Bitcoin and Market Data Over Time")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

def pre_processing_data():
    """
    Preprocess the Bitcoin dataset by plotting the BTC close price and checking for missing values.

    This function performs initial data exploration and preprocessing steps. It plots
    the Bitcoin close price over time using a line plot to visualize historical price
    movements. Additionally, it checks for missing values in the 'BTC Close' column
    to ensure data integrity before modeling.
    """
    df = convert_df_date_time()
    btc_prices = df['BTC Close']

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=btc_prices.index, y=btc_prices, label="BTC Close Price")
    plt.title("Bitcoin Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    missing_values = btc_prices.isnull().sum()
    print("Missing values in BTC Close:", missing_values)

def train_arima_model():
    """
    Train an ARIMA model on the BTC Close price and evaluate its performance.

    This function trains an ARIMA (AutoRegressive Integrated Moving Average) model
    on the Bitcoin close price data. It performs a train/test split (80% train, 20% test)
    and evaluates the model's performance using Mean Squared Error (MSE). The function
    also plots the actual vs. predicted BTC close prices to visually assess the model's
    performance.

    ARIMA models are widely used for time series forecasting and can capture trends,
    seasonality, and noise in the data. The order of the ARIMA model (p, d, q) can be
    adjusted to optimize performance.
    """
    df = convert_df_date_time()
    btc_prices = df['BTC Close'].dropna()

    train_size = int(len(btc_prices) * 0.8)
    train_data = btc_prices.iloc[:train_size]
    test_data = btc_prices.iloc[train_size:]

    model = ARIMA(train_data, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())

    forecast_steps = len(test_data)
    predictions = model_fit.forecast(steps=forecast_steps)

    mse = mean_squared_error(test_data, predictions)
    print(f"\nARIMA Test MSE: {mse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label="Train (BTC Close)")
    plt.plot(test_data.index, test_data, label="Actual Test (BTC Close)")
    plt.plot(test_data.index, predictions, label="ARIMA Forecast", linestyle='--')
    plt.title("ARIMA Model - BTC Close Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    """
    Main execution block to run the data processing and modeling pipeline.

    This block calls the functions to load the data, print the first few rows,
    preprocess the data, plot the scaled features, and train the ARIMA model.
    It serves as the entry point for the script, ensuring that the functions are
    executed in the correct order for a complete analysis.
    """
    data = get_data_frame()
    print_head()
    pre_processing_data()
    plot_scaled()
    train_arima_model()
