import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def get_data_frame():
    df = pd.read_csv("bitcoin_dataset.csv")
    return df

def print_head():
    data_frame = get_data_frame()
    print(data_frame.head())

def convert_df_date_time():
    df = get_data_frame()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def plot_scaled():
    df = convert_df_date_time()
    ft_columns = df.columns.difference(['Trend']) if 'Trend' in df.columns else df.columns
    scaler = MinMaxScaler(feature_range=(0,1))
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
    df = convert_df_date_time()
    btc_prices = df['BTC Close']

    # Plot BTC Close price over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=btc_prices.index, y=btc_prices, label="BTC Close Price")
    plt.title("Bitcoin Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    missing_values = btc_prices.isnull().sum()
    print(missing_values)


if __name__ == "__main__":
    data = get_data_frame()
    print_head()
    pre_processing_data()
    plot_scaled()
