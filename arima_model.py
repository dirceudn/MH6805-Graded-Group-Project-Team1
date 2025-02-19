import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


class BitcoinARIMAAnalysis:
    """
    A class to analyze Bitcoin price trends using ARIMA modeling.
    This class includes methods for loading, preprocessing, and modeling Bitcoin time-series data.
    """

    def __init__(self, file_path="bitcoin_dataset.csv"):
        """
        Initializes the BitcoinARIMAAnalysis class and loads the dataset.

        :param file_path: Path to the CSV file containing the Bitcoin dataset.
        """
        self.file_path = file_path
        self.df = self.load_data()
        self.process_data()

    def load_data(self):
        """
        Loads the Bitcoin dataset from a CSV file.

        :return: A pandas DataFrame containing the dataset.
        """
        df = pd.read_csv(self.file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def process_data(self):
        """
        Preprocesses the dataset by ensuring proper datetime format and handling missing values.
        """
        self.df.dropna(subset=['BTC Close'], inplace=True)

    def plot_scaled_features(self):
        """
        Scales and plots all features in the dataset.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_df = self.df.copy()
        scaled_df[self.df.columns] = scaler.fit_transform(self.df[self.df.columns])

        plt.figure(figsize=(12, 6))
        for col in scaled_df.columns:
            plt.plot(scaled_df.index, scaled_df[col], label=col, alpha=0.7)

        plt.xlabel("Date")
        plt.ylabel("Scaled Value (0-1)")
        plt.title("Normalized Bitcoin and Market Data Over Time")
        plt.legend()
        plt.grid()
        plt.show()

    def train_arima_model(self, order=(1, 1, 1)):
        """
        Trains an ARIMA model to predict Bitcoin closing prices.

        :param order: Tuple representing ARIMA model order (p, d, q).
        :return: Trained ARIMA model and its forecast.
        """
        btc_prices = self.df['BTC Close']
        train_size = int(len(btc_prices) * 0.8)
        train_data, test_data = btc_prices.iloc[:train_size], btc_prices.iloc[train_size:]

        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        print(model_fit.summary())

        forecast = model_fit.forecast(steps=len(test_data))
        mse = mean_squared_error(test_data, forecast)
        print(f"\nARIMA Test MSE: {mse:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label="Train Data")
        plt.plot(test_data.index, test_data, label="Actual Test Data")
        plt.plot(test_data.index, forecast, label="ARIMA Forecast", linestyle='--')
        plt.title("ARIMA Model - BTC Close Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

        return model_fit, forecast


if __name__ == "__main__":
    arima_analysis = BitcoinARIMAAnalysis()
    arima_analysis.plot_scaled_features()
    arima_model, arima_forecast = arima_analysis.train_arima_model()
