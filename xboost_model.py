import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class BitcoinAnalysis:
    """
    A class for analyzing Bitcoin data in relation to NASDAQ and Gold market data.

    Attributes:
        gold_data (pd.DataFrame): DataFrame containing gold market data.
        nasdaq_data (pd.DataFrame): DataFrame containing NASDAQ market data.
        btc_data (pd.DataFrame): DataFrame containing Bitcoin market data.
        file_path (str): Path to the dataset file.
        df (pd.DataFrame): DataFrame containing the loaded dataset.
    """

    def __init__(self, file_path="bitcoin_dataset.csv"):
        """
        Initializes the BitcoinAnalysis class with the specified file path.

        Args:
            file_path (str): The path to the dataset file. Defaults to "bitcoin_dataset.csv".
        """
        self.gold_data = None
        self.nasdaq_data = None
        self.btc_data = None
        self.file_path = file_path
        self.df = self.load_data()
        self.process_data()

    def load_data(self):
        """
        Loads the dataset from the specified file path and preprocesses it.

        Returns:
            pd.DataFrame: The preprocessed DataFrame containing the dataset.
        """
        df = pd.read_csv(self.file_path)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def process_data(self):
        """
        Processes the loaded dataset to extract Bitcoin, NASDAQ, and Gold market data.
        """
        btc_columns = ['BTC Open', 'BTC High', 'BTC Low', 'BTC Close', 'BTC Volume']
        self.btc_data = self.df[btc_columns].copy()
        self.btc_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        nasdaq_columns = ['NASDAQ Open', 'NASDAQ High', 'NASDAQ Low', 'NASDAQ Close', 'NASDAQ Volume']
        self.nasdaq_data = self.df[nasdaq_columns].copy()
        self.nasdaq_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        gold_columns = ['Gold Open', 'Gold High', 'Gold Low', 'Gold Close', 'Gold Volume']
        self.gold_data = self.df[gold_columns].copy()
        self.gold_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    @staticmethod
    def plot_candlestick(data, title):
        """
        Plots a candlestick chart for the given data.

        Args:
            data (pd.DataFrame): DataFrame containing the market data.
            title (str): Title of the plot.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(title)

        dates = data.index
        x = np.arange(len(dates))

        opens = data['Open']
        highs = data['High']
        lows = data['Low']
        closes = data['Close']

        colors = ['g' if close >= open else 'r' for open, close in zip(opens, closes)]

        width = 0.6
        ax1.vlines(x, lows, highs, color='black', linewidth=1)
        ax1.bar(x, closes - opens, width, bottom=opens, color=colors)

        ax2.bar(x, data['Volume'], width, color=colors)
        ax2.set_ylabel('Volume')

        ax1.set_xticklabels([])
        ax2.set_xticks(x[::len(x) // 10])  # Show every nth date
        ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates[::len(x) // 10]], rotation=45)

        # Set labels and grid
        ax1.set_ylabel('Price')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_all_candlesticks(self):
        """
        Plots candlestick charts for Bitcoin, NASDAQ, and Gold market data.
        """
        self.plot_candlestick(self.btc_data, 'Bitcoin Candlestick Chart')
        self.plot_candlestick(self.nasdaq_data, 'NASDAQ Candlestick Chart')
        self.plot_candlestick(self.gold_data, 'Gold Candlestick Chart')

    def plot_correlation_heatmap(self):
        """
        Plots a heatmap of the correlation matrix for the dataset.
        """
        plt.figure(figsize=(10, 6))
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def prepare_ml_data(self):
        """
        Prepares the data for machine learning models by scaling and splitting it into training and testing sets.

        Returns:
            tuple: A tuple containing the scaled training and testing data, target values, scaler, and feature columns.
        """
        target_col = 'BTC Close'
        feature_cols = ['NASDAQ Open', 'NASDAQ High', 'NASDAQ Low', 'NASDAQ Volume',
                        'Gold Open', 'Gold High', 'Gold Low', 'Gold Volume']

        # Ensure only existing columns are used
        feature_cols = [col for col in feature_cols if col in self.df.columns]

        data = self.df[feature_cols]
        target = self.df[target_col]

        # Shifting features by one day
        data_shifted = data.shift(1).dropna()
        target_shifted = target.loc[data_shifted.index]

        # Splitting into training and testing data
        split_ratio = 0.8
        split_index = int(len(data_shifted) * split_ratio)

        X_train = data_shifted.iloc[:split_index]
        X_test = data_shifted.iloc[split_index:]
        y_train = target_shifted.iloc[:split_index]
        y_test = target_shifted.iloc[split_index:]

        # Standardizing the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_y, feature_cols

    def train_xgboost(self):
        """
        Trains an XGBoost model on the prepared data and evaluates its performance.

        Returns:
            tuple: A tuple containing the trained model, feature importance DataFrame, and model parameters.
        """
        X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_y, feature_cols = self.prepare_ml_data()

        # Define base model with fixed parameters instead of grid search
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )

        # Train the model
        model.fit(X_train_scaled, y_train_scaled)

        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        self._evaluate_model(y_test, y_pred, "XGBoost")
        self._plot_predictions(y_test, y_pred, "XGBoost")
        self._plot_feature_importance(feature_importance_df, "XGBoost")

        return model, feature_importance_df, {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5
        }

    def train_svr(self):
        """
        Trains a Support Vector Regression (SVR) model on the prepared data and evaluates its performance.

        Returns:
            tuple: A tuple containing the trained model, feature importance DataFrame, and model parameters.
        """
        X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_y, feature_cols = self.prepare_ml_data()

        param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['linear', 'rbf', 'poly']
        }

        model = SVR()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,
                                   scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_scaled)

        best_model = grid_search.best_estimator_

        importance_results = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_results.importances_mean
        }).sort_values(by='Importance', ascending=False)

        y_pred_scaled = best_model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        self._evaluate_model(y_test, y_pred, "SVR")
        self._plot_predictions(y_test, y_pred, "SVR")
        self._plot_feature_importance(feature_importance_df, "SVR")

        return best_model, feature_importance_df, grid_search.best_params_

    @staticmethod
    def _evaluate_model(y_true, y_pred, model_name):
        """
        Evaluates the performance of a model using mean absolute error, mean squared error, and root mean squared error.

        Args:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted target values.
            model_name (str): Name of the model being evaluated.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        print(f"\n{model_name} Evaluation:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")

    @staticmethod
    def _plot_predictions(y_test, y_pred, model_name):
        """
        Plots the actual vs predicted values for a given model.

        Args:
            y_test (pd.Series): True target values.
            y_pred (array-like): Predicted target values.
            model_name (str): Name of the model.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label='Actual', color='blue')
        plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='dashed')
        plt.xlabel('Date')
        plt.ylabel('BTC Close Price')
        plt.title(f'{model_name} Prediction vs Actual (Using Previous Day Features)')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    @staticmethod
    def _plot_feature_importance(feature_importance_df, model_name):
        """
        Plots the feature importance for a given model.

        Args:
            feature_importance_df (pd.DataFrame): DataFrame containing feature importance values.
            model_name (str): Name of the model.
        """
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance in {model_name} Model')
        plt.gca().invert_yaxis()
        plt.show()

if __name__ == "__main__":
    analysis = BitcoinAnalysis()
    analysis.plot_all_candlesticks()
    analysis.plot_correlation_heatmap()
    xgb_model, xgb_importance, xgb_params = analysis.train_xgboost()
    svr_model, svr_importance, svr_params = analysis.train_svr()
