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

"""
    Bitcoin Price Prediction Using Machine Learning

    Chng Zuo En Calvin
    Dirceu de Medeiros Teixeira
    Kelvin Thein
    Melani Sugiharti The

    Flexi Master in Financial Technology
    MH6805: Machine Learning in Finance
    Graded Group Project Team 1
"""

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

    def __init__(self, file_path="https://raw.githubusercontent.com/dirceudn/MH6805-Graded-Group-Project-Team1/refs/heads/main/bitcoin_dataset.csv"):
        """
        Initializes the BitcoinAnalysis class with the specified file path.

        This constructor sets up the initial state of the BitcoinAnalysis object by loading
        and processing the dataset. It initializes empty attributes for holding market data
        and loads the dataset from the specified file path.

        Args:
            file_path (str): The path to the dataset file containing Bitcoin, NASDAQ, and Gold
                         market data. Defaults to "bitcoin_dataset.csv".
        """
        self.gold_data = None
        self.nasdaq_data = None
        self.btc_data = None
        self.file_path = file_path
        self.df = self.load_data()
        self.process_data()
        self.df = self.add_technical_indicators(self.df)


    def load_data(self):
        """
        Loads the dataset from the specified file path and preprocesses it for analysis.

        This method reads the CSV file, standardizes column names by stripping whitespace,
        converts the 'Date' column to datetime format, and sets it as the DataFrame index
        for time-series analysis.

        Returns:
            pd.DataFrame: The preprocessed DataFrame containing the dataset with 'Date' as index.
        """
        df = pd.read_csv(self.file_path)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def process_data(self):
        """
        Processes the loaded dataset to extract and organize Bitcoin, NASDAQ, and Gold market data.

        This method separates the main DataFrame into three distinct DataFrames for each market:
        Bitcoin, NASDAQ, and Gold. For each market, it extracts the relevant columns (Open, High,
        Low, Close, Volume) and renames them to a standardized format for easier comparison and analysis.
        The resulting DataFrames are stored as attributes of the class.
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
        Plots a candlestick chart for the given market data with volume information.

        This method creates a two-panel figure: the upper panel displays a candlestick chart
        showing price movements (Open, High, Low, Close), and the lower panel shows trading volume.
        Green candlesticks represent rising prices (Close >= Open), while red candlesticks
        represent falling prices (Close < Open).

        Args:
            data (pd.DataFrame): DataFrame containing market data with columns 'Open', 'High',
                             'Low', 'Close', and 'Volume'.
            title (str): Title of the plot, typically indicating which market is being displayed.

        Note:
            - Vertical lines represent the price range (High to Low)
            - Colored bars represent the opening and closing prices
            - Volume is displayed with corresponding colors to price movement
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
        Plots candlestick charts for Bitcoin, NASDAQ, and Gold market data in sequence.

        This method calls plot_candlestick() three times, once for each market dataset
        (Bitcoin, NASDAQ, and Gold), creating a visual comparison of price movements
        across these different markets. Each chart displays both price action and volume
        information with appropriate titles.

        Note:
            This is a convenience method that generates all three charts in succession
            rather than requiring separate method calls for each market.
        """
        self.plot_candlestick(self.btc_data, 'Bitcoin Candlestick Chart')
        self.plot_candlestick(self.nasdaq_data, 'NASDAQ Candlestick Chart')
        self.plot_candlestick(self.gold_data, 'Gold Candlestick Chart')

    def plot_correlation_heatmap(self):
        """
        Plots a heatmap of the correlation matrix for all numerical features in the dataset.

        This method calculates the Pearson correlation coefficients between all pairs of
        numerical columns in the dataset and visualizes them using a color-coded heatmap.
        Positive correlations are shown in warm colors (red), negative correlations in cool
        colors (blue), and the correlation strength is indicated by color intensity.

        The heatmap includes annotation of the exact correlation values and uses a diverging
        color map centered at zero for better interpretation of positive vs. negative correlations.

        Note:
            This visualization is particularly useful for identifying relationships between
            Bitcoin prices and other market indicators such as NASDAQ and Gold metrics.
        """
        plt.figure(figsize=(10, 6))
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def prepare_ml_data(self):
        """
        Prepares the data for machine learning models by performing feature engineering,
        data splitting, and scaling operations.

        This method:
        1. Selects relevant features from NASDAQ and Gold data as predictors
        2. Designates Bitcoin closing price as the target variable
        3. Shifts features by one day to predict using previous day's market data
        4. Splits the data into training (80%) and testing (20%) sets
        5. Standardizes both features and target variables using StandardScaler

        Returns:
            tuple: A 6-element tuple containing:
                - X_train_scaled (numpy.ndarray): Scaled training features
                - X_test_scaled (numpy.ndarray): Scaled testing features
                - y_train_scaled (numpy.ndarray): Scaled training target values
                - y_test (pandas.Series): Unscaled testing target values for evaluation
                - scaler_y (StandardScaler): Fitted scaler for inverse transforming predictions
                - feature_cols (list): Names of feature columns used in the model

        Note:
            The feature shift creates a one-day lag to predict Bitcoin prices based on
            the previous day's NASDAQ and Gold market data, simulating a realistic
            prediction scenario.
        """
        target_col = 'BTC Close'
        feature_cols = ['BTC Open', 'BTC High', 'BTC Low', 'BTC Volume', 'NASDAQ Open', 'NASDAQ Close', 'NASDAQ High',
                        'NASDAQ Low', 'NASDAQ Volume',
                        'Gold Open', 'Gold Close', 'Gold High', 'Gold Low', 'Gold Volume',
                        'SMA_10', 'EMA_10', 'RSI_14', 'ROC', 'Rolling_STD', 'Upper_Band', 'Lower_Band', 'MACD',
                        'Stochastic_Oscillator', 'Signal_Line']

        feature_cols = [col for col in feature_cols if col in self.df.columns]

        data = self.df[feature_cols]
        target = self.df[target_col]

        data_shifted = data.shift(1).dropna()
        target_shifted = target.loc[data_shifted.index]

        split_ratio = 0.8
        split_index = int(len(data_shifted) * split_ratio)

        X_train = data_shifted.iloc[:split_index]
        X_test = data_shifted.iloc[split_index:]
        y_train = target_shifted.iloc[:split_index]
        y_test = target_shifted.iloc[split_index:]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_y, feature_cols

    def train_xgboost(self):
        """
        Trains an XGBoost regression model on the prepared data and evaluates its performance.

        This method:
        1. Obtains prepared ML data from prepare_ml_data()
        2. Configures an XGBoost regressor with predefined hyperparameters
        3. Trains the model on the scaled training data
        4. Calculates feature importance to identify influential predictors
        5. Makes predictions on the test set and transforms them back to original scale
        6. Evaluates model performance using several metrics
        7. Generates visualizations for predictions and feature importance

        The XGBoost model uses a gradient boosting framework with decision trees
        to predict Bitcoin closing prices based on NASDAQ and Gold market features.

        Returns:
            tuple: A 3-element tuple containing:
                - model (xgb.XGBRegressor): The trained XGBoost model
                - feature_importance_df (pd.DataFrame): DataFrame with feature importance scores
                - dict: Dictionary of hyperparameters used for the model

        Note:
            Unlike the SVR implementation, this method uses fixed hyperparameters
            rather than grid search for computational efficiency.
        """
        X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_y, feature_cols = self.prepare_ml_data()

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )

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

    @staticmethod
    def add_technical_indicators(df):
        df['SMA_10'] = df['BTC Close'].rolling(window=10).mean()
        df['EMA_10'] = df['BTC Close'].ewm(span=10, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (
                    1 + df['BTC Close'].diff().apply(lambda x: (x if x > 0 else 0)).rolling(14).mean() / df[
                'BTC Close'].diff().apply(lambda x: (-x if x < 0 else 0)).rolling(14).mean()))
        df['ROC'] = df['BTC Close'].pct_change(periods=10) * 100
        df['Rolling_STD'] = df['BTC Close'].rolling(window=10).std()
        df['Upper_Band'] = df['SMA_10'] + (df['Rolling_STD'] * 2)
        df['Lower_Band'] = df['SMA_10'] - (df['Rolling_STD'] * 2)
        df['MACD'] = df['BTC Close'].ewm(span=12, adjust=False).mean() - df['BTC Close'].ewm(span=26,
                                                                                             adjust=False).mean()
        df['Stochastic_Oscillator'] = ((df['BTC Close'] - df['BTC Close'].rolling(14).min()) / (
                    df['BTC Close'].rolling(14).max() - df['BTC Close'].rolling(14).min())) * 100
        df['Signal_Line'] = df['Stochastic_Oscillator'].rolling(3).mean()
        return df

    def train_svr(self):
        """
        Trains a Support Vector Regression (SVR) model with hyperparameter tuning and
        evaluates its performance.

        This method:
        1. Obtains prepared ML data from prepare_ml_data()
        2. Defines a parameter grid for hyperparameter tuning
        3. Performs grid search with 3-fold cross-validation to find optimal parameters
        4. Uses permutation importance to assess feature relevance (since SVR lacks built-in
       feature importance)
        5. Makes predictions on the test set and transforms them back to original scale
        6. Evaluates model performance using several metrics
        7. Generates visualizations for predictions and feature importance

        SVR finds a hyperplane in a high-dimensional space that maximizes the margin
        while limiting prediction errors to within a specified threshold.

        Returns:
            tuple: A 3-element tuple containing:
                - best_model (SVR): The trained SVR model with optimal hyperparameters
                - feature_importance_df (pd.DataFrame): DataFrame with feature importance scores
                - dict: Dictionary of optimal hyperparameters found by grid search

        Note:
            This method uses GridSearchCV to systematically explore different combinations
            of hyperparameters (C, epsilon, kernel) to find the optimal model configuration.
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
        Evaluates the performance of a regression model using multiple error metrics.

        This method calculates and prints three common regression evaluation metrics:
        - Mean Absolute Error (MAE): Average absolute difference between predictions and actual values
        - Mean Squared Error (MSE): Average of squared differences between predictions and actual values
        - Root Mean Squared Error (RMSE): Square root of MSE, providing an error measure in the same
        units as the target variable

        Lower values for all metrics indicate better model performance.

        Args:
            y_true (array-like): True target values from the test set
            y_pred (array-like): Model predictions for the test set
            model_name (str): Name of the model being evaluated (for labeling output)

        Note:
            This is a static method as it doesn't depend on instance state and can be
            used independently of any specific BitcoinAnalysis instance.
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
        Creates a time series plot comparing actual Bitcoin prices against model predictions.

        This method visualizes how well the model's predictions align with actual Bitcoin
        closing prices over time. It plots both series on the same axes, with actual values
        as a solid blue line and predictions as a dashed red line.

        The x-axis displays dates formatted as year-month, and the plot includes a grid,
        legend, and descriptive labels for clear interpretation.

        Args:
            y_test (pd.Series): True Bitcoin closing prices from the test set, with dates as index
            y_pred (array-like): Model predictions for the test set
            model_name (str): Name of the model being visualized (for plot title)

        Note:
            This visualization is crucial for qualitative assessment of model performance,
            revealing patterns that numeric metrics alone might miss, such as periods of
            over/under-prediction or lagging responses to price changes.
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
        Creates a horizontal bar chart displaying the relative importance of each feature
        in the predictive model.

        This method visualizes which features (from NASDAQ and Gold market data) have the
        strongest influence on Bitcoin price predictions according to the model. Features
        are sorted by importance, with the most influential features at the top of the chart.

        Args:
            feature_importance_df (pd.DataFrame): DataFrame containing feature names and their
                                             corresponding importance scores
            model_name (str): Name of the model being visualized (for plot title)

        Note:
            Feature importance interpretation differs between models:
            - For XGBoost: Based on gain (improvement in accuracy brought by a feature)
            - For SVR: Based on permutation importance (decrease in model performance when
            a feature is randomly shuffled)

            This visualization helps identify which market indicators are most predictive
            of Bitcoin price movements.
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
    print(xgb_importance)
    print(xgb_params)
    print(svr_importance)
    print(svr_params)
