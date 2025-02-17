import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, f1_score
import warnings
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore', category=FutureWarning)

def get_data_frame():
    """
        Load and return the Bitcoin dataset from CSV file.

        Returns:
            pd.DataFrame: Raw Bitcoin dataset containing various price metrics and technical indicators.

        Notes:
            - Uses pandas.read_csv() for data loading
            - Assumes file is in current working directory
            - No preprocessing applied at this stage
    """
    df = pd.read_csv("bitcoin_dataset.csv")
    return df

def convert_df_date_time():
    """
        Convert date column to datetime format and set as index.

        Returns:
            pd.DataFrame: Time-series indexed dataframe with datetime index

        Methods:
            - pd.to_datetime(): Convert string dates to datetime objects
            - set_index(): Set 'Date' as dataframe index
            - sort_index(): Ensure chronological ordering
    """
    df = get_data_frame()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def get_df_standardized():
    """
    Standardize numerical features using z-score normalization.

    Returns:
        pd.DataFrame: Standardized dataframe with all features except 'Trend' normalized

    Methods:
        - StandardScaler(): Normalize features to mean=0, std=1
        - pandas.DataFrame.copy(): Create copy to avoid modifying original data
        - df.columns.difference(): Exclude 'Trend' column from scaling
    """
    df = convert_df_date_time()
    df.sort_index(inplace=True)  # Ensure time-series order
    ft_columns = df.columns.difference(['Trend']) if 'Trend' in df.columns else df.columns

    scaler = StandardScaler()
    data_frame_scaled = df.copy()
    data_frame_scaled[ft_columns] = scaler.fit_transform(data_frame_scaled[ft_columns])
    return data_frame_scaled


def analyze_feature_importance():
    """
    Analyze feature importance using Random Forest classifier.

    Returns:
        list: Top 60% most important features based on mean decrease impurity

    Methods:
        - RandomForestClassifier(): Ensemble method for feature importance calculation
        - class_weight='balanced': Handle class imbalance
        - feature_importances_: RF's intrinsic feature importance metric
        - Rolling window statistics for time-series features
    """
    df_scaled = get_df_standardized()

    features = df_scaled.columns.difference(['Trend'])
    target = (df_scaled['Trend'] > 0.5).astype(int)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight='balanced',
        max_depth=15,
        min_samples_leaf=5,
        max_features='sqrt'
    )
    model.fit(df_scaled[features], target)

    importances = model.feature_importances_
    feature_importances = list(zip(features, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    print("Feature importances:")
    for feature, importance in feature_importances:
        print(f"{feature}: {importance:.4f}")

    num_top_features = max(10, int(len(features) * 0.6))
    top_features = [feature for feature, _ in feature_importances[:num_top_features]]
    return top_features

def add_technical_indicators(df):
    """
    Enhance dataframe with technical analysis indicators.

    Args:
        df (pd.DataFrame): Original dataframe with price data

    Returns:
        pd.DataFrame: Enhanced dataframe with 25+ technical indicators

    Indicators Added:
        - Moving Averages (SMA, EMA)
        - Momentum Indicators (RSI, ROC, MACD)
        - Volatility Measures (Rolling STD, Bollinger Bands)
        - Trend Indicators (ADX, Stochastic Oscillator)

    Methods:
        - pandas.rolling(): Window calculations
        - ewm(): Exponential weighted moving averages
        - diff(): Price changes
        - ffill/bfill: Handle NaN values from window calculations
    """
    df['SMA_7'] = df['BTC Close'].rolling(window=7).mean()
    df['SMA_14'] = df['BTC Close'].rolling(window=14).mean()
    df['SMA_30'] = df['BTC Close'].rolling(window=30).mean()

    df['Price_Change_1d'] = df['BTC Close'].pct_change(periods=1)
    df['Price_Change_7d'] = df['BTC Close'].pct_change(periods=7)
    df['Price_Change_14d'] = df['BTC Close'].pct_change(periods=14)

    df['Volatility_7d'] = df['BTC Close'].rolling(window=7).std()
    df['Volatility_14d'] = df['BTC Close'].rolling(window=14).std()

    # MACD
    ema12 = df['BTC Close'].ewm(span=12).mean()
    ema26 = df['BTC Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    window = 20
    df['BB_Middle'] = df['BTC Close'].rolling(window=window).mean()
    df['BB_Std'] = df['BTC Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    # RSI
    delta = df['BTC Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain_14 = gain.rolling(window=14).mean()
    avg_loss_14 = loss.rolling(window=14).mean()
    rs_14 = avg_gain_14 / avg_loss_14
    df['RSI_14'] = 100 - (100 / (1 + rs_14))

    # Rate of Change
    df['ROC_5'] = df['BTC Close'].pct_change(periods=5) * 100
    df['ROC_10'] = df['BTC Close'].pct_change(periods=10) * 100
    df['ROC_20'] = df['BTC Close'].pct_change(periods=20) * 100

    # Stochastic Oscillator
    n = 14
    df['SO_K'] = 100 * ((df['BTC Close'] - df['BTC Low'].rolling(n).min()) /
                        (df['BTC High'].rolling(n).max() - df['BTC Low'].rolling(n).min()))
    df['SO_D'] = df['SO_K'].rolling(3).mean()

    # Average Directional Index
    high_delta = df['BTC High'].diff()
    low_delta = df['BTC Low'].diff()

    plus_dm = high_delta.where((high_delta > 0) & (high_delta > low_delta.abs()), 0)
    minus_dm = low_delta.abs().where((low_delta < 0) & (low_delta.abs() > high_delta), 0)

    tr1 = df['BTC High'] - df['BTC Low']
    tr2 = (df['BTC High'] - df['BTC Close'].shift()).abs()
    tr3 = (df['BTC Low'] - df['BTC Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    df['ADX'] = dx.rolling(14).mean()

    # EMAs
    df['EMA_5'] = df['BTC Close'].ewm(span=5).mean()
    df['EMA_10'] = df['BTC Close'].ewm(span=10).mean()
    df['EMA_21'] = df['BTC Close'].ewm(span=21).mean()

    df = df.ffill().bfill()
    return df


def get_data_coordinates(look_back, top_features=None):
    """
        Prepare time-series sequences for model training.

        Args:
            look_back (int): Number of historical time steps to use as features
            top_features (list): Optional list of selected features

        Returns:
            tuple: (X, y, feature_names) shaped for time-series modeling

        Methods:
            - StandardScaler(): Feature normalization
            - numpy.array(): Create 3D array (samples, timesteps, features)
            - pandas.difference(): Feature selection
            - Sequence generation with sliding window
    """
    df = convert_df_date_time()
    df.sort_index(inplace=True)
    df = add_technical_indicators(df)

    ft_columns = df.columns.difference(['Trend']) if 'Trend' in df.columns else df.columns
    scaler = StandardScaler()

    data_frame_scaled = df.copy()
    data_frame_scaled[ft_columns] = scaler.fit_transform(data_frame_scaled[ft_columns])
    df_scaled = data_frame_scaled

    if top_features:
        valid_features = [f for f in top_features if f in df_scaled.columns]
        features = valid_features
    else:
        features = df_scaled.columns.difference(['Trend'])

    data = df_scaled[features].values
    target = (df_scaled['Trend'] > 0.5).astype(int).values

    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(target[i + look_back])

    X = np.array(X)
    y = np.array(y)
    return X, y, features


def augment_minority_class(X, y):
    """
    Balance class distribution through synthetic sample generation.

    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector

    Returns:
        tuple: Balanced (X, y) through linear interpolation and noise injection

    Methods:
        - numpy.random.choice(): Minority sample selection
        - Linear interpolation between samples
        - Gaussian noise addition (μ=0, σ=0.01*feature_std)
        - numpy.vstack()/concatenate(): Combine original and synthetic samples
    """
    unique, counts = np.unique(y, return_counts=True)
    if len(counts) == 2 and abs(counts[0] - counts[1]) <= 5:
        # effectively balanced, no augmentation
        return X, y

    class_counts = dict(zip(unique, counts))
    minority_class = 0 if class_counts[0] < class_counts[1] else 1
    majority_class = 1 - minority_class
    num_to_augment = class_counts[majority_class] - class_counts[minority_class]
    if num_to_augment <= 0:
        return X, y

    minority_indices = np.where(y == minority_class)[0]
    augmented_X, augmented_y = [], []

    for _ in range(num_to_augment):
        idx1, idx2 = np.random.choice(minority_indices, 2, replace=len(minority_indices) < 2)
        alpha = np.random.random()
        new_sample = X[idx1] * alpha + X[idx2] * (1 - alpha)

        noise_scale = 0.01 * np.std(new_sample, axis=0)
        noise = np.random.normal(0, noise_scale, new_sample.shape)
        augmented_sample = new_sample + noise

        augmented_X.append(augmented_sample)
        augmented_y.append(minority_class)

    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    X_balanced = np.vstack([X, augmented_X])
    y_balanced = np.concatenate([y, augmented_y])
    return X_balanced, y_balanced


def get_split_data_length(X):
    """
        Calculate dataset splits for time-series cross-validation.

        Returns:
            tuple: (train_size, val_size) in samples

        Split Strategy:
            - 70% training
            - 15% validation
            - 15% testing
    """
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)
    return train_size, val_size


def get_train_values(train_size, val_size, X, y):
    """
        Split data into training, validation, and test sets with class balancing.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)

        Methods:
            - Array slicing for time-series splits
            - augment_minority_class(): Applied to training set only
            - numpy.unique(): Class distribution analysis
    """
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

    X_train_bal, y_train_bal = augment_minority_class(X_train, y_train)

    print("Balanced Training Class Distribution:", dict(zip(*np.unique(y_train_bal, return_counts=True))))
    print("Validation Class Distribution:", dict(zip(*np.unique(y_val, return_counts=True))))
    print("Test Class Distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

    return X_train_bal, y_train_bal, X_val, y_val, X_test, y_test


# ------------------------------
#  Flatten Time-Series for XGB
# ------------------------------
def flatten_time_series(X):
    """
    Transform 3D time-series data to 2D for tree-based models.

    Args:
        X (np.array): 3D array of shape (samples, timesteps, features)

    Returns:
        np.array: 2D array of shape (samples, timesteps*features)

    Methods:
        - numpy.reshape(): Flatten temporal dimension
    """
    N, L, F = X.shape
    return X.reshape(N, L * F)


def training_model():
    """
    Train and evaluate XGBoost classifier on time-series Bitcoin data.

    Returns:
        tuple: (trained model, selected features)

    Pipeline Steps:
        1. Feature importance analysis with Random Forest
        2. Time-series feature engineering (60-day lookback)
        3. Class balancing through synthetic minority oversampling
        4. XGBoost training with early stopping
        5. Comprehensive evaluation on test set

    Evaluation Metrics:
        - Accuracy, AUC-ROC, F1 Score
        - Confusion Matrix
        - Precision-Recall Analysis
        - Optimal Threshold Calculation

    Methods:
        - xgb.XGBClassifier(): Gradient boosted trees implementation
        - Early stopping with 30-round patience
        - sklearn.metrics: Comprehensive model evaluation
        - matplotlib/seaborn: Visualization tools
    """

    top_features = analyze_feature_importance()

    look_back = 60
    X, y, features = get_data_coordinates(look_back=look_back, top_features=top_features)
    print(f"Original X shape (3D): {X.shape}, y shape: {y.shape}")
    print(f"Using {len(features)} features: {features}")

    train_size, val_size = get_split_data_length(X)
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_values(train_size, val_size, X, y)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    X_train_flat = flatten_time_series(X_train)  # shape => (N_train, look_back*features)
    X_val_flat   = flatten_time_series(X_val)
    X_test_flat  = flatten_time_series(X_test)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        early_stopping_rounds=30,
        verbose=True
    )

    val_probs = model.predict_proba(X_val_flat)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()
    val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0
    print(f"\n[Validation] Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")

    test_probs = model.predict_proba(X_test_flat)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)

    accuracy = (test_preds == y_test).mean()
    if len(set(y_test)) > 1:
        test_auc = roc_auc_score(y_test, test_probs)
    else:
        test_auc = 0

    print(f"[Test] Accuracy: {accuracy:.4f}, AUC: {test_auc:.4f}")

    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title("Confusion Matrix (XGB)")
    plt.tight_layout()
    plt.savefig("bitcoin_xgb_confusion_matrix.png")
    plt.show()

    report = classification_report(y_test, test_preds)
    print("Classification Report:")
    print(report)

    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = f1_scores.argmax()
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    print(f"Optimal threshold: {best_threshold:.4f}")
    opt_preds = (test_probs > best_threshold).astype(int)
    opt_accuracy = (opt_preds == y_test).mean()
    opt_cm = confusion_matrix(y_test, opt_preds)
    opt_report = classification_report(y_test, opt_preds)
    opt_f1 = f1_score(y_test, opt_preds)

    print("\nMetrics with optimal threshold:")
    print(f"Accuracy: {opt_accuracy:.4f}")
    print(f"F1 Score: {opt_f1:.4f}")
    print("Confusion Matrix:")
    print(opt_cm)
    print("Classification Report:")
    print(opt_report)

    return model, features

if __name__ == "__main__":
    model, selected_features = training_model()
