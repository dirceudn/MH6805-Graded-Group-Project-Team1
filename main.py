import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def get_data_frame():
    df = pd.read_csv("bitcoin_dataset.csv")
    return df


def pre_processing_data_frame():
    # Convert Date to datetime format for easier handling of time-series data.
    # Set Date as the index for time-series models.

    df = get_data_frame()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Handle Missing or Duplicated Values
    # Check for missing values and decide whether to fill or drop them.
    # Check for duplicates and remove them.

    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def normalize_data():
    """
    4. Normalize Data for Machine Learning
    Normalize numeric values for ML models like Random Forest, XGBoost, and LSTMs.
    Use MinMaxScaler or StandardScaler.
    """
    df = pre_processing_data_frame()
    scaler = MinMaxScaler()
    scaled_features = ['BTC Open', 'BTC High', 'BTC Low', 'BTC Close', 'BTC Volume',
                       'NASDAQ Open', 'NASDAQ High', 'NASDAQ Low', 'NASDAQ Close', 'Gold Open']

    df[scaled_features] = scaler.fit_transform(df[scaled_features])



def main():
    print("Hello World!")

if __name__ == "__main__":
    data = get_data_frame()
    data.info()
    print(data.head())
    main()



