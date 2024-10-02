import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

DATASET_PATH = "C:/ml_data/cic_iot_2023/cic_iot_2023_dos_benign_100000.csv"


def _to_xy(df):
    y = df["label"]
    X = df.drop("label", axis=1)
    return X, y

def _one_hot_encode(df, col_name):
    dummies = pd.get_dummies(df.loc[:, col_name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(col_name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(col_name, axis=1, inplace=True)


def get_train_test_val():
    df = pd.read_csv(DATASET_PATH)

    # One-hot-encoding for categorical columns
    categorical_columns = ["Protocol Type"]
    for cat_col in categorical_columns:
        _one_hot_encode(df, cat_col)

    # Encode labels
    labels = df['label'].copy()
    labels[labels == 'DoS'] = 0
    labels[labels == 'Benign'] = 1
    df['label'] = labels.astype(str).astype(int)

    # Split train, test, val
    df_train = df.sample(frac=0.75, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]
    df_val = df_train.sample(frac=0.1, random_state=42)  # validation is part of train
    # df_train = df_train.loc[~df_train.index.isin(df_val.index)]

    # Remove positive samples from train and val
    df_train = df_train[df_train["label"] != 1]
    df_val = df_val[df_val["label"] != 1]

    # Split to X, y
    X_train, y_train = _to_xy(df_train)
    X_test, y_test = _to_xy(df_test)
    X_val, y_val = _to_xy(df_val)

    # Scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert y to numpy
    y_train = y_train.to_numpy(dtype=int)
    y_test = y_test.to_numpy(dtype=int)
    y_val = y_val.to_numpy(dtype=int)

    return X_train, X_test, X_val, y_train, y_test, y_val
