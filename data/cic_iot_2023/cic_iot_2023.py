import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

DATASET_PATH = "C:/ml_data/cic_iot_2023/cic_iot_2023_dos_benign_300000.csv"
OUTLIERS_FRAC = 0.12  # Fraction of outliers in test set


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


def _adapt_ratio(df_test):
    """Adapt ratio of normal/anomalous data"""
    inliers = df_test[df_test["label"] == 0]
    outliers = df_test[df_test["label"] == 1]

    cur_outiers_frac = outliers.shape[0] / df_test.shape[0]
    if cur_outiers_frac > OUTLIERS_FRAC:
        # Remove outliers
        keep_frac = OUTLIERS_FRAC * (1 - cur_outiers_frac) / (cur_outiers_frac * (1 - OUTLIERS_FRAC))
        outliers = outliers.sample(frac=keep_frac)
    else:
        # Remove inliers
        keep_frac = (cur_outiers_frac * (1 - OUTLIERS_FRAC)) / (OUTLIERS_FRAC * (1 - cur_outiers_frac))
        inliers = inliers.sample(frac=keep_frac)

    return pd.concat([inliers, outliers], ignore_index=True)


def get_train_test():
    # Read dataset
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

    # Adapt ratio of normal/anomalous samples in df_test
    df_test = _adapt_ratio(df_test)

    # Remove positive samples from train (ALAD is trained on inliers)
    df_train = df_train[df_train["label"] != 1]

    # Split to X, y
    X_train, y_train = _to_xy(df_train)
    X_test, y_test = _to_xy(df_test)

    # Scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert y to numpy
    y_train = y_train.to_numpy(dtype=int)
    y_test = y_test.to_numpy(dtype=int)

    return X_train, X_test, y_train, y_test
