from src.pipeline import DataPrepPipeline
import torch
import pandas as pd
import numpy as np

def preprocess(df, target, multiclass=False):
    df_clean = df.dropna(subset=[target])
    X_df = df_clean.drop(columns=[
        target, 'Offer To 1st Close',
        'Pricing Date', 'Issuer Name', 'ticker',
        'Primary Exchange', 'Instit Owner (% Shares Out)',
        'Industry Sector', 'lead_bookrunner'
    ])
    y_df = df_clean[target]
    y_df = pd.get_dummies(y_df) if multiclass else y_df

    X_train_df, y_train_df, X_test_df, y_test_df = _test_train_split(X_df, y_df)

    pipeline = DataPrepPipeline(X_train_df.columns.tolist())
    pipeline.fit(X_train_df)

    X_train = pipeline.transform(X_train_df)
    X_test  = pipeline.transform(X_test_df)
    y_train = torch.from_numpy(y_train_df.values.copy()).float()
    y_test  = torch.from_numpy(y_test_df.values.copy()).float()

    return X_train, y_train, X_test, y_test


def _test_train_split(X_df, y_df, train_size=0.8, random_state=42):
    train_ix = X_df.sample(frac=0.8, random_state=42).index
    test_ix = X_df.drop(train_ix).index

    X_train_df = X_df.loc[train_ix]
    y_train_df = y_df.loc[train_ix]

    X_test_df  = X_df.loc[test_ix]
    y_test_df  = y_df.loc[test_ix]

    return X_train_df, y_train_df, X_test_df, y_test_df

def three_class(df):
  df = df.dropna(subset=['Offer To 1st Close']).copy()
  thresholds = [ 
    df['Offer To 1st Close'] < 0, 
    (df['Offer To 1st Close'] >= 0) & (df['Offer To 1st Close'] <= 20),
    df['Offer To 1st Close'] > 20 ]
  df['three_class'] = np.select(thresholds, [0,1,2])
  return df