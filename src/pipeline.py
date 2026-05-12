import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

features = ['Offer Size (M)', 'Offer Price','Initial Pub Offer (Shares Offered)', 'Market Cap at Offer (M)',
    'offer_size_to_mktcap', 'has_bulge_bracket', 'vix', 'nasdaq', 'fed_funds', 'treasury_10y', 'cpi',
    'unemployment', 'gdp', 'ipo_volume', 'market_return_1m']
risk_features = features + ['regulatory_risk', 'competitive_risk', 'financial_risk', 'overall_risk']


def three_class(df):
    """
    Adds a `three_class` column : 0 = overpriced(<0%), 1 = Mild (0-20%), 2 = Strong (>20%)
    """
    df = df.dropna(subset=["Offer To 1st Close"])
    threshold = [
        df['Offer To 1st Close'] < 0,
        (df['Offer To 1st Close'] >= 0) & (df['Offer To 1st Close'] <= 20),
        df['Offer To 1st Close'] > 20 ]
    df["three_class"] = np.select(threshold, [0,1,2])
    return df

def data_splitter(df, target, features):
    """
    This is our 80/20 split that is used by all our models
    """
    df = df.dropna(subset=[target])
    X_df = df[features]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size =0.2, stratify=y, random_state=42) # We use same random_state seed to have same split all models
    return X_train, X_test, y_train, y_test

## Data Pipelines

class XGBoostPipeline:
    """
    Feature selector for XGBosst, it doesn't need imputation - XGBoost handles NAN natively,
    also since tree-based it doesn't need scaler.
    """
    def __init__(self,features):
        self.features = features

    def fit(self,X):
        return self

    def transform(self,X):
        return X[self.features].values.astype(np.float32)
    
class ScaledPipeline:
    """
    Does Median imputation + StandardScaler. Used by logistic regression, neural network
    and random forest who all need imputation. Random forest, since it is a tree model, doesn't 
    benefit from scaled features, but isn't harmed by it either so this pipeline is used.
    """
    def __init__(self,features):
        self.features = features
        self.medians = None
        self.scaler= StandardScaler()
      
    def fit(self,X):
        self.medians = X[self.features].median()
        X_filled = X[self.features].fillna(self.medians)
        self.scaler.fit(X_filled)
        return self
    
    def transform(self,X):
        X_filled = X[self.features].fillna(self.medians)
        return self.scaler.transform(X_filled).astype(np.float32)
    
def prepare_xgboost(df, target, features):
    """
    Splits and prepares the data fro XGBoost
    """
    X_train, X_test, y_train,y_test = data_splitter(df, target, features)
    pipeline = XGBoostPipeline(features)
    pipeline.fit(X_train)
    return pipeline.transform(X_train), pipeline.transform(X_test), y_train, y_test

def prepare_scaled(df, target, features):
    """
    Splits and prepares the data for NN / logistic/ random forest
    """
    X_train, X_test, y_train,y_test = data_splitter(df, target, features)
    pipeline = ScaledPipeline(features)
    pipeline.fit(X_train)
    return pipeline.transform(X_train), pipeline.transform(X_test), y_train, y_test


def make_loader(X_train, X_test, y_train, y_test, batch_size = 64):
    """
    makes DataLoaders for our model training
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size= batch_size, shuffle= True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size= batch_size)
    return train_loader, test_loader, y_test