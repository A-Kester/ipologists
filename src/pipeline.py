import numpy as np
import torch

class DataPrepPipeline:
  def __init__(self, features):
    self.features = features
  def fit(self, X):
    return self       # XGBoost automatically deals with NA values so no need to do anything
  def transform(self, X):
    return torch.from_numpy(X[self.features].values).float()

