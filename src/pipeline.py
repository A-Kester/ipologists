class DataPrepPipeline:
  def __init__(self, features):
    self.features = features
  def fit(self, X):
    return self       # XGBoost automatically deals with NA values so no need to do anything
  def transform(self, X):
    return X[self.features].values.astype(np.float32)

