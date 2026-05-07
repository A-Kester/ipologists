import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score

# Logistic Regression
class LogisticRegression:
    def __init__(self, d_features, k_classes):
        self.W = torch.zeros(d_features, k_classes)

    def forward(self, X):
        S = X @ self.W
        return torch.softmax(S, dim=1)

class GradientDescentOptimizer:
  def __init__(self, model, learning_rate=0.01):
    self.model = model
    self.learning_rate = learning_rate

  def step(self, X, y):
    self.model.W -= self.learning_rate * self.grad_func(X, y)

  def grad_func(self, X, y):
    q = self.model.forward(X)
    return X.T @ (q - y) / X.shape[0]


