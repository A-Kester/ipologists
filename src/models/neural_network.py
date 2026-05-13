import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes =2, hidden_dim = 128):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    def forward(self,X):
        return self.pipeline(X)