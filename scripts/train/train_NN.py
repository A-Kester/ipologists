import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import pickle
from src.pipeline import features, risk_features, prepare_scaled, make_loader, three_class
from src.models.neural_network import NeuralNetwork
from sklearn.metrics import classification_report

full = pd.read_csv("data/final/dataset_full.csv")



# We train all models separately, because the hyperparameter tuning is different for all

# Binary Classification Model on Full Dataset 
X_train, X_test, y_train, y_test = prepare_scaled(full, "underpriced", features)
train_loader, test_loader, y_test = make_loader(X_train, X_test, y_train, y_test)


counts= np.bincount(y_train)
weigths = torch.tensor(1.0/counts, dtype=torch.float32) #Added weights to account for the class imbalance


model_binary = NeuralNetwork(input_dim=X_train.shape[1], num_classes= 2, hidden_dim= 256)
opt = torch.optim.Adam(model_binary.parameters(), lr =1e-4)
loss_fn = nn.CrossEntropyLoss(weight=weigths)

for epoch in range(500):
    model_binary.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        loss = loss_fn(model_binary(X_batch), y_batch)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    if epoch %25 ==0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")

with open("src/models/saved_models/neural_network/nn_binary_full.pkl", "wb") as f:
    pickle.dump(model_binary, f)


# 3-Class Classification Model on Full Dataset
X_train, X_test, y_train, y_test = prepare_scaled(three_class(full), "three_class", features)
train_loader, test_loader, y_test = make_loader(X_train, X_test, y_train,y_test)

counts = np.bincount(y_train)
weights = torch.tensor(1.0/counts, dtype=torch.float32)

model_3class = NeuralNetwork(input_dim=X_train.shape[1], num_classes=3, hidden_dim= 128)
opt= torch.optim.Adam(model_3class.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(weight = weights)

for epoch in range(1000):
    model_3class.train()
    epoch_loss =0
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        loss= loss_fn(model_3class(X_batch), y_batch)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    if epoch % 50 ==0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")
with open("src/models/saved_models/neural_network/nn_3class.pkl", "wb") as f:
    pickle.dump(model_3class, f)



