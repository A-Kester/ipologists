import torch
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from src.pipeline import (
    prepare_scaled,
    make_loader,
    features, 
    three_class
)
from src.models.logistic import LogisticRegression, GradientDescentOptimizer
names   = ["binary", "3class"]
full = pd.read_csv('data/final/dataset_full.csv')
risk = pd.read_csv('data/final/dataset_with_risk.csv')

target=["underpriced", "three_class"]
df = [full, three_class(full)]

for i in range(2):
    X_train, X_test, y_train, y_test = prepare_scaled(
        df=df[i],
        target=target[i],
        features=features
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(pd.get_dummies(y_train).values, dtype=torch.float32) # get one hot encoding of classes
    y_test = torch.tensor(pd.get_dummies(y_test).values, dtype=torch.float32)

    model = LogisticRegression(d_features=X_train.shape[1], k_classes=i+2)
    opt = GradientDescentOptimizer(model, learning_rate=0.001)
    losses = []

    for epoch in range(6000):
        opt.step(X_train, y_train)
        q = model.forward(X_train)
        loss = -torch.mean(torch.sum(y_train*torch.log(q), dim=1))
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    with open(f"src/models/saved_models/logistic/logistic_{names[i]}_full.pkl", "wb") as f:
        pickle.dump(model,f)