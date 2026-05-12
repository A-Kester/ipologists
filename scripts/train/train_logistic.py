import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from src.pipeline import (
    prepare_scaled,
    make_loader,
    features, 
    three_class
)
from src.models.logistic import LogisticRegression, GradientDescentOptimizer

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
    opt = GradientDescentOptimizer(model, learning_rate=0.01)
    losses = []

    for epoch in range(20):
        opt.step(X_train, y_train)

    s_pred = model.forward(X_test)
    y_test_preds = s_pred.argmax(dim=1)
    y_test_labels = y_test.argmax(dim=1).int()
    print("Binary Logistic Regression" if i == 0 else "Three-Class Logistic Regression")
    print(f"\tAccuracy: {accuracy_score(y_test_labels, y_test_preds):.3f}")