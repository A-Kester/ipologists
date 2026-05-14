import torch
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import classification_report
from src.pipeline import (prepare_scaled, features, three_class)
from src.models.logistic import LogisticRegression, GradientDescentOptimizer
labels = [['Overpriced', 'Underpriced'], ['Overpriced', 'Mild (0-20%)', 'Strong (>20%)']]
names   = ["binary", "3class"]
full = pd.read_csv('data/final/dataset_full.csv')
risk = pd.read_csv('data/final/dataset_with_risk.csv')

target=["underpriced", "three_class"]
df = [full, three_class(full)] # Three class label to for our 3-class classification

for i in range(2):
    # Prepare and scale features
    X_train, X_test, y_train, y_test = prepare_scaled(
        df=df[i],
        target=target[i],
        features=features
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(pd.get_dummies(y_train).values, dtype=torch.float32) # get one hot encoding of classes
    y_test = torch.tensor(pd.get_dummies(y_test).values, dtype=torch.float32)

    # initialize model: binary used 2, and 3-class uses 3
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
    s_pred =model.forward(X_test)
    y_test_preds = s_pred.argmax(dim=1)
    y_test_labels= y_test.argmax(dim=1).int()

    print(classification_report(y_test_labels, y_test_preds, target_names=labels[i]))

    # Saving preds to save time on notebook
    np.save(f"src/models/saved_models/logistic/lr_{names[i]}_preds.npy", y_test_preds.numpy())
    # Saving model to pickle file
    with open(f"src/models/saved_models/logistic/logistic_{names[i]}_full.pkl", "wb") as f:
        pickle.dump(model,f)