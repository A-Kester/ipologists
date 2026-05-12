import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as nnf
#from sklearn.metrics import accuracy_score
from src.pipeline import (
    prepare_scaled,
    make_loader,
    features, 
    three_class
)
from src.models.random_forest import RandomForest

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
    y_train = torch.tensor(y_train, dtype=torch.float32) 
    y_test = torch.tensor(y_test, dtype=torch.float32)

    rf = RandomForest(n_trees=20, max_depth=10, num_features=int(X_train.shape[1] ** 0.5))
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)

    acc = (preds == y_test).float().mean()
    print("Binary Random Forest" if i == 0 else "Three-Class Random Forest")
    print(f"\tAccuracy: {acc.item():.3f}")