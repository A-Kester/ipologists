import pickle 
import pandas as pd
import torch
from sklearn.metrics import classification_report
from src.pipeline import prepare_scaled, features, three_class
from src.models.random_forest import RandomForest

full = pd.read_csv("data/final/dataset_full.csv")

# Our models are separated so that they can be tuned independently, since they have different classes and levels of complexity
# By doing so it ensures that each model is optimized to its best. 

# Binary Random Forest
X_train, X_test, y_train, y_test = prepare_scaled(full, "underpriced", features)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


rf_binary = RandomForest(n_trees=40, max_depth= 6, num_features=int(X_train.shape[1] ** 0.5 ))
rf_binary.fit(X_train, y_train)
preds = rf_binary.predict(X_test)

print("Binary Random Forest")
print("=" * 50)
print(classification_report(y_test, preds, target_names=['Overpriced', 'Underpriced']))

with open("src/models/saved_models/random_forest/rf_binary_full.pkl", "wb") as f:
    pickle.dump(rf_binary, f)

# Three-Class Random Forest

X_train, X_test, y_train, y_test = prepare_scaled(three_class(full), "three_class", features) 

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

rf_3class = RandomForest(n_trees=40, max_depth=8, num_features=int(X_train.shape[1] ** 0.5))
rf_3class.fit(X_train, y_train)
preds = rf_3class.predict(X_test)

print("Three-Class Random Forest")
print("=" * 50)
print(classification_report(y_test, preds, target_names=['Overpriced', 'Mild (0-20%)', 'Strong (>20%)']))

with open("src/models/saved_models/random_forest/rf_3class_full.pkl", "wb") as f:
    pickle.dump(rf_3class, f)