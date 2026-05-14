"""
Implemented using:
    -https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html
    -Assistance from Generative AI for debugging purposes
    """

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from src.pipeline import prepare_xgboost, features, risk_features, three_class

full = pd.read_csv("data/final/dataset_full.csv")
risk = pd.read_csv("data/final/dataset_with_risk.csv")

# We will separate our training for binary and 3-class to optimize each to its best. 

# Binary XGBoost
X_train, X_test, y_train, y_test = prepare_xgboost(full, "underpriced", features)

weights = (y_train ==0).sum() / (y_train== 1).sum() # weights from classes

# Initialize the model

xgb_binary = XGBClassifier( n_estimators = 4000, max_depth =10, learning_rate = 0.01, scale_pos_weight = weights, random_state =42, early_stopping_rounds=50)

xgb_binary.fit(X_train, y_train, eval_set= [(X_test, y_test)], verbose = False)
preds = xgb_binary.predict(X_test)

print("Binary XGBoost")
print("=" * 50)
print(classification_report(y_test, preds, target_names=['Overpriced', 'Underpriced']))

# Save preds to save time in notebook
np.save("src/models/saved_models/xgBoost/xgb_binary_preds.npy", preds)
# Save model
with open("src/models/saved_models/xgBoost/xgb_binary_full.pkl", "wb") as f:
    pickle.dump(xgb_binary, f)


# Three-Class XGBoost
X_train, X_test, y_train, y_test = prepare_xgboost(three_class(full), "three_class", features)

xgb_3class = XGBClassifier(n_estimators =2000, max_depth = 10, learning_rate =0.01, objective = "multi:softmax", num_class=3, random_state=42)

counts = np.bincount(y_train)
sample_weights = np.array([1.0/counts[y] ** 0.5 for y in y_train])

xgb_3class.fit(X_train, y_train, sample_weight = sample_weights, eval_set = [(X_test, y_test)], verbose = False)

preds = xgb_3class.predict(X_test)
print("Three-Class XGBoost")
print("=" * 50)
print(classification_report(y_test, preds, target_names=['Overpriced', 'Mild (0-20%)', 'Strong (>20%)']))
np.save("src/models/saved_models/xgBoost/xgb_3class_preds.npy", preds)
with open("src/models/saved_models/xgBoost/xgb_3class_full.pkl", "wb") as f:
    pickle.dump(xgb_3class, f)





# Risk Datasets Binary 

# Dataset without risk features our baseline

X_train, X_test, y_train, y_test = prepare_xgboost(risk, "underpriced", features)
scale= (y_train==0).sum() / (y_train ==1).sum()

xgb_risk_baseline = XGBClassifier(n_estimators = 2000, max_depth = 4, learning_rate=0.01, scale_pos_weight=scale, random_state = 42, early_stopping_rounds=50)
xgb_risk_baseline.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose =False)
preds_baseline = xgb_risk_baseline.predict(X_test)

with open("src/models/saved_models/xgBoost/xgb_baseline_risk.pkl", "wb") as f:
    pickle.dump(xgb_risk_baseline, f)

# Dataset with risk features 

X_train, X_test, y_train, y_test = prepare_xgboost(risk, "underpriced", risk_features)
scale= (y_train==0).sum() / (y_train ==1).sum()

xgb_risk_full = XGBClassifier(n_estimators = 2000, max_depth = 4, learning_rate=0.01, scale_pos_weight=scale, random_state = 42, early_stopping_rounds=50)
xgb_risk_full.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose =False)
preds_full = xgb_risk_full.predict(X_test)



with open("src/models/saved_models/xgBoost/xgb_full_risk.pkl", "wb") as f:
    pickle.dump(xgb_risk_full, f)