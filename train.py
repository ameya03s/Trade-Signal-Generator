import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def prep_features_labels(df, features):
    return df[features], df["labels"] # features is x, labels is y

def train(x_train, y_train, x_eval, y_eval):
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        )
    
    model.fit(x_train, y_train)

    # os.makedirs("model", exist_ok=True)
    # joblib.dump(model, 'model/model.joblib')

    acc = model.score(x_eval, y_eval)
    return acc