import numpy as np
import pandas as pd
import xgboost as xgb

def load_data(path):
    return pd.read_csv(path)

def data_split(df):
    num_to_train = int(len(df) * 0.7)
    df_train = df[:num_to_train]
    df_eval = df[num_to_train+2:] # skipping a day to get a one day gap
    return df_train, df_eval

def prep_features_labels(df, features):
    return df[features], df["labels"] # features is x, labels is y

def train(x_train, x_eval, y_train, y_eval):
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        )
    
    print(model.objective)
    print(model.n_estimators)
    print(model.max_depth)
    print(model.learning_rate)
    return model