from data.ticker_data import *
from features.features import get_feature_list, add_features
from labeler import *
from train import prep_features_labels

def prep_data(ticker, start, end):
  df_train = get_dataset()
  df_train = add_features(df_train)
  df_train['labels'] = add_labels(df_train, "Close", "atr_14", 0.5, 1)

  df_eval = get_data(ticker, start, end)
  df_eval = add_features(df_eval)
  df_eval['labels'] = add_labels(df_eval, "Close", "atr_14", 0.5, 1)

  df_train = df_train.dropna(subset=get_feature_list() + ["labels"])
  df_eval = df_eval.dropna(subset=get_feature_list()+['labels'])

  return df_train, df_eval

def prep_train_set(df_train, df_eval):
  x_train, y_train = prep_features_labels(df_train, get_feature_list())
  x_eval, y_eval = prep_features_labels(df_eval, get_feature_list())

  return x_train, y_train, x_eval, y_eval