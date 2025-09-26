"""Utilities to prepare training/evaluation datasets and feature matrices.

Functions here fetch historical market data, compute features and labels,
and return clean splits for model training and evaluation.
"""

from data.ticker_data import *
from features.features import get_feature_list, add_features
from labeler import *
from train import prep_features_labels
import os

def prep_data(ticker, start, end):
  """Prepare train and evaluation DataFrames with features and labels.

  Args:
    ticker: Symbol to fetch for the evaluation period.
    start: ISO date string for evaluation start.
    end: ISO date string for evaluation end.

  Returns:
    Tuple of (df_train, df_eval) with engineered features and `labels` column.
  """
  df_train = get_dataset()
  df_train = add_features(df_train)
  df_train['labels'] = add_labels(df_train, "Close", "atr_14", 0.5, 1)

  df_eval = get_data(ticker, start, end)
  df_eval = add_features(df_eval)
  df_eval['labels'] = add_labels(df_eval, "Close", "atr_14", 0.5, 1)

  # drop all NaN values from both training and eval sets
  df_train = df_train.dropna(subset=get_feature_list() + ["labels"])
  df_eval = df_eval.dropna(subset=get_feature_list()+['labels'])

  return df_train, df_eval

def prep_train_set(df_train, df_eval):
  """Build train/eval splits for model training.

  Args:
    df_train: Training DataFrame with features and labels.
    df_eval: Evaluation DataFrame with features and labels.

  Returns:
    x_train, y_train, x_eval, y_eval suitable for fitting a classifier.
  """
  x_train, y_train = prep_features_labels(df_train, get_feature_list())
  x_eval, y_eval = prep_features_labels(df_eval, get_feature_list())

  return x_train, y_train, x_eval, y_eval