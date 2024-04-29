import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    with open('unused_features.txt', 'r') as file:
        unused_features = [line.strip() for line in file]
        
    df = df.drop(columns = unused_features)
    return df

def resample_data(df, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='rank'), df['rank'], test_size=test_size, random_state=42)

    smote = SMOTE(random_state=42)
    X_over_resampled, y_over_resampled = smote.fit_resample(X_train, y_train)
    
    X_resampled = X_over_resampled.drop(['query_id'], axis=1)
    y_resampled = y_over_resampled
    groups_resampled = X_over_resampled['query_id']
    
    return X_resampled, y_resampled, groups_resampled


def train(X_resampled, y_resampled, groups_resampled):
    group_kfold_base = KFold(n_splits=5, shuffle=True, random_state=42)

    params = {
            'objective': 'lambdarank',
            'metric': 'ndcg', # NDCG:top=5
            'ndcg_eval_at': [5],
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 25,
            'min_data_in_leaf': 15,
            'num_iterations': 1000,
            'verbose': -1
        }

        
    ndcg_5_scores = []

    for train_index, test_index in group_kfold_base.split(X_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        train_query_id, test_query_id = groups_resampled.iloc[train_index], groups_resampled.iloc[test_index]

        train_group = train_query_id.value_counts().sort_index().values
        valid_group = test_query_id.value_counts().sort_index().values

        train_data = lightgbm.Dataset(X_train, label=y_train, group=train_group)
        validation_data = lightgbm.Dataset(X_test, label=y_test, group=valid_group)

        model = lightgbm.train(
            params,
            train_data,
            valid_sets=[validation_data]
        )

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        ndcg_score_val = ndcg_score([y_test], [y_pred], k=5)
        ndcg_5_scores.append(ndcg_score_val)
    
    return model, np.mean(ndcg_5_scores)


def fit(params, X_train, y_train, X_val, y_val, groups, train_index, test_index):
    params = {
            'objective': 'lambdarank',
            'metric': 'ndcg', # NDCG:top=5
            'ndcg_eval_at': [5],
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 25,
            'min_data_in_leaf': 15,
            'num_iterations': 1000,
            'verbose': -1
        }

    train_query_id, test_query_id = groups.iloc[train_index], groups.iloc[test_index]

    train_group = train_query_id.value_counts().sort_index().values
    valid_group = test_query_id.value_counts().sort_index().values

    train_data = lightgbm.Dataset(X_train, label=y_train, group=train_group)
    validation_data = lightgbm.Dataset(X_val, label=y_val, group=valid_group)
    
    model = lightgbm.train(
            params,
            train_data,
            valid_sets=[validation_data]
        )

    return model


def predict(model, X_test):
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return y_pred
