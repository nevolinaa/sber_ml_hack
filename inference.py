import pandas as pd
import numpy as np
import xgboost as xgb
import re
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import joblib

from tqdm._tqdm_notebook import tqdm_notebook

PATH_DATA = './data'
MODEL_PATH = "model.pkl"

# Считываем данные
tr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')

transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')

gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
gender_train.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
gender_test.drop(columns=['Unnamed: 0'], inplace=True, axis=1)

transactions_with_gender_train = transactions.join(gender_train, how='inner', on='client_id')
transactions_with_gender_test = transactions.join(gender_test, how='inner', on='client_id')

mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
trans_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')

transactions_with_mcc_train = transactions_with_gender_train.join(mcc_codes, how='inner', on='mcc_code')
transactions_with_mcc_test = transactions_with_gender_test.join(mcc_codes, how='inner', on='mcc_code')

transactions_train_0 = transactions_with_mcc_train.join(trans_types, how='inner', on='trans_type')
transactions_test_0 = transactions_with_mcc_test.join(trans_types, how='inner', on='trans_type')

tr_mcc_categories = pd.read_csv(os.path.join(PATH_DATA, 'mcc_categories.csv'), sep=';', index_col='mcc_code')

transactions_train = transactions_train_0.join(tr_mcc_categories, how='inner', on='mcc_code')
transactions_test = transactions_test_0.join(tr_mcc_categories, how='inner', on='mcc_code')

# Функции, которыми можно пользоваться для построения классификатора,
# оценки его результатов и построение прогноза для тестовой части пользователей

# Cross-validation score (среднее значение метрики ROC AUC на тренировочных данных)
def cv_score(params, train, y_true):
    cv_res=xgb.cv(params, xgb.DMatrix(train, y_true),
                  early_stopping_rounds=10, maximize=True,
                  num_boost_round=10000, nfold=5, stratified=True)
    index_argmax = cv_res['test-auc-mean'].argmax()
    print('Cross-validation, ROC AUC: {:.3f}+-{:.3f}, Trees: {}'.format(cv_res.loc[index_argmax]['test-auc-mean'],
                                                                        cv_res.loc[index_argmax]['test-auc-std'],
                                                                        index_argmax))

# Построение модели + возврат результатов классификации тестовых пользователей
def fit_predict(params, num_trees, train, test, target):
    params['learning_rate'] = params['eta']
    clf = xgb.train(params, xgb.DMatrix(train.values, target, feature_names=list(train.columns)),
                    num_boost_round=num_trees, maximize=True)
    y_pred = clf.predict(xgb.DMatrix(test.values, feature_names=list(train.columns)))
    submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])

    joblib.dump(clf, MODEL_PATH)
    return clf, submission

params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    
    'gamma': 0,
    'lambda': 0,
    'alpha': 0,
    'min_child_weight': 0,
    
    'eval_metric': 'auc',
    'objective': 'binary:logistic' ,
    'booster': 'gbtree',
    'njobs': -1,
    'tree_method': 'approx'
}

trans_code_to_type = {'1': 'покупки', '2': 'переводы',
                      '4': 'банковские услуги',
                      '6': 'возвраты', '7': 'взнос наличных и пополенения',
                      '8': 'прочее'}

for df in [transactions_train, transactions_test]:
    df['weekday'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    df['month'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) // 30)
    df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['night'] = df['hour'].between(0, 5).astype(int)
    df['morning'] = df['hour'].between(6, 11).astype(int)
    df['afternoon'] = df['hour'].between(12, 17).astype(int)
    df['evening'] = df['hour'].between(18, 23).astype(int)

    for category in pd.unique(df['mcc_category']):
        df[category] = df['mcc_category'].isin([category])
    
    df['trans_type_new'] = [trans_code_to_type[str(i)[0]] for i in df['trans_type']]
    
def features_creation(x): 
    features = []
    #features.append(pd.Series(x[x['amount']>0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
    #                                                    .add_prefix('positive_transactions_')))
    #features.append(pd.Series(x[x['amount']<0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
    #                                                    .add_prefix('negative_transactions_')))
    features.append(pd.Series(x[x['amount']>0]['amount'].agg(['min', 'max', 'median', 'std'])\
                                                        .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount']<0]['amount'].agg(['min', 'max', 'median', 'std'])\
                                                        .add_prefix('negative_transactions_')))
    
    #for time in ['weekday', 'month', 'hour', 'night', 'morning', 'afternoon', 'evening']:
    #    features.append(pd.Series(x[x[time] == True]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
    #                                                    .add_prefix(f'{time}_')))
    for time in ['weekday', 'month', 'hour', 'night']:
        features.append(pd.Series(x[x[time] == True]['amount'].agg(['min', 'mean','count'])\
                                                        .add_prefix(f'{time}_')))

    for category in ['Авто', 'Одежда', 'Красота', 'Супермаркеты', 'Развлечения', 'Здоровье', 'Техника']:
        features.append(pd.Series(x[x['mcc_category'] == category]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                                                        .add_prefix(f'{category}_')))
      
    #for type_ in trans_code_to_type.values():
    #    features.append(pd.Series(x[x['trans_type_new'] == type_]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
    #                            .add_prefix(f'trans_{type_}_')))
      
    #for city in pd.unique(df['trans_city']):
    #    features.append(pd.Series(x[x['trans_city'] == city]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
    #                                                    .add_prefix(f'{city}_')))
 
    return pd.concat(features)

data_train = transactions_train.groupby(transactions_train.index).apply(features_creation)
data_test = transactions_test.groupby(transactions_test.index).apply(features_creation)

target = data_train.join(gender_train, how='inner')['gender']
cv_score(params, data_train, target)

# Число деревьев для XGBoost имеет смысл выставлять по результатам на кросс-валидации 
clf, submission = fit_predict(params, 70, data_train, data_test, target)

submission.to_csv('result.csv')