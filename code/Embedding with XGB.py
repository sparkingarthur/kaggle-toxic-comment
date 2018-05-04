# -*- coding: utf-8 -*-
"""
@author: Faron
"""
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

ID = 'id'
TARGET = 'loss'
DATA_DIR = "../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)

SEED = 0
NFOLDS = 5
NTHREADS = 4

xgb_params = {
    'seed': 0,
    'colsample_bytree': 1,
    'silent': 1,
    'subsample': 1.0,
    'learning_rate': 1.0,
    'objective': 'reg:linear',
    'max_depth': 100,
    'num_parallel_tree': 1,
    'min_child_weight': 250,
    'eval_metric': 'mae',
    'nthread': NTHREADS,
    'nrounds': 1
}


def get_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    y_train = train[TARGET].ravel()

    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)

    ntrain = train.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)

    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]

    train_test = train_test[cats]

    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

    x_train = np.array(train_test.iloc[:ntrain, :])
    x_test = np.array(train_test.iloc[ntrain:, :])

    return x_train, y_train, x_test


def get_oof(clf, x_train, y_train, x_test):
    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain,))

    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED).split(x_train)

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)

    clf.train(x_train, y_train)
    oof_test = clf.predict(x_test)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_sparse_ohe(x_train, x_test, min_obs=10):
    ntrain = x_train.shape[0]

    train_test = np.concatenate((x_train, x_test)).reshape(-1, )

    # replace infrequent values by nan
    val = dict((k, np.nan if v < min_obs else k) for k, v in dict(Counter(train_test)).items())
    k, v = np.array(list(zip(*sorted(val.items()))))
    train_test = v[np.digitize(train_test, k, right=True)]

    ohe = csr_matrix(pd.get_dummies(train_test, dummy_na=False, sparse=True))

    x_train_ohe = ohe[:ntrain, :]
    x_test_ohe = ohe[ntrain:, :]

    return x_train_ohe, x_test_ohe


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 1000)

    def train(self, x_train, y_train, x_valid=None, y_valid=None, sample_weights=None):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
    
    # pred_leaf=True => getting leaf indices
    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x), pred_leaf=True).astype(int)


x_train, y_train, x_test = get_data()

dtrain = xgb.DMatrix(x_train, label=y_train)

xg = XgbWrapper(seed=SEED, params=xgb_params)
xg_cat_embedding_train, xg_cat_embedding_test = get_oof(xg, x_train, y_train, x_test)

xg_cat_embedding_ohe_train, xg_cat_embedding_ohe_test = get_sparse_ohe(xg_cat_embedding_train, xg_cat_embedding_test)

print("OneHotEncoded XG-Embeddings: {},{}".format(xg_cat_embedding_ohe_train.shape, xg_cat_embedding_ohe_test.shape))
