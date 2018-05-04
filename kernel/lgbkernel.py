# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import wordbatch
import os
from wordbatch.extractors import WordBag, WordHash

import lightgbm as lgbm
import re
import time
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import nltk
import collections

from nltk.corpus import stopwords
import re

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'


def patchdata(merge):
    merge['comment_text'].fillna('comment_text', inplace=True)
    merge['uc'] = merge.apply(lambda rw: len(re.findall(r'[A-Z]', rw.comment_text)), axis=1)


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


def main():
    start_time = time.time()
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    train = pd.read_csv('../input/train.csv', encoding='utf-8')
    test = pd.read_csv('../input/test.csv', encoding='utf-8')
    label_colmn = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    nrow_train = train.shape[0]
    merge = pd.concat([train, test])
    patchdata(merge)

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0]
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)

    submission = test[['id']]
    test_id = test['id']
    MAX_FEATURES_ITEM_DESCRIPTION = 50000

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3))
    X_name = tv.fit_transform(merge['comment_text'])

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze = True
    X_wb_name = wb.fit_transform(merge['comment_text'])
    del (wb)
    X_wb_name = X_wb_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['uc']],
                                          sparse=True).values)
    sparse_merge = hstack((X_dummies, X_wb_name, X_name)).tocsr()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]

    print(sparse_merge.shape)

    gc.collect()
    # NFOLDS = 5
    # kfold =KFold(n_splits=NFOLDS, shuffle=True, random_state=32)
    # kf = kfold.split(X)

    params = {"objective": "binary",
              "boosting_type": "gbdt",
              "learning_rate": 0.1,
              "num_leaves": 32,
              "feature_fraction": 0.9,
              "tree_learner": "feature",
              "verbosity": -1,
              "max_bin": 255,
              "metric": 'auc',
              "nthread": -1
              }
    # X=load_npz('../sparse_matrix.npz')
    # X_test=load_npz('../sparse_matrixtest.npz')
    # save_npz('sparse_matrix.npz', X)
    # save_npz('sparse_matrixtest.npz', X_test)
    lgbmcv_pred = np.zeros(shape=(len(test_id), 6))

    # sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
    best_trees = []
    for j, labelx in enumerate(label_colmn):
        y = train.loc[:, labelx]
        print('predict ', labelx)
        X_train, X_validate, label_train, label_validate = train_test_split(X, y, test_size=0.33, random_state=42)
        dtrain = lgbm.Dataset(X_train, label_train)
        dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgbm.train(params, train_set=dtrain, num_boost_round=2000,
                         valid_sets=dvalid, verbose_eval=100, early_stopping_rounds=100)
        lgbmcv_pred[:, j] += bst.predict(X_test, num_iteration=bst.best_iteration)
        best_trees.append(bst.best_iteration)
    # print('[{}] Predict lgbm completed')
    submission = submission.reindex(columns=label_colmn)
    submission['id'] = test[['id']]
    submission[label_colmn[0:6]] = lgbmcv_pred
    submission.to_csv("lgbmsubmision.csv", index=False)


if __name__ == '__main__':
    mp.set_start_method('forkserver', True)
    main()