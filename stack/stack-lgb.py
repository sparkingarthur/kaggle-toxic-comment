import pandas as pd
import numpy as np
import re
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Embedding, Dropout, Activation,GRU,Conv1D,GlobalMaxPool1D,MaxPooling1D,CuDNNGRU,TimeDistributed, Lambda, multiply,concatenate,CuDNNLSTM,Bidirectional
from keras.layers import SpatialDropout1D,GlobalMaxPool1D,GlobalAvgPool1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from sklearn.metrics import roc_auc_score

import copy

import keras

import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
np.random.seed(2018)


MAX_SEQUENCE_LENGTH = 600
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300


cv_1 = pd.read_csv('./other/cv_result/0.9881nn_cv_MLP-fasttext-10-fold-0.csv')
cv_2 = pd.read_csv('./cv_0.9853lgb-10-fold-0.csv')
t_1 = pd.read_csv('./other/result/0.9881_nn_MLP-fasttext-10-fold-0.csv')
t_2 = pd.read_csv('./other/result/mylgb-10-fold-0-985-lb980.csv')
cv_3 = pd.read_csv('./other/cv_result/lvl0_wordbatch_clean_oof1.csv')
t_3 = pd.read_csv('./other/result/lvl0_wordbatch_clean_sub1.csv')


labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
cv_3.drop(labels,axis=1,inplace=True)
#print(cv_3.head())

cv_1 = cv_1.reindex(columns=['id']+labels)
cv_2 = cv_2.reindex(columns=['id']+labels)
t_1 = t_1.reindex(columns=['id']+labels)
t_2 = t_2.reindex(columns=['id']+labels)

#print(cv_2.head())
cv1_columns = [x+'_cv1'for x in labels]
cv_1.columns = ['id'] + cv1_columns
t1_columns = [x+'_t1'for x in labels ]
t_1.columns = ['id'] + t1_columns
#print(t_1.columns)

cv2_columns = [x+'_cv2'for x in labels]
cv_2.columns = ['id'] + cv2_columns
t2_columns = [x+'_t2'for x in labels ]
t_2.columns = ['id']+ t2_columns

cv3_columns = [x+'_cv3'for x in labels]
cv_3.columns = ['id'] + cv3_columns
t3_columns = [x+'_t3'for x in labels ]
t_3.columns = ['id']+ t3_columns

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#print(len(test_df))

train_df = train_df.merge(cv_1,on=['id'])
train_df = train_df.merge(cv_2,on=['id'])
train_df = train_df.merge(cv_3,on=['id'])

#print(train_df.head(5))
test_df = test_df.merge(t_1,on=['id'])
test_df = test_df.merge(t_2,on=['id'])
test_df = test_df.merge(t_3,on=['id'])

#print(len(test_df))
#print(test_df.head())
STACK_WITH_ORIGIN = False
STACK_METHOD = 'NN'

print("Stack with orgin:", str(STACK_WITH_ORIGIN))
cv_id = pd.read_csv('../input/cv_id_10.txt')

train_df['cv_id']=cv_id['cv_id_10']
test_df['cv_id']=-1
y = train_df[labels].values

cv1_x = train_df[cv1_columns].values
cv2_x = train_df[cv2_columns].values
cv3_x = train_df[cv3_columns].values

cv_data = np.hstack((cv1_x,cv2_x,cv3_x))
print(cv_data.shape)

t1_x = test_df[t1_columns].values
t2_x = test_df[t2_columns].values
t3_x = test_df[t3_columns].values
test_data = np.hstack((t1_x,t2_x,t3_x))

list_sentences_test = test_df["comment_text"].fillna("NA").values

cv_models=[]
cv_results=[]
cv_scores=[]
cv_best_trees=[]
Kfold = 10
CV_RESULT_OUT = True
lgbmresult_pred = np.zeros(shape=(len(test_df), len(labels)))#store the result of test
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    #'max_depth':5,
    'num_leaves': 50,
    'learning_rate': 0.04,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 6,
    'verbose': 0,
    #'is_unbalance':'true',
    'num_threads': -1
}
for i in range(0,Kfold):
    idx_train = train_df[train_df['cv_id'] != i].index
    idx_val = train_df[train_df['cv_id'] == i].index
    valid_id = train_df[train_df['cv_id'] == i]['id'].values
    print("fold %d" % (i))
    data_train = cv_data[idx_train]
    data_val = cv_data[idx_val]
    labels_train = y[idx_train]
    labels_val = y[idx_val]
    print("cv_train_shape")
    print(data_train.shape)
    print("cv_val_shape")
    print(data_val.shape)
    each_label_score = []
    models_6 = {}
    best_trees_6 = {}

    lgbmcv_pred = np.zeros(shape=(len(valid_id), len(labels)))  # store the result of cv_results
    for j, labelx in enumerate(labels):  # fit for each type
        print("modeling columns:"+str(labelx))
        train_Y = labels_train[:, j]  # label for train
        valid_Y = labels_val[:, j]  # label for valid
        dtrain = lgb.Dataset(data_train, train_Y)
        dvalid = lgb.Dataset(data_val, valid_Y, reference=dtrain)
        lis = {}
        bst = lgb.train(params, train_set=dtrain, num_boost_round=2000, evals_result=lis,
                        valid_sets=dvalid, verbose_eval=False, early_stopping_rounds=100)
        one_score = lis['valid_0']['auc']
        labelcvscore = max(one_score)
        each_label_score.append(labelcvscore)
        print("label cv score1:" + str(labelcvscore))
        models_6[labelx] = bst
        best_trees_6[labelx] = bst.best_iteration
        if CV_RESULT_OUT:
            # print("for cv_result_out")
            lgbmcv_pred[:, j] = bst.predict(data_val, num_iteration=bst.best_iteration)
    if CV_RESULT_OUT:
        rdf = pd.DataFrame()
        rdf = rdf.reindex(columns=labels)
        rdf['id'] = valid_id
        rdf[labels[0:6]] = lgbmcv_pred
        cv_results.append(rdf)
    cv_models.append(models_6)
    cv_best_trees.append(best_trees_6)
    foldcv_score = np.mean(each_label_score)
    cv_scores.append(foldcv_score)
    print("cv score:"+str(foldcv_score))

r=[]
avg_val_score = np.average(cv_scores)
print(cv_scores,avg_val_score)
#保存模型在cv上的预测结果，做stacking的时候可以直接将文件作为一个特征merge进来
index = 'LGBstack-mlp-lgb-fm'
if CV_RESULT_OUT:
    pd.concat(cv_results).to_csv("../cv_result/cv_%.4f"% (avg_val_score)+str(index)+".csv",index=False)

print("predict begin....")
for cv in range(0,Kfold):
    print("predict fold:"+str(cv))
    models_6 = cv_models[cv]
    best_trees_6 = cv_best_trees[cv]
    for j, labelx in enumerate(labels):  # test for each type
        print("predict label:"+str(labelx))
        bstmodel = models_6[labelx]
        bst_it = best_trees_6[labelx]
        lgbmresult_pred[:, j] += bstmodel.predict(test_data, num_iteration=bst_it)
print("average results.....")
lgbmresult_pred = lgbmresult_pred / 10  #10-fold
print("new dataframe...")
ids = test_df['id']
submission = pd.DataFrame()
submission = submission.reindex(columns=['id']+labels)
submission['id'] = ids
submission[labels[0:6]] = lgbmresult_pred
print("write to result csv...")
submission.to_csv("../result/%.4f"% (avg_val_score)+str(index)+".csv", index=False)