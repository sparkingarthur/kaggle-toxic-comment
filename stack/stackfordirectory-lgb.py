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
from keras.layers.advanced_activations import PReLU
from keras import backend as K
from sklearn.metrics import roc_auc_score

import copy

import keras

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
np.random.seed(2017)
import random
random.seed(2018)

rootpath = "./cnn-rnn"
result_path = "./cnn-rnn/result"
cv_path = "./cnn-rnn/cv_result"
all_subs = os.listdir(result_path)
print(all_subs)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
subs = [pd.read_csv(os.path.join(result_path, f)) for f in all_subs]
cvs = [pd.read_csv(os.path.join(cv_path, f)) for f in all_subs]

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

y = train_df[list_classes].values

new_train_df = pd.DataFrame()
new_test_df = pd.DataFrame()
new_train_df['id'] = train_df['id']
new_test_df['id'] = test_df['id']

# for i in range(0,len(cvs)):
#     cv = cvs[i]
#     train_data = train_data.merge(cv,on=['id'])
#
# for i in range(0,len(subs)):
#     sub = subs[i]
#     test_data = test_data.merge(sub,on=['id'])
for cv in cvs:
    new_train_df = new_train_df.merge(cv, on=['id'])
for sub in subs:
    new_test_df = new_test_df.merge(sub,on=['id'])

cv_id = pd.read_csv('../input/cv_id_10.txt')
#print(new_train_df.head())
traincolumns = [column for column in new_train_df.columns if column not in ['id']]
print(len(traincolumns))
coidxs = []
for i in range(1,len(traincolumns)+1):
    coidxs.append(i)
cv_data = new_train_df.iloc[:,coidxs].values
test_data = new_test_df.iloc[:,coidxs].values

print(cv_data.shape,test_data.shape)

new_train_df['cv_id']=cv_id['cv_id_10']
new_test_df['cv_id']=-1

def buildstackingNN():
    cv_input = Input(shape=(len(traincolumns),), dtype='float32')
    merged = Dense(1000)(cv_input)
    relu = PReLU()(merged)
    #merged = concatenate([merged,tanh,relu])
    merged = Dropout(0.1)(relu)
    preds = Dense(6, activation='sigmoid')(merged)
    model = Model(inputs=[cv_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])
    # print(model.summary())
    return model

cv_models=[]
cv_results=[]
cv_scores=[]
cv_best_trees=[]
Kfold = 10
CV_RESULT_OUT = True
lgbmresult_pred = np.zeros(shape=(len(test_df), len(list_classes)))#store the result of test

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
    idx_train = new_train_df[new_train_df['cv_id'] != i].index
    idx_val = new_train_df[new_train_df['cv_id'] == i].index
    valid_id = new_train_df[new_train_df['cv_id'] == i]['id'].values
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

    lgbmcv_pred = np.zeros(shape=(len(valid_id), len(list_classes)))  # store the result of cv_results
    for j, labelx in enumerate(list_classes):  # fit for each type
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
        rdf = rdf.reindex(columns=list_classes)
        rdf['id'] = valid_id
        rdf[list_classes[0:6]] = lgbmcv_pred
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
index = 'LGBstack-cnn-rnn'
if CV_RESULT_OUT:
    pd.concat(cv_results).to_csv("../cv_result/cv_%.4f"% (avg_val_score)+str(index)+".csv",index=False)

print("predict begin....")
for cv in range(0,Kfold):
    print("predict fold:"+str(cv))
    models_6 = cv_models[cv]
    best_trees_6 = cv_best_trees[cv]
    for j, labelx in enumerate(list_classes):  # test for each type
        print("predict label:"+str(labelx))
        bstmodel = models_6[labelx]
        bst_it = best_trees_6[labelx]
        lgbmresult_pred[:, j] += bstmodel.predict(test_data, num_iteration=bst_it)
print("average results.....")
lgbmresult_pred = lgbmresult_pred / 10  #10-fold
print("new dataframe...")
ids = test_df['id']
submission = pd.DataFrame()
submission = submission.reindex(columns=['id']+list_classes)
submission['id'] = ids
submission[list_classes[0:6]] = lgbmresult_pred
print("write to result csv...")
submission.to_csv("../result/%.4f"% (avg_val_score)+str(index)+".csv", index=False)