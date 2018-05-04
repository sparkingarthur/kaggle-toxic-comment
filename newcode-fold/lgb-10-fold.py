import lightgbm as lgb
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
import sys
import copy
import gc
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
from scipy import sparse

import re, string
np.random.seed(2018)


path = '../input/'
TRAIN_DATA_FILE = path + 'train_pre2.csv'
TEST_DATA_FILE = path + 'test_pre2.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
cv_id = pd.read_csv('../input/cv_id_10.txt')

train_df['cv_id']=cv_id['cv_id_10']
test_df['cv_id']=-1

# train_df = train_df[0:1000]
# test_df = test_df[0:1000]

all_df = pd.concat([train_df,test_df])

print('Number of rows and columns in the train data set:',train_df.shape)
print('Number of rows and columns in the test data set:',test_df.shape)


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split() # clean the repeated
#
vect_word = TfidfVectorizer(max_features=10000, lowercase=True, analyzer='word',tokenizer=tokenize,
                        stop_words= 'english',ngram_range=(1,1),dtype=np.float32)
vect_word1 = TfidfVectorizer(max_features=10000, lowercase=True, analyzer='word',tokenizer=tokenize,
                        stop_words= 'english',ngram_range=(1,2),dtype=np.float32)
vect_char = TfidfVectorizer(max_features=10000, lowercase=True, analyzer='char',
                        stop_words= 'english',ngram_range=(1,4),dtype=np.float32)

# vect_word = HashingVectorizer(n_features=10000, lowercase=True, analyzer='word',tokenizer=tokenize,
#                         stop_words= 'english',ngram_range=(1,1),dtype=np.float32)
# vect_word1 = HashingVectorizer(n_features=10000, lowercase=True, analyzer='word',tokenizer=tokenize,
#                         stop_words= 'english',ngram_range=(1,6),dtype=np.float32)
# vect_char = HashingVectorizer(n_features=10000, lowercase=True, analyzer='char',
#                         stop_words= 'english',ngram_range=(1,4),dtype=np.float32)

#vect_word.fit(list(train['comment_text']) + list(test['comment_text']))
#tr_vect = vect_word.fit_transform(train['comment_text'])

vect_word = vect_word.fit(all_df['comment_text'])
vect_word1 = vect_word1.fit(all_df['comment_text'])
vect_char = vect_char.fit(all_df['comment_text'])

del all_df
gc.collect()

ts_vect = vect_word.transform(test_df['comment_text'])
ts_vect1 = vect_word1.transform(test_df['comment_text'])
ts_vect_char = vect_char.transform(test_df['comment_text'])
gc.collect()

#x_train = sparse.hstack([tr_vect,tr_vect1, tr_vect_char])
x_test = sparse.hstack([ts_vect,ts_vect1, ts_vect_char])

# x_train = x_train.toarray()
# print(x_train[0])
# specify your configurations as a dict
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
CV_RESULT_OUT = True
cv_models=[]
cv_results=[]
cv_scores=[]
cv_best_trees=[]
Kfold = 10
label_colmn=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

lgbmresult_pred = np.zeros(shape=(len(test_df), len(label_colmn)))#store the result of test
# train
for cv in range(0,Kfold):
    print("fold:"+str(cv))
    train_data = train_df[train_df['cv_id'] != cv]
    valid_data = train_df[train_df['cv_id'] == cv]
    rid = valid_data['id']

    train_vw = vect_word.transform(train_data['comment_text'])
    train_vw1 = vect_word1.transform(train_data['comment_text'])
    train_char = vect_char.transform(train_data['comment_text'])

    valid_vw = vect_word.transform(valid_data['comment_text'])
    valid_vw1 = vect_word1.transform(valid_data['comment_text'])
    valid_char = vect_char.transform(valid_data['comment_text'])

    train_X = sparse.hstack([train_vw,train_vw1,train_char])
    valid_X = sparse.hstack([valid_vw, valid_vw1, valid_char])
    each_label_score=[]
    models_6={}
    best_trees_6={}

    lgbmcv_pred = np.zeros(shape=(len(rid), len(label_colmn))) # store the result of cv_results
    for j, labelx in enumerate(label_colmn): #fit for each type
        train_Y = train_data.loc[:, labelx]#label for train
        valid_Y = valid_data.loc[:, labelx]#label for valid
        dtrain = lgb.Dataset(train_X, train_Y)
        dvalid = lgb.Dataset(valid_X, valid_Y, reference=dtrain)
        lis={}
        bst = lgb.train(params, train_set=dtrain, num_boost_round=2000,evals_result=lis,
                         valid_sets=dvalid, verbose_eval=10, early_stopping_rounds=100)
        one_score = lis['valid_0']['auc']
        each_label_score.append(max(one_score))
        models_6[labelx]=bst
        best_trees_6[labelx]=bst.best_iteration
        if CV_RESULT_OUT:
            #print("for cv_result_out")
            lgbmcv_pred[:, j] = bst.predict(valid_X, num_iteration=bst.best_iteration)
    if CV_RESULT_OUT:
        rdf = pd.DataFrame()
        rdf = rdf.reindex(columns=label_colmn)
        rdf['id'] = rid
        rdf[label_colmn[0:6]]=lgbmcv_pred
        cv_results.append(rdf)
    cv_models.append(models_6)
    cv_best_trees.append(best_trees_6)
    cv_scores.append(np.mean(each_label_score))
#test
avg_val_score = np.average(cv_scores)
print(cv_scores,avg_val_score)

index='lgb-10-fold-2'

#保存模型在cv上的预测结果，做stacking的时候可以直接将文件作为一个特征merge进来
if CV_RESULT_OUT:
    pd.concat(cv_results).to_csv("../cv_result/cv_%.4f"% (avg_val_score)+str(index)+".csv",index=False)

ids = test_df['id']
print("predict begin....")
for cv in range(0,Kfold):
    models_6 = cv_models[cv]
    best_trees_6 = cv_best_trees[cv]
    for j, labelx in enumerate(label_colmn):  # test for each type
        bstmodel = models_6[labelx]
        bst_it = best_trees_6[labelx]
        lgbmresult_pred[:, j] += bstmodel.predict(x_test, num_iteration=bst_it)
print("average results.....")
lgbmresult_pred = lgbmresult_pred / 10  #10-fold
print("new dataframe...")
submission = pd.DataFrame()
submission = submission.reindex(columns=label_colmn)
submission['id'] = ids
submission[label_colmn[0:6]] = lgbmresult_pred
print("write to result csv...")
submission.to_csv("../result/%.4f"% (avg_val_score)+str(index)+".csv", index=False)






