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
np.random.seed(2018)

rootpath = "./temp"
result_path = "./temp/result"
cv_path = "./temp/cv_result"
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
Kfold = 10

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
    model = buildstackingNN()
    best = [-1, 0, 0, 0]  # socre, epoch, model.copy , cv_result
    earlystop = 20
    for epoch in range(10000):
        model.fit(data_train,labels_train,batch_size=512, epochs=1, verbose=0)
        r = model.predict(data_val ,batch_size=512)
        s = roc_auc_score(labels_val,r)
        print(i,epoch,s)
        if s > best[0]:# the bigger is better
            print("epoch " + str(epoch) + " improved from " + str(best[0]) + " to " + str(s))
            best = [s,epoch,copy.copy(model),r]
        if epoch-best[1]>earlystop:
            break
    #save cv_results
    tpd=pd.DataFrame(columns=[['id']+list_classes])
    tpd['id'] = valid_id
    tpd[list_classes] = best[-1]
    cv_results.append(tpd)
    cv_models.append(best[2])
    cv_scores.append(best[0])

r=[]
avg_val_score = np.average(cv_scores)
print(cv_scores,avg_val_score)
print("prediction begin....")

for i in range(Kfold):
    print("prediction "+ str(i))
    if len(r) == 0:
        r = cv_models[i].predict(test_data,batch_size=1024)
    else:
        r += cv_models[i].predict(test_data,batch_size=1024)
r /= Kfold
index = 'NNstack-capsule-charword'

print("write files...")
pd.concat(cv_results).to_csv("../cv_result/%.4fnn_cv_"% (avg_val_score)+str(index)+".csv",index=False)

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = r

sample_submission.to_csv("../result/%.4f_nn_"% (avg_val_score)+ index+".csv",index=False)