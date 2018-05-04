import numpy as np
import pandas as pd
import copy

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,BatchNormalization
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

#read data
df_Train= pd.read_csv('../input/train.csv')
df_Test = pd.read_csv('../input/test.csv')
cv_id = pd.read_table('../input/cv_id.txt')
#preprocess & params
maxlen=200
max_features = 20000
batchsize = 32

df_Train['cv_id']=cv_id['cv_id']
df_Test['cv_id']=-1
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_cvclasses=["cv_toxic", "cv_severe_toxic", "cv_obscene", "cv_threat", "cv_insult", "cv_identity_hate"]

df_Train.fillna("unknown",inplace=True)
df_Test.fillna("unknown",inplace=True)
train_test=pd.concat([df_Train,df_Test],axis=0)
#may be some preprocess

train=train_test[train_test['cv_id']!=-1]
test=train_test[train_test['cv_id']==-1]

tokenizer = text.Tokenizer(num_words=max_features)
list_sentences_train_test = train["comment_text"].fillna("NA").values
tokenizer.fit_on_texts(list(list_sentences_train_test))

del train_test
#build model
def buildmodel():
    embed_size = 128
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #model.summary()
    return model

#train
from sklearn.metrics import log_loss
def myscore(y_true,y_pred):
    score = 0
    n_labels=y_true.shape[1]
    for i in range(0,n_labels):
        y_t=y_true[:,i]
        y_p=y_pred[:,i]
        score += log_loss(y_t,y_p,labels=[0,1])
    return score / n_labels

cv_models=[]
cv_results=[]
cv_scores=[]
for i in range(5):
    print("Fold:",i)
    train_d = train[train['cv_id']!=i]
    #train_d = train_d[:30]
    valid_d = train[train['cv_id']==i]
    #valid_d = valid_d[:30]

    list_sentences_train = train_d['comment_text'].values
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

    list_sentences_valid = valid_d['comment_text'].values
    list_tokenized_valied = tokenizer.texts_to_sequences(list_sentences_valid)

    x_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    y_train = train_d[list_classes]
    x_valid = sequence.pad_sequences(list_tokenized_valied,maxlen=maxlen)
    y_valid = valid_d[list_classes].values

    model=buildmodel()
    best=[100,0,0,0]#socre, epoch, model.copy , cv_result
    earlystop=10
    for epoch in range(1000):
        model.fit(x_train,y_train.values,batch_size=batchsize, epochs=1, verbose=1)
        r=model.predict(x_valid ,batch_size=batchsize)
        s=myscore(y_valid,r)
        print(i,epoch,s)
        if s < best[0]:# the smaller loss is better
            print("epoch " + str(epoch) + " improved from " + str(best[0]) + " to " + str(s))
            best=[s,epoch,copy.copy(model),r]
        if epoch-best[1]>earlystop:
            break
    #save cv_results
    tpd=pd.DataFrame()
    tpd['id'] = valid_d['id']
    for i in range(0, len(list_cvclasses)):
        cls = list_cvclasses[i]
        tpd[cls] = best[-1][:, i]
    cv_results.append(tpd)
    cv_models.append(best[2])
    cv_scores.append(best[0])

#test = test[:30]
list_sentences_test = test["comment_text"].values
list_tokenized_valied = tokenizer.texts_to_sequences(list_sentences_test)
x_test = sequence.pad_sequences(list_tokenized_valied, maxlen=maxlen)
r=[]
print(cv_scores,np.average(cv_scores))
print("prediction begin....")
for i in range(5):
    print("prediction "+ str(i))
    if len(r) == 0:
        r = cv_models[i].predict(x_test,batch_size=2048)
    else:
        r += cv_models[i].predict(x_test,batch_size=2048)

r /= 5
index = 'lstmbaseline0'

print("write files...")
pd.concat(cv_results).to_csv("../cv_result/nn_cv_"+str(index)+".csv",index=False)
sub = pd.DataFrame()
sub['id']=test['id']
for i in range(0, len(list_classes)):
    cls = list_classes[i]
    sub[cls] = r[:, i]
sub.to_csv("../result/nn_"+str(index)+".csv",index=False)

