import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import codecs
INPUT = './'

from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM, GRU, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler, Callback
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D,CuDNNGRU, TimeDistributed,Bidirectional,Flatten,GlobalAveragePooling1D
from keras.layers import Input, concatenate, multiply, maximum, add, average, merge, Lambda
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D,\
    Activation, LSTM, SimpleRNN, Bidirectional, MaxPooling1D, Flatten, Masking, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras import backend as K
from sklearn.metrics import roc_auc_score
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import random
max_features = 200000
maxlen = 200
SEED = 17
np.random.seed(SEED)
random.seed(SEED)
train = pd.read_csv(INPUT+"train_pre2.csv",encoding='utf-8')[:]
test = pd.read_csv(INPUT+"test_pre2.csv",encoding='utf-8')[:]

"""
data enhance
"""
import random
def sample_enhance(x):
    """
    sample index for enhance, 1/5 data every time
    """
    shuffle = []
    for i in range(0,len(x)):
        shuffle.append(random.randint(0,len(x)-1))
    return list(set(shuffle[:len(shuffle)//5]))

def en_shuffle(X_t, shu, x=-10):
    """
    x is the position start shuffle
    """
    trans = X_t[shu].transpose()
    np.random.shuffle(trans[x:])
    return trans.transpose()

def en_drop(X_t,shu):
    """
    a is the position drop
    """
    for i in shu:
        a = random.randint(2,10)
        m = X_t[i]
        X_t[i] = np.hstack([np.array([0]),m[0:-a],m[-(a-1):]])
    return X_t[shu]

def en_replace(X_t,shu):
    """
    a is the position to replace
    """
    x = maxlen
    for i in shu:
        a = random.randint(x-11,x-1)
        X_t[i][a] = X_t[i][a]+a
    return X_t[shu]


list_sentences_train = train["comment_text"].fillna("[na]").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_classes_pred = ["toxic_pred", "severe_toxic_pred", "obscene_pred", "threat_pred", "insult_pred", "identity_hate_pred"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("[na]").values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

#EMBEDDING_FILE='./input/wiki.en.vec'
EMBEDDING_FILE='./input/crawl-300d-2M.vec'
#EMBEDDING_FILE='./input/glove.840B.300d.txt'

EMBEDDING_MY = './vectors.txt'
embed_size = 300 # how big is each word vector
cn = 0
def get_coefs(word,*arr): 
    global cn
    cn += 1
    dict_v = np.asarray(arr, dtype='float32')
    if len(dict_v) != embed_size:
        dict_v = np.zeros((embed_size))
    return word, dict_v

embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
    values = line.split()
    word = ' '.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors.' % len(embeddings_index))


f_emb = codecs.open(EMBEDDING_MY,'r','utf-8')
emb_list = f_emb.readlines()
cn = 0
embeddings_index_my = dict(get_coefs(*o.strip().split()) for o in emb_list)
print(cn)
f_emb.close()

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(emb_mean,emb_std)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
no_embedding = 0
for word, i in word_index.items():
    if i >= nb_words: continue
    embedding_vector = embeddings_index.get(word)
    embedding_vector_my = embeddings_index_my.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    elif embedding_vector_my is not None: 
        embedding_matrix[i] = embedding_vector_my
    else: 
        print(word)
        no_embedding += 1
print(no_embedding,nb_words-no_embedding,len(word_index))



def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    """
    fix your model here
    """
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

batch_size = 32
epochs = 10

kfolds_num = 10
from sklearn.model_selection import KFold
from sklearn.utils import class_weight

num = "kernal_en"
if not os.path.isdir(INPUT+"models_5/"):
    os.mkdir(INPUT+"models_5/")
if not os.path.isdir(INPUT+"models_5/"+str(num)):
    os.mkdir(INPUT+"models_5/"+str(num))

file_path_best=INPUT+"models_5/"+str(num)+"/"+"weights_best"+str(num)+".hdf5"
skf = KFold(n_splits=kfolds_num, shuffle=True)
result_avg = []
cnt = 0
for train_idx, test_idx in skf.split(X_t):        
    X_train = X_t[train_idx]
    y_train = y[train_idx]
    X_dev = X_t[test_idx]
    y_val = y[test_idx]
    model = get_model()
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(train)/batch_size) * epochs
    #lr_init, lr_fin = 0.001, 0.0005
    lr_init, lr_fin = 0.001, 0.0001

    lr_decay = exp_decay(lr_init, lr_fin, steps)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
    checkpoint_best = ModelCheckpoint(file_path_best, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
    callbacks_list = [checkpoint_best, early]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=3, validation_data=(X_dev,y_val), callbacks=callbacks_list)
    
    X_train = X_t[train_idx]
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_shuffle(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_drop(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_replace(X_train,en_index)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_dev,y_val), callbacks=callbacks_list)
    X_train = X_t[train_idx]
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_shuffle(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_drop(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_replace(X_train,en_index)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_dev,y_val), callbacks=callbacks_list)
    
    X_train = X_t[train_idx]
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_shuffle(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_drop(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_replace(X_train,en_index)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_dev,y_val), callbacks=callbacks_list)
    
    X_train = X_t[train_idx]
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_shuffle(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_drop(X_train,en_index)
    en_index = sample_enhance(X_train)
    X_train[en_index] = en_replace(X_train,en_index)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_dev,y_val), callbacks=callbacks_list)

    X_train = X_t[train_idx]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs-5, validation_data=(X_dev,y_val), callbacks=callbacks_list)

    if os.path.isfile(file_path_best):
        #print('load ',file_path_best)
        model.load_weights(file_path_best)

    y_test = model.predict([X_te], batch_size=256, verbose=1)
    y_p = model.predict([X_dev], batch_size=256, verbose=1)
    result_avg.append(y_test)
    auc_row = 0
    for i in range(6):
        auc_row += roc_auc_score(y_val[:,i],y_p[:,i])
    print("now auc = {}".format(auc_row/6))
    cnt = cnt + 1

y_test = np.array(result_avg).mean(axis=0)
sample_submission = pd.read_csv(INPUT+"input/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv(INPUT+"baseline_986_enhcance_wiki"+str(num)+".csv", index=False)
