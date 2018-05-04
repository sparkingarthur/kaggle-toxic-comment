from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import gensim
import pandas as pd
import numpy as np
import re
from keras.layers import Dense, Input, Flatten, AveragePooling1D,Concatenate, Convolution1D, MaxPooling1D, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, CuDNNGRU, Bidirectional, TimeDistributed
from keras.models import Model
from attention import Attention
from sklearn.metrics import f1_score
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_SENT_LENGTH = 100
MAX_SENTS = 30
VALIDATION_SPLIT = 0.01


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def convert_data(data, label=True):
    x = list(data[2])
    y = []
    if label:
        y = np.array([[d] for d in list(data[3])])
    return x, y

def han():
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(MAX_TOKENS+1, embedding_dim, weights=[embeddings])(sentence_input)
    l_lstm = Bidirectional(CuDNNGRU(100, return_sequences=True))(embedded_sequences)
    # l_att = Attention(100)(l_lstm)
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(100, return_sequences=True))(review_encoder)
    # l_att_sent = Attention(30)(l_lstm_sent)
    doc_modeling = Dropout(0.2)(l_lstm_sent)
    preds = Dense(1, activation='sigmoid')(doc_modeling)
    model = Model(review_input, preds)

    model.compile(loss='binary_crossentropy',
                  optimizer='adamax',
                  metrics=['acc', f1])
    return model


if __name__ == '__main__':
    batch_size = 64
    hidden_dim_1 = 200
    hidden_dim_2 = 100
    word2vec = gensim.models.Word2Vec.load("embeddings256_para").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    MAX_TOKENS = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]
    model = han()
    model.summary()
    print('Loading data...')
    old_data = pd.read_csv('./BDCI2017-360/pre/train_c.tsv', sep='\\t', engine='python', header=None, encoding='utf-8')
    data = pd.read_csv('./BDCI2017-360/semi/train_c.tsv', sep='\\t', engine='python', header=None, encoding='utf-8')


    # add pre-data
    temp = old_data.loc[:200000]
    old_pos = temp[temp[3]==1]
    old_data = old_data.loc[200000:]
    old_data = old_pos.append(old_data, ignore_index=True)
    train = old_data.append(data, ignore_index=True)
    train = train.drop_duplicates(subset=[0], keep='last').sample(frac=1)

    train_num = int(len(train)*0.9)
    print(len(train[train[3] == 1]))
    # exit(0)
    # _, y_train = convert_data(train)
    print('Convert training data...')
    x_train, y_train = convert_data(train)

    docs = []
    for text in x_train:
        doc_as_array = []
        sentences = re.split('[。！？]', text)[:MAX_SENTS]
        for s in sentences:
            if len(s.split()) > 2:
                doc_as_array.append([word2vec.vocab[token].index if token in word2vec.vocab else MAX_TOKENS for token in s])
        doc_as_array = pad_sequences(doc_as_array, MAX_SENT_LENGTH)
        docs.append(doc_as_array)
    docs = pad_sequences(docs, MAX_SENTS)

    x_train = docs
    print(docs.shape,y_train.shape)
    print("model fitting - Hierachical attention network")

    model.fit(x_train[:train_num], y_train[:train_num],
              batch_size=batch_size, verbose=1,
              epochs=7, validation_data=[x_train[train_num:], y_train[train_num:]])
    y_pred = model.predict(x_train[train_num:])
    y_pred = [1 if yp > 0.5 else 0 for yp in y_pred]
    print('f1:', f1_score(y_train[train_num:], y_pred))
