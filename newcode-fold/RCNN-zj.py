import numpy as np
import pandas as pd
import pickle
import gensim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, CuDNNLSTM, Bidirectional, Lambda, Input, TimeDistributed, concatenate, SimpleRNN
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"


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
        y = np.ravel(list(data[3]))
    return x, y


def def_model():
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    embedder = Embedding(MAX_TOKENS + 1, embedding_dim, weights=[embeddings], trainable=False)
    doc_embedding = embedder(document)
    l_embedding = embedder(left_context)
    r_embedding = embedder(right_context)

    # I use LSTM RNNs instead of vanilla RNNs as described in the paper.
    forward = CuDNNLSTM(hidden_dim_1, return_sequences=True)(l_embedding)  # See equation (1).
    backward = CuDNNLSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2).
    together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3).

    semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)  # See equation (4).

    # Keras provides its own max-pooling layers, but they cannot handle variable length input
    # (as far as I can tell). As a result, I define my own max-pooling layer here.
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)  # See equation (5).

    output = Dense(NUM_CLASSES, input_dim=hidden_dim_2, activation="sigmoid")(pool_rnn)  # See equations (6) and (7).

    model = Model(inputs=[document, left_context, right_context], outputs=output)
    model.compile(optimizer="adamax", loss="binary_crossentropy", metrics=["accuracy", f1])

    return model


if __name__ == '__main__':
    maxlen = 1000
    batch_size = 128
    word2vec = gensim.models.Word2Vec.load("embeddings256_para").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    MAX_TOKENS = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]
    hidden_dim_1 = 200
    hidden_dim_2 = 100
    NUM_CLASSES = 1

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

    print('Convert training data...')
    x_train, y_train = convert_data(train)

    bs = 128
    doc_as_array = []
    left_context_as_array = []
    right_context_as_array = []
    for text in x_train:
        tokens = text.split()
        tokens = [word2vec.vocab[token].index if token in word2vec.vocab else MAX_TOKENS for token in tokens]
        doc_as_array.append(np.array(tokens))
        # We shift the document to the right to obtain the left-side contexts.
        left_context_as_array.append(np.array([MAX_TOKENS] + tokens[:-1]))
        # We shift the document to the left to obtain the right-side contexts.
        right_context_as_array.append(np.array(tokens[1:] + [MAX_TOKENS]))
    doc_as_array = sequence.pad_sequences(doc_as_array, maxlen)
    left_context_as_array = sequence.pad_sequences(left_context_as_array, maxlen)
    right_context_as_array = sequence.pad_sequences(right_context_as_array, maxlen)
    print(doc_as_array.shape, left_context_as_array.shape, right_context_as_array.shape)
    docs_train = [doc_as_array[:train_num], left_context_as_array[:train_num], right_context_as_array[:train_num]]
    docs_val = [doc_as_array[train_num:], left_context_as_array[train_num:], right_context_as_array[train_num:]]
    model = def_model()
    hist = model.fit(docs_train, y_train[:train_num], batch_size=bs, epochs=10,
                     verbose=1, validation_data=[docs_val, y_train[train_num:]])
    y_pred = model.predict(docs_val)
    y_pred = [1 if yp > 0.5 else 0 for yp in y_pred]
    print('f1:', f1_score(y_train[train_num:], y_pred))