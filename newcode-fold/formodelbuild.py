#coding=utf-8
########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,GRU,Conv1D,GlobalMaxPool1D,MaxPooling1D,CuDNNGRU,TimeDistributed, Lambda, multiply,concatenate,CuDNNLSTM,Bidirectional,SpatialDropout1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from attavglayer import AttentionWeightedAverage
from keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU
import keras
import sys
import copy

from sklearn.metrics import roc_auc_score

from keras import backend as K
from keras.engine.topology import Layer
# from keras import initializations
from keras import initializers, regularizers, constraints

MAX_SEQUENCE_LENGTH = 600
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS, 100000)
def modelbuild1():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                #weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = Activation('tanh')(embedded_sequences)
    embedded_sequences = SpatialDropout1D(0.1, name='embed_drop')(embedded_sequences)
    x = Bidirectional(GRU(32, return_sequences=True))(embedded_sequences)
    x1 = Dropout(0.1)(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x1)
    x2 = Dropout(0.1)(x)
    merged = concatenate([x1,x2])
    merged = AttentionWeightedAverage()(merged)
    #merged = Dense(250, activation='relu')(x)
    merged = Dropout(0.15)(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def buildMLP():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                #weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    # x = SpatialDropout1D(0.1)(embedded_sequences)
    x = Dense(30, activation='sigmoid')(embedded_sequences)
    x3 = Dense(470)(embedded_sequences)
    x3 = PReLU()(x3)
    x1 = concatenate([x, x3])
    x1 = Dropout(0.02)(x1)
    x = Dense(256, activation='sigmoid')(x1)
    x2 = Dense(11, activation='linear')(x1)
    x3 = Dense(11)(x1)
    x3 = PReLU()(x3)
    x1 = concatenate([x, x2, x3])
    x = Dropout(0.1)(x1)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    merged = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(merged)
    preds = Dense(6, activation='sigmoid')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def buildmulticnn():
    filternumber = 64
    filter_sizes = [1, 2, 5]
    dilation = [1, 2, 3]
    l2_weight_decay = 0.00001
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                #weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    # embedded_sequences = GaussianNoise(0.1)(embedded_sequences)
    embedded_sequences = Dropout(0.15)(embedded_sequences)
    convs = []
    for (ft_sz, dile) in zip(filter_sizes, dilation):
        conv = Conv1D(filternumber, kernel_size=ft_sz, padding='same')(embedded_sequences)
        conv = PReLU()(conv)
        conv = Dropout(0.1)(conv)
        # f_W = Flatten()(conv)
        # f_W = Dense(filternumber, activation='relu')(f_W)
        # f_W = Dropout(0.1)(f_W)
        # conv = BatchNormalization()(conv)
        # conv = MaxPooling1D(pool_size=2, strides=1, padding='valid')(conv)
        # conv = Conv1D(filternumber, kernel_size=ft_sz, padding='valid', activation='tanh')(conv)
        # conv = Dropout(0.1)(conv)
        # conv = BatchNormalization()(conv)
        conv1 = GlobalMaxPool1D()(conv)
        conv2 = AttentionWeightedAverage()(conv)
        conv = keras.layers.concatenate([conv1, conv2])
        convs.append(conv)

    merged = keras.layers.concatenate(convs)
    # merged = Flatten()(merged)
    # merged = Dropout(rate_drop_dense)(merged)
    merged = Dense(128)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.15)(merged)
    #merged = GaussianNoise(0.1)(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def buildmultiembedding():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer_glove = Embedding(nb_words,
                                      300,
                                      #weights=[embedding_matrix_glove],
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      trainable=False)
    embedded_sequences_glove = embedding_layer_glove(comment_input)

    embedding_layer_fasttext = Embedding(nb_words,
                                         300,
                                         #weights=[embedding_matrix_fasttext],
                                         input_length=MAX_SEQUENCE_LENGTH,
                                         trainable=False)
    embedded_sequences_fasttext = embedding_layer_fasttext(comment_input)

    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(32, return_sequences=True,dropout=0.3))(embedded_sequences_glove)
    x = Dropout(0.02)(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = Dropout(0.1)(x)

    y = Bidirectional(LSTM(32, return_sequences=True))(embedded_sequences_fasttext)
    y = Dropout(0.05)(y)
    y = Bidirectional(LSTM(32, return_sequences=True))(y)
    y = Dropout(0.15)(y)

    x = concatenate([x, y])
    x = TimeDistributed(Dense(128, activation='tanh'))(x)
    # x = Dropout(rate_drop_dense)(x)
    # x = Attention(MAX_SEQUENCE_LENGTH)(x)
    x = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(x)
    merged = Dense(250, activation='relu')(x)
    merged = Dropout(0.5)(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def buildhanxc():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                               # weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    x = SpatialDropout1D(0.2)(embedded_sequences)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(x)
    # l_att = Attention(100)(l_lstm)
    sentEncoder = Model(comment_input, l_lstm)

    review_encoder = TimeDistributed(sentEncoder)(embedded_sequences)
    review_encoder = Lambda(lambda x: K.max(x, axis=1))(review_encoder)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=False))(review_encoder)
    # l_att_sent = Attention(30)(l_lstm_sent)
    doc_modeling = Dropout(0.2)(l_lstm_sent)
    preds = Dense(6, activation='sigmoid')(doc_modeling)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    #optimizer = Adam(lr=3e-4)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model
def buildmodelpostpool():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                #weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(32, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # merged = concatenate([avg_pool, max_pool])
    # merged = Dropout(0.1)(merged)
    x = Dropout(0.1)(x)
    merged = Dense(6, activation='sigmoid')(x)
    #avg_pool = GlobalAveragePooling1D()(preds)
    preds = GlobalMaxPooling1D()(merged)
    # merged = concatenate([avg_pool, max_pool])


    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    #optimizer = Adam(lr=3e-4)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def buildmulembbigru():
        comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer_glove = Embedding(nb_words,
                                          300,
                                          #weights=[embedding_matrix_glove],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)
        embedded_sequences_glove = embedding_layer_glove(comment_input)
        # embedded_sequences_glove = Activation('tanh')(embedded_sequences_glove)
        embedded_sequences_glove = SpatialDropout1D(0.1)(embedded_sequences_glove)
        embedding_layer_fasttext = Embedding(nb_words,
                                             300,
                                             #weights=[embedding_matrix_fasttext],
                                             input_length=MAX_SEQUENCE_LENGTH,
                                             trainable=False)
        embedded_sequences_fasttext = embedding_layer_fasttext(comment_input)
        # embedded_sequences_fasttext = Activation('tanh')(embedded_sequences_fasttext)
        # embedded_sequences_fasttext = SpatialDropout1D(0.1)(embedded_sequences_fasttext)
        x1 = Bidirectional(CuDNNGRU(32, return_sequences=True))(embedded_sequences_glove)
        # x1 = Activation('tanh')(x1)
        x = Dropout(0.2)(x1)
        y1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(embedded_sequences_fasttext)
        y1 = Dropout(0.1)(y1)
        y2 = Bidirectional(CuDNNGRU(64, return_sequences=True, go_backwards=True))(embedded_sequences_fasttext)
        y = concatenate([y1, y2])
        x = concatenate([x, y])
        x = TimeDistributed(Dense(128, activation='tanh'))(x)
        x1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(x)
        x2 = GlobalMaxPool1D()(x)
        merged = concatenate([x1, x2])
        merged = Dropout(0.1)(merged)
        preds = Dense(6, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[comment_input],
                      outputs=preds)
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=1e-3),
                      metrics=['accuracy'])
        print(model.summary())
        return model
def build_char_word_rnn():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                #weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    x = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    char_input = Input(shape=(1500,))
    embedding_layer = Embedding(500,
                                256)
    char_emb_sequences = embedding_layer(char_input)
    convs = []
    for i in range(1, 8):
        conv = Conv1D(32, kernel_size=i, padding='valid')(char_emb_sequences)
        conv = PReLU()(conv)
        conv = Dropout(0.1)(conv)
        conv = GlobalMaxPooling1D()(conv)
        convs.append(conv)
    char_merged = concatenate(convs)
    merged = concatenate([avg_pool, max_pool, char_merged])
    merged = Dropout(0.1)(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input,char_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])
    print(model.summary())
    return model

build_char_word_rnn()
