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
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,GRU,Conv1D,GlobalMaxPool1D,MaxPooling1D,CuDNNGRU,TimeDistributed, Lambda, multiply,concatenate,CuDNNLSTM,Bidirectional
from keras.layers import SpatialDropout1D,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers.merge import concatenate
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import sys
import copy

from sklearn.metrics import roc_auc_score


from keras import backend as K
from keras.engine.topology import Layer
# from keras import initializations
from keras import initializers, regularizers, constraints
from attavglayer import AttentionWeightedAverage
from keras.optimizers import Adam,Nadam

os.environ['CUDA_VISIBLE_DEVICES']='1,2'
np.random.seed(2017)
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        print(input_shape)
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

path = '../input/'

TRAIN_DATA_FILE = path + 'train_pre2.csv'
TEST_DATA_FILE = path + 'test_pre2.csv'

MAX_SEQUENCE_LENGTH = 600
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.10

act = 'relu'
########################################
## index word vectors
########################################

print('Indexing word vectors')
# Glove Vectors
embeddings_index_glove = {}
embeddings_index_fasttext={}
embeddings_index_word2vec={}
def loadGloveVector():
    # Glove Vectors
    f = open('../input/glove.840B.300d.txt')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index_glove[word] = coefs
    f.close()
    print('Glove pretrained vectors loaded')

LOADWORD2VEC = False

def loadWord2Vector():
    # Word2Vec Vectors
    import gensim.models.word2vec as w2v # word2vec model
    #word2vect = w2v.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #word2vect = w2v.Word2Vec.load('word2vec_.model')
    word2vect = w2v.Word2Vec.load('../input/word2vec_128_e.model')
    print('Word2vec pretrained vectors loaded')
    word_vectors = word2vect.wv
    del word2vect
    words = word_vectors.index2word
    #print(word_vectors['fuck'])
    for i in range(len(words)):
        word = words[i]
        coefs = np.asarray(word_vectors[word],dtype='float32')
        embeddings_index_word2vec[word] = coefs
        #print(str(i)+": "+words[i])
    del word_vectors
def loadFastTextVector():
    f = open('../input/crawl-300d-2M.vec', encoding='utf-8')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index_fasttext[word] = coefs
    f.close()
    print('fasttext pretrained vectors loaded')



loadGloveVector()
loadFastTextVector()
if LOADWORD2VEC:
    loadWord2Vector()

print('Total %s word glove vectors.' % len(embeddings_index_glove))
print('Total %s word2vec vectors.' % len(embeddings_index_glove))
print('Total %s word fasttext vectors.' % len(embeddings_index_fasttext))
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
cv_id = pd.read_csv('../input/cv_id_10.txt')

train_df['cv_id']=cv_id['cv_id_10']
test_df['cv_id']=-1

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
#special_character_removal = re.compile(r'[^a-z\d,.! ]', re.IGNORECASE)
# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

#split the punctuation
chact = re.compile(r'([^a-z]+)', re.IGNORECASE)

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46371

# Find substrings that consist of `nchars` non-space characters
# and that are repeated at least `ntimes` consecutive times,
# and replace them with a single occurrence.
# Examples:
# abbcccddddeeeee -> abcde (nchars = 1, ntimes = 2)
# abbcccddddeeeee -> abbcde (nchars = 1, ntimes = 3)
# abababcccababab -> abcccab (nchars = 2, ntimes = 2)
def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)

def substitute_repeats(text, ntimes=3):
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    return text

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # text = chact.split(text)
    # text = " ".join(text)

    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Remove repeated characters
    text = substitute_repeats(text)
    # Replace Numbers
    # text = replace_numbers.sub('n', text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)

list_sentences_train = train_df["comment_text"].fillna("NA").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("NA").values

comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))

test_comments = []
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)  #fenci
tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # all train data
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH) # all test data
print('Shape of test_data tensor:', test_data.shape)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix_glove = np.zeros((nb_words, 300))
embedding_matrix_fasttext = np.zeros((nb_words,300))
if LOADWORD2VEC:
    embedding_matrix_word2vec =np.zeros((nb_words,128))
print("Preparing embedding matrix....")
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector_glove = embeddings_index_glove.get(word)
    if embedding_vector_glove is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_glove[i] = embedding_vector_glove

    embedding_vector_fasttext = embeddings_index_fasttext.get(word)
    if embedding_vector_fasttext is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_fasttext[i] = embedding_vector_fasttext
    if LOADWORD2VEC:
        embedding_vector_word2vec = embeddings_index_word2vec.get(word)
        if embedding_vector_word2vec is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_word2vec[i] = embedding_vector_word2vec

print('Null word glove embeddings: %d' % np.sum(np.sum(embedding_matrix_glove, axis=1) == 0))
print('Null word fasttext embeddings: %d' % np.sum(np.sum(embedding_matrix_fasttext, axis=1) == 0))
if LOADWORD2VEC:
    print('Null word word2vec embeddings: %d' % np.sum(np.sum(embedding_matrix_word2vec, axis=1) == 0))

if LOADWORD2VEC:
    print("Preparing fasttext embedding matrix....")
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index_fasttext.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_fasttext[i] = embedding_vector

print('Null word fasttext embeddings: %d' % np.sum(np.sum(embedding_matrix_fasttext, axis=1) == 0))

def buildmodel():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer_glove = Embedding(nb_words,
                                300,
                                weights=[embedding_matrix_glove],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences_glove = embedding_layer_glove(comment_input)
    #embedded_sequences_glove = Activation('tanh')(embedded_sequences_glove)
    #embedded_sequences_glove = SpatialDropout1D(0.05)(embedded_sequences_glove)
    embedding_layer_fasttext = Embedding(nb_words,
                                      300,
                                      weights=[embedding_matrix_fasttext],
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      trainable=False)
    embedded_sequences_fasttext = embedding_layer_fasttext(comment_input)
    #embedded_sequences_fasttext = Activation('tanh')(embedded_sequences_fasttext)
    #embedded_sequences_fasttext = SpatialDropout1D(0.05)(embedded_sequences_fasttext)
    if LOADWORD2VEC:
        embedding_layer_word2vec = Embedding(nb_words,
                                             128,
                                             weights=[embedding_matrix_word2vec],
                                             input_length=MAX_SEQUENCE_LENGTH,
                                             trainable=True)
        embedded_sequences_word2vec = embedding_layer_word2vec(comment_input)
        embedded_sequences_word2vec = Dropout(0.01)(embedded_sequences_word2vec)
    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    # embedded_sequences = Dropout(0.1)(embedded_sequences)
    x = SpatialDropout1D(0.2)(embedded_sequences_glove)
    x = Bidirectional(CuDNNGRU(32, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(x)

    y = SpatialDropout1D(0.2)(embedded_sequences_fasttext)
    y = Bidirectional(CuDNNGRU(32, return_sequences=True))(y)
    y = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    y = concatenate([avg_pool, max_pool])
    y = Dropout(0.1)(y)

    merged = keras.layers.concatenate([x, y])
    merged = Dense(128)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.15)(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    # print(model.summary())
    return model

cv_models=[]
cv_results=[]
cv_scores=[]
Kfold = 10
for i in range(0,Kfold):
    idx_train = train_df[train_df['cv_id'] != i].index
    idx_val = train_df[train_df['cv_id'] == i].index
    valid_id = train_df[train_df['cv_id'] == i]['id'].values
    data_train = data[idx_train]
    labels_train = y[idx_train]
    data_val = data[idx_val]
    labels_val = y[idx_val]
    print("fold %d"%(i))
    print("train_shape")
    print(data_train.shape, labels_train.shape)
    print("val_shape")
    print(data_val.shape, labels_val.shape)
    model = buildmodel()
    best = [-1, 0, 0, 0]  # socre, epoch, model.copy , cv_result
    earlystop = 5
    for epoch in range(1000):
        model.fit(data_train,labels_train,batch_size=512, epochs=1, verbose=1)
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
        r = cv_models[i].predict(test_data,batch_size=512)
    else:
        r += cv_models[i].predict(test_data,batch_size=512)
r /= Kfold
index = 'multiemb-rnn-cnn-10-fold-0'

print("write files...")
pd.concat(cv_results).to_csv("../cv_result/%.4fnn_cv_"% (avg_val_score)+str(index)+".csv",index=False)

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = r

sample_submission.to_csv("../result/%.4f_nn_"% (avg_val_score)+ index+".csv",index=False)

