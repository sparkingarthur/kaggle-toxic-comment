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
os.environ['CUDA_VISIBLE_DEVICES']='7'
np.random.seed(2018)


MAX_SEQUENCE_LENGTH = 600
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300


cv_1 = pd.read_csv('./0.9881nn_cv_MLP-fasttext-10-fold-0.csv')
cv_2 = pd.read_csv('./cv_0.9853lgb-10-fold-0.csv')
t_1 = pd.read_csv('./0.9881_nn_MLP-fasttext-10-fold-0.csv')
t_2 = pd.read_csv('./mylgb-10-fold-0-985-lb980.csv')

labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
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

train_df = pd.read_csv('../input/train_pre2.csv')
test_df = pd.read_csv('../input/test_pre2.csv')

#print(len(test_df))

train_df = train_df.merge(cv_1,on=['id'])
train_df = train_df.merge(cv_2,on=['id'])

#print(train_df.head(5))
test_df = test_df.merge(t_1,on=['id'])
test_df = test_df.merge(t_2,on=['id'])

#print(len(test_df))
#print(test_df.head())
STACK_WITH_ORIGIN = False
STACK_METHOD = 'NN'

print("Stack with orgin:", str(STACK_WITH_ORIGIN))
cv_id = pd.read_csv('../input/cv_id_10.txt')

train_df['cv_id']=cv_id['cv_id_10']
test_df['cv_id']=-1

embeddings_index = {}
def loadGloveVector():
    # Glove Vectors
    f = open('../input/glove.840B.300d.txt')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Glove pretrained vectors loaded')


def loadWord2Vector():
    # Word2Vec Vectors
    import gensim.models.word2vec as w2v  # word2vec model
    # word2vect = w2v.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # word2vect = w2v.Word2Vec.load('word2vec_.model')
    word2vect = w2v.Word2Vec.load('word2vec_128_e.model')
    print('Word2vec pretrained vectors loaded')
    word_vectors = word2vect.wv
    del word2vect
    words = word_vectors.index2word
    # print(word_vectors['fuck'])
    for i in range(len(words)):
        word = words[i]
        coefs = np.asarray(word_vectors[word], dtype='float32')
        embeddings_index[word] = coefs
        # print(str(i)+": "+words[i])
    del word_vectors


def loadFastTextVector():
    f = open('../input/crawl-300d-2M.vec', encoding='utf-8')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('fasttext pretrained vectors loaded')

# loadvector = "word2vec"
# loadvector = "glove"
loadvector = "fasttext"
if STACK_WITH_ORIGIN:
    ########################################
    ## index word vectors
    ########################################
    print('Indexing word vectors')
    # Glove Vectors


    if loadvector == "word2vec":
        loadWord2Vector()
        EMBEDDING_DIM = 128
    elif loadvector == "glove":
        loadGloveVector()
    elif loadvector == "fasttext":
        loadFastTextVector()

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

cv1_x = train_df[cv1_columns].values
cv2_x = train_df[cv2_columns].values

t1_x = test_df[t1_columns].values
t2_x = test_df[t2_columns].values

list_sentences_test = test_df["comment_text"].fillna("NA").values

if STACK_WITH_ORIGIN:
    comments = []
    for text in list_sentences_train:
        comments.append(text_to_wordlist(text))

    test_comments = []
    for text in list_sentences_test:
        test_comments.append(text_to_wordlist(text))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)  # fenci
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # all train data
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)  # all test data
    print('Shape of test_data tensor:', test_data.shape)

    ########################################
    ## prepare embeddings
    ########################################
    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

def buildNNwithori():
    comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    cv1_input = Input(shape=(6,),dtype='float64')
    cv2_input = Input(shape=(6,),dtype='float64')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAvgPool1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    merged = concatenate([avg_pool, max_pool,cv1_input,cv2_input])
    merged = Dropout(0.1)(merged)
    preds = Dense(6, activation='sigmoid')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[comment_input,cv1_input,cv2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])
    # print(model.summary())
    return model

def buildNNwithoutori():
    cv1_input = Input(shape=(6,), dtype='float32')
    cv2_input = Input(shape=(6,), dtype='float32')
    merged = concatenate([cv1_input,cv2_input])
    merged = Dense(250,activation='relu')(merged)
    merged = Dropout(0.1)(merged)
    preds = Dense(6, activation='sigmoid')(merged)
    model = Model(inputs=[ cv1_input, cv2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])
    # print(model.summary())
    return model

# if STACK_METHOD == 'NN':
#     print("NNmodel stack")
#     model =
# else:
#     print('TODO')

cv_models=[]
cv_results=[]
cv_scores=[]
Kfold = 10
for i in range(0,Kfold):
    idx_train = train_df[train_df['cv_id'] != i].index
    idx_val = train_df[train_df['cv_id'] == i].index
    valid_id = train_df[train_df['cv_id'] == i]['id'].values
    if STACK_WITH_ORIGIN:
        data_train = data[idx_train]
        data_val = data[idx_val]
        print("train_shape")
        print(data_train.shape)
        print("val_shape")
        print(data_val.shape)


    cv1_train = cv1_x[idx_train]
    cv2_train = cv2_x[idx_train]
    labels_train = y[idx_train]
    cv1_val = cv1_x[idx_val]
    cv2_val = cv2_x[idx_val]
    labels_val = y[idx_val]
    print("fold %d"%(i))

    print("cv_train_shape")
    print(cv1_train.shape,cv2_train.shape)

    print("cv_val_shape")
    print(cv1_val.shape, cv2_val.shape)

    if STACK_METHOD == 'NN':
        print("NNmodel stack")
        if STACK_WITH_ORIGIN:
            model = buildNNwithori()
        else:
            model = buildNNwithoutori()
    else:
        print('TODO')
        break
    best = [-1, 0, 0, 0]  # socre, epoch, model.copy , cv_result
    earlystop = 5
    if STACK_METHOD == 'NN':
        print("NNmodel stack")
        if STACK_WITH_ORIGIN:
            train_ips=[data_train,cv1_train,cv2_train]
            val_ips=[data_val,cv1_val,cv2_val]
        else:
            train_ips = [cv1_train, cv2_train]
            val_ips=[cv1_val,cv2_val]
    else:
        print('TODO')
        break
    for epoch in range(1000):
        model.fit(train_ips,labels_train,batch_size=512, epochs=1, verbose=0)
        r = model.predict(val_ips ,batch_size=512)
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

if STACK_METHOD == 'NN':
    print("NNmodel stack")
    if STACK_WITH_ORIGIN:
        test_ips = [test_data, t1_x, t2_x]
    else:
        test_ips = [t1_x, t2_x]
else:
    print('TODO')

for i in range(Kfold):
    print("prediction "+ str(i))
    if len(r) == 0:
        r = cv_models[i].predict(test_ips,batch_size=1024)
    else:
        r += cv_models[i].predict(test_ips,batch_size=1024)
r /= Kfold
index = 'stack-mlp-lgb-'+str(STACK_METHOD)+'swo'+str(STACK_WITH_ORIGIN)+'-10-fold-0'

print("write files...")
pd.concat(cv_results).to_csv("../cv_result/%.4fnn_cv_"% (avg_val_score)+str(index)+".csv",index=False)

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = r

sample_submission.to_csv("../result/%.4f_nn_"% (avg_val_score)+ index+".csv",index=False)