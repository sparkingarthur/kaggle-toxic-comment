import numpy as np
import pandas as  pd
import copy # perform deep copyong rather than referencing in python
from tqdm import tqdm # progress bar

import nltk # general NLP
import re # regular expressions
import gensim.models.word2vec as w2v # word2vec model

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


path = '../input/'
TRAIN_DATA_FILE = path + 'train.csv'
TEST_DATA_FILE = path + 'test.csv'
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)

# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Replace Numbers
    text = replace_numbers.sub('n', text)

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
allwordslist=[]
for text in list_sentences_train:
    words = text_to_wordlist(text)
    comments.append(words)
    allwordslist.append(words.split())

test_comments = []
for text in list_sentences_test:
    words = text_to_wordlist(text)
    test_comments.append(words)
    allwordslist.append(words.split())

def load_extend_vacabulary():
    f = open('../w2v/badwords.csv')
    for line in f:
        words = text_to_wordlist(line)
        allwordslist.append(words.split())
    f.close()
    f = open('../w2v/Terms-to-Block.csv')
    for line in f:
        words = text_to_wordlist(line)
        allwordslist.append(words.split())
    f.close()
    print("extended words loaded.")
load_extend_vacabulary()
#allwordslist = " ".join(allwordslist)
#print(allwordslist)
#print(allwordslist.find('fuck'))

# hyper parameters of the word2vec model
num_features = 128 # dimensions of each word embedding
min_word_count = 1 # this is not advisable but since we need to extract
# feature vector for each word we need to do this
#num_workers = multiprocessing.cpu_count() # number of threads running in parallel
context_size = 7 # context window length
downsampling = 1e-3 # downsampling for very frequent words
seed = 1 # seed for random number generator to make results reproducible

word2vec_ = thrones2vec = w2v.Word2Vec(
    sg = 1, seed = seed,
    workers = 6,
    size = num_features,
    min_count = min_word_count,
    sample = downsampling
)
# first we need to built the vocab
print("training for word2vec")
word2vec_.build_vocab(allwordslist)
# now we need to train the model
word2vec_.train(allwordslist, total_examples = word2vec_.corpus_count, epochs = word2vec_.iter)
word2vec_.save('word2vec_128_e.model')
print(word2vec_['fuck'])