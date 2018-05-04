#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:15:50 2018

@author: hzs
"""

import pandas as pd
import re,string
from textacy.preprocess import preprocess_text


train = pd.read_csv('./train_pre.csv')
test = pd.read_csv('./test_pre.csv')


#https://www.kaggle.com/prashantkikani/toxic-logistic-preprocessing    

repl = {
    "&lt;3": " good ",
#    ":d": " good ",
#    ":dd": " good ",
#    ":p": " good ",
#    "8)": " good ",
#    ":-)": " good ",
#    ":)": " good ",
#    ";)": " good ",
#    "(-:": " good ",
#    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
#    ":/": " bad ",
#    ":&gt;": " sad ",
    ":')": " sad ",
#    ":-(": " bad ",
#    ":(": " bad ",
#    ":s": " bad ",
#    ":-s": " bad ",
#    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["comment_text"] = new_train_data
test["comment_text"] = new_test_data

    
    






nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 
nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero
nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 
                            # that people doon't generally use all number format for old dates, but write them out 
nDELIM = r'(?:[\/\-\._])?'  # 
NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""
DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
    |
    # or just a number, but without a leading zero
    (?:[123]?\d)
)"""
MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""
DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

YEAR_4D = r"""(?:[12]\d\d\d)"""
DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon
:[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""

COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

myDate = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)


def text_to_wordlist(text):

# text = text.lower().split()
# text = " ".join(text)

#https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("https?:* ", "<URL>", text)
    text = re.sub("www.* ", "<URL>", text)
    text = re.sub("\[\[User(.*)\|", '<USER>', text)
    text = re.sub("<3", '<HEART>', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
    text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
    text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
    text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
    text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
    text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
    text = re.sub("/", " / ", text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub("([!]){2,}", "! <REPEAT>", text)
    text = re.sub("([?]){2,}", "? <REPEAT>", text)
    text = re.sub("([.]){2,}", ". <REPEAT>", text)
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <ELONG>", text)

#https://www.kaggle.com/edrushton/removing-dates-data-cleaning  
    #date  
    text = re.sub('myDate','_date_',text)


        
    # Replace ips
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ',text)
    #remove http links in the text    
    text = re.sub("(http://.*?\s)|(http://.*)",'',text)
    
#==============================================================================
#     # remove any text starting with User...     
#     text = re.sub("\[\[User.*",'',text)
#==============================================================================

    # Replace \\n
    text = re.sub('\\n',' ',text)

    
    
    
    
#==============================================================================
#     # Isolate punctuation
#     text = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', text)
#==============================================================================

    # Remove some special characters
    text = re.sub(r'([\;\:\|•«\n])', ' ', text)
    
    text = re.sub('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation), r' \1 ', text)
    
    
    
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('0', ' zero ')
    text = text.replace('1', ' one ')
    text = text.replace('2', ' two ')
    text = text.replace('3', ' three ')
    text = text.replace('4', ' four ')
    text = text.replace('5', ' five ')
    text = text.replace('6', ' textix ')
    text = text.replace('7', ' texteven ')
    text = text.replace('8', ' eight ')
    text = text.replace('9', ' nine ')
    


#==============================================================================
#     # Split the sentences into words
#     s = tweet_tokenizer.tokenize(s)
# 
#     # Lemmatize
#     s = [lem.lemmatize(word, "v") for word in s]
# 
#     # Remove Stopwords
#     s = ' '.join([w for w in s if not w in eng_stopwords])
#==============================================================================

#https://www.kaggle.com/sanghan/attention-with-fasttext-embeddings/notebook
    
    text = preprocess_text(text, fix_unicode=True,
                           lowercase=False,
                           no_currency_symbols=True,
                           transliterate=True,
                           no_urls=True,
                           no_emails=True,
                           no_contractions=False,
                           no_phone_numbers=True,
                           no_punct=False).strip()    



    return(text)

train['comment_text'] = train['comment_text'].map(lambda x:text_to_wordlist(x))
test['comment_text'] = test['comment_text'].map(lambda x:text_to_wordlist(x))

train.to_csv('train_pre2.csv',index=None)
test.to_csv('test_pre2.csv',index=None)

#https://www.kaggle.com/gaussmake1994/word-character-n-grams-tfidf-regressions-lb-051
#==============================================================================
# stemmer = EnglishStemmer()
# 
# @lru_cache(30000)
# def stem_word(text):
#     return stemmer.stem(text)
# 
# 
# lemmatizer = WordNetLemmatizer()
# 
# @lru_cache(30000)
# def lemmatize_word(text):
#     return lemmatizer.lemmatize(text)
# 
# 
# def reduce_text(conversion, text):
#     return " ".join(map(conversion, wordpunct_tokenize(text.lower())))
# 
# 
# def reduce_texts(conversion, texts):
#     return [reduce_text(conversion, str(text))
#             for text in tqdm(texts)]
#     
# train['comment_text_stemmed'] = reduce_texts(stem_word, train['comment_text'])
# test['comment_text_stemmed'] = reduce_texts(stem_word, test['comment_text'])
# train['comment_text_lemmatized'] = reduce_texts(lemmatize_word, train['comment_text'])
# test['comment_text_lemmatized'] = reduce_texts(lemmatize_word, test['comment_text'])
#==============================================================================
#==============================================================================
# list_sentences_train = train["comment_text"].fillna("NA").values
# 
# list_sentences_test = test["comment_text"].fillna("NA").values
# 
# 
# comments = []
# for text in list_sentences_train:
#     comments.append(text_to_wordlist(text))
#     
# test_comments=[]
# for text in list_sentences_test:
#     test_comments.append(text_to_wordlist(text))
#==============================================================================
