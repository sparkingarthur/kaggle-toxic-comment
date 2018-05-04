import pandas as pd
import numpy as np
import random
train = pd.read_csv('../input/train_pre.csv')
train.fillna('NA',inplace=True)
comment = train['comment_text']

SAMPLE_RATE = 0.15
SHUFFLE_RATE = 0.3
def sample_enhance(x,sample_rate):
    """
    sample index for enhance, 1/5 data every time
    """
    shuffle_idx = []
    for i in range(0,len(x)):
        shuffle_idx.append(random.randint(0,len(x)-1))
    return list(set(shuffle_idx[:int(len(shuffle_idx)*sample_rate)]))


def shuffle_comment(comment,shuffle_rate):
    try:
        word_list = comment.split(' ')
        length = len(word_list)
        shuffle_len = int(length*shuffle_rate)
        start = random.randint(0,length-shuffle_len-1)
        a = word_list[start:start+shuffle_len]
        random.shuffle(a)
        word_list[start:start + shuffle_len] = a
        comment = " ".join(word_list)
    except:
        print(comment)
        return comment
    return comment

idx = sample_enhance(train,SAMPLE_RATE)
# idx = [0,1,2,3,4,5]
sample_train = train.iloc[idx]
sample_train['comment_text'] = sample_train['comment_text'].map(lambda x:shuffle_comment(x,SHUFFLE_RATE))
train.iloc[idx] = sample_train
# sample_train = train.iloc[idx]
train.to_csv('train_pre_enhance.csv',index=None)
# print(sample_train)

