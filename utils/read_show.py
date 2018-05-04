import pandas as pd
import numpy as np

# extend1 = pd.read_csv('../w2v/badwords.csv')
# f = open('../w2v/badwords.csv')
# for line in f:
#     values = line.split()
#     word = ' '.join(values)
# f.close()

train = pd.read_csv('../input/train.csv')
print(train.head(10))

test = pd.read_csv('../input/test.csv')
print(test.head(10))

