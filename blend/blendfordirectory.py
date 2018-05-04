import pandas as pd
import numpy as np
import os

WEIGHTED=False

submissions_path = "../newnn/fusion"
# submissions_path = "../allresults"
all_files = os.listdir(submissions_path)
print(all_files)
outs = [pd.read_csv(os.path.join(submissions_path, f)) for f in all_files]
print(len(outs))
index = 'final'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

from sklearn.preprocessing import minmax_scale
for sub in outs:
    sub[label_cols] = minmax_scale(sub[label_cols])

blend = outs[0].copy()
blend[label_cols] = blend[label_cols]*0.0
weights=[5,6]
if WEIGHTED==True:
    avgrate = 1 / np.sum(weights)
else:
    avgrate = 1 / len(outs)
# weights=[1,1,1,1,1,0.8,1.2]
print(len(weights))
if WEIGHTED==True:
    for sub,weight in zip(outs,weights):
        blend[label_cols] += sub[label_cols] * avgrate * weight
else:
    for sub in outs:
        blend[label_cols] += sub[label_cols] * avgrate
blend.to_csv("../blendtemp/new-"+str(index)+".csv",index=False)


