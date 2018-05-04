import pandas as pd
import numpy as np
import os

submissions_path = "../directoryfussion/fusion"
all_files = os.listdir(submissions_path)
print(all_files)
outs = [pd.read_csv(os.path.join(submissions_path, f)) for f in all_files]
outs_weights=[1,1,1,1,1,1,1]
index = 'f-more-geom'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
blend = outs[0].copy()
blend[label_cols] = blend[label_cols]*(1./blend[label_cols]) # ==1
#print(blend.head())
avgrate = 1 / len(outs)
for sub,weight in zip(outs,outs_weights):
    blend[label_cols] *= (sub[label_cols]**weight)
blend[label_cols] **= (1. / len(outs))
#blend[label_cols] **= 1
from scipy.special import expit, logit
def fu(x):
    return expit(logit(x)-0.5)
#blend[label_cols] = blend[label_cols].apply(fu)
#print(blend.head())
blend.to_csv("sub_f_"+str(index)+".csv",index=False)

