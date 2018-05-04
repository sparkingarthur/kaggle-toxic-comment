import pandas as pd
import numpy as np

sub1 = pd.read_csv('../newnn/fusion/new-mynnwithkernel.csv')
# #sub1=pd.read_csv('../kernel/cnn-rnn-lb0.052.csv')
sub2 = pd.read_csv('../newnn/fusion/new-mlp-lgb-fm-capsulenet-charwordrnn2.csv')
sub3 = pd.read_csv('../newnn/fusion/0.9910LGBstack-allnnlb9866.csv')
sub4 = pd.read_csv('../newnn/fusion/blend_it_alllb9867.csv')
#
# sub1 = pd.read_csv('../newnn/fusion/new-fusion-overfit-all.csv')
# # #sub1=pd.read_csv('../kernel/cnn-rnn-lb0.052.csv')
# sub2 = pd.read_csv('../newnn/fusion/new-mlp-lgb-fm-capsulenet-charwordrnn2.csv')
# sub3 = pd.read_csv('../newnn/fusion/preprocessed_blendlb9860.csv')
# sub4 = pd.read_csv('../newnn/fusion/new-mynnwithkernel.csv')
# sub5 = pd.read_csv('../newnn/fusion/0.9910LGBstack-allnnlb9866.csv')

from sklearn.preprocessing import minmax_scale
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    sub1[label] = minmax_scale(sub1[label])
    sub2[label] = minmax_scale(sub2[label])
    sub3[label] = minmax_scale(sub3[label])
    sub4[label] = minmax_scale(sub4[label])
    #sub5[label] = minmax_scale(sub5[label])

#sub3=pd.read_csv('../directoryfussion/temp/0.0427_multicnnglove_vectors_512.00_0.10.csv')
index = 'blend-sp9.94'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
blend = sub1.copy()
blend[label_cols] = sub1[label_cols]*0.25 + sub2[label_cols]*0.2 + sub3[label_cols]*0.15 + sub4[label_cols]*0.2 #+ sub5[label_cols]*0.20
#blend[label_cols] = (sub1[label_cols]*7 + sub2[label_cols]*5 + sub3[label_cols]*6) / 18
blend.to_csv("../blendtemp/sub_f_"+str(index)+".csv",index=False)