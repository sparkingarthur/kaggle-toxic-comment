#coding=utf-8
import numpy as np
import pandas as pd
import random
train = pd.read_csv('../input/train.csv')
#print(len(train[train['cv_5']==4]))
lens = len(train)
print(lens)
# cv_id=pd.read_csv('../input/cv_id_10.txt')
#
# train['cv_10']=cv_id['cv_id_10']
# print(len(train[train['cv_10']==9]))
# index=train[train['cv_10']==9].index
# print(len(index))
#
# # cv_10 = np.random.randint(0,10,lens)
# # # print(cv_10.shape)
# # cv_5 = np.random.randint(0,5,lens)
# #
# #
# # cv_f_10 = pd.DataFrame()
# # cv_f_5 = pd.DataFrame()
# # cv_f_10['cv_id_10']=cv_10
# # cv_f_5['cv_id_5']=cv_5
# #
# # cv_f_10.to_csv('cv_id_10.csv',index=False)
# # cv_f_5.to_csv('cv_id_5.csv',index=False)
#
# list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# new = pd.DataFrame(columns=[['id']+list_classes])
# print(new)
# a = np.array([1,2,3,4,5,6])
# new['id']=[0]
# new[list_classes]=[1,2,3,4,5,6]
# print(new)

a=pd.DataFrame(np.array([[3,1],[2,5],[4,5]]))
a.columns=['A','B']
print(a)
index = a[a['A'] == 3].index
a.drop(index,axis=0,inplace=True)
print(a)