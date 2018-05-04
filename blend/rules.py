import pandas as pd
import numpy as np

sub = pd.read_csv('../blendtemp/sub_f_blend-sp9.91.csv')

def simplejudge(x, y):
    if y > x:
        return y
    else:
        return x

def thresholdjudge(x,y,thes=0.5):
    if x < thes < y:
        return y
    else:
        return x

sub['toxic'] = list(map(lambda x, y: thresholdjudge(x, y) , sub['toxic'], sub['severe_toxic']))

index = 'blend-sp9.911'
sub.to_csv("../blendtemp/sub_f_"+str(index)+".csv",index=False)