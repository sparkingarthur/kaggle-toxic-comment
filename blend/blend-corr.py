import pandas as pd
import sys
from scipy.stats import ks_2samp
import numpy as np
# first_file = sys.argv[1]
# second_file = sys.argv[2]
first_file = 'D:/code/competition/kaggle-toxic/newnn/fusion/new-mynnwithkernel.csv'
second_file = 'D:/code/competition/kaggle-toxic/newnn/fusion/new-mlp-lgb-fm-stack.csv'
# first_file = 'D:/code/competition/kaggle-toxic/blendtemp/sub_f_blend-sp9.911.csv'
# second_file = 'D:/code/competition/kaggle-toxic/blendtemp/new-fusion-o-0.csv'

# first_file = 'D:/code/competition/kaggle-toxic/blendtemp/sub_f_blend-sp9.911.csv'
# second_file = 'D:/code/competition/kaggle-toxic/result/0.9783LGBstack-allnn-pseudo.csv'
from sklearn.preprocessing import minmax_scale
def corr(first_file, second_file):
    # assuming first column is `class_name_id`
    first_df = pd.read_csv(first_file)
    second_df = pd.read_csv(second_file)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    first_df[class_names] = minmax_scale(first_df[class_names])
    second_df[class_names] = minmax_scale(second_df[class_names])
    pearson = []
    for class_name in class_names:
        # all correlations
        print('\n Class: %s' % class_name)
        p=first_df[class_name].corr(
                  second_df[class_name], method='pearson')
        pearson.append(p)
        print(' Pearson\'s correlation score: %0.6f' %p              )
        print(' Kendall\'s correlation score: %0.6f' %
              first_df[class_name].corr(
                  second_df[class_name], method='kendall'))
        print(' Spearman\'s correlation score: %0.6f' %
              first_df[class_name].corr(
                  second_df[class_name], method='spearman'))
        ks_stat, p_value = ks_2samp(first_df[class_name].values,
                                    second_df[class_name].values)
        print(' Kolmogorov-Smirnov test:    KS-stat = %0.6f    p-value = %0.3e\n'
              % (ks_stat, p_value))
    print("pearson avg %.6f" % np.mean(pearson))
corr(first_file, second_file)
