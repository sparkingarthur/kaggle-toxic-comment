import pandas as pd
cv_3 = pd.read_csv('./lvl0_wordbatch_clean_oof.csv')
t_3 = pd.read_csv('./lvl0_wordbatch_clean_sub.csv')


labels=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
cv_3.drop(labels,axis=1,inplace=True)
print(cv_3.head())


