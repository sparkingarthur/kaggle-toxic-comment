import pandas as pd
sub = pd.read_csv('../result/mylgb-10-fold-0.csv')
for column in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
    sub[column] /= 10.
label_colmn=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
sub = sub.reindex(columns=label_colmn)
sub.to_csv('mylgb-10-fold-0-985.csv',index=False)