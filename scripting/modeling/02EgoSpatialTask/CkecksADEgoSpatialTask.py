# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 20:05:44 2025

@author: aramendi
"""

import pandas as pd
import seaborn as sns
df= pd.read_excel(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\ProlificOrder2\conditionFileEgoSpatial.xlsx")

df['position'] = (
    df['n_objeto']
      .str.extract(r'^([LR])', expand=False)          # grab leading L or R
      .map({'L': 'left', 'R': 'right'})               # map to words
)

from scipy import stats
i= 'AD'
acc_self     = df[i].loc[df['Type_trialEncoded'] == 'Self']
acc_landmark = df[i].loc[df['Type_trialEncoded'] == 'Landmark']

# paired t‑test
t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)
print(p_value_scipy)

sns.barplot(x='Type_trialEncoded', y='AD', data= df)

from scipy import stats
i= 'AD'
acc_self     = df[i].loc[df['SelfvsLandmark_proximity'] == 'correct_self']
acc_landmark = df[i].loc[df['SelfvsLandmark_proximity'] == 'correct_landmark']

# paired t‑test
t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)

print(p_value_scipy)


from scipy import stats
i= 'AD'
acc_self     = df[i].loc[df['Self_proximity'] == 'self_close']
acc_landmark = df[i].loc[df['Self_proximity'] == 'self_far']

# paired t‑test
t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)

print(p_value_scipy)



from scipy import stats
i= 'AD'

acc_self     = df[i].loc[df['Self_proximity'] == 'self_close']
acc_landmark = df[i].loc[df['Self_proximity'] == 'self_far']

# paired t‑test
t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)

print(p_value_scipy)