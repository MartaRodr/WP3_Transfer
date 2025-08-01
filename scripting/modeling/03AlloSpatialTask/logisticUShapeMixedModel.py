# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:32:59 2025

@author: aramendi
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf 


dfegoSpatial= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\egospatialtask.csv")
bad_participants = (
    dfegoSpatial.groupby('PROLIFIC_PID')['Accuracy']
         .mean()
         .loc[lambda s: s < 0.60]
         .index
    .tolist()
)


df4MT = pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\allotask.csv")
df4MT['Accuracy']= df4MT['Ncorrect'] / 24
df4MT= df4MT.rename(columns= {'key_resp_3.corr':'acc'})

mean_per_pid = df4MT.groupby('PROLIFIC_PID')['angularDisparity'].transform('mean')

# 2) Centra respecto a la media de cada participante
df4MT['angDisparity_cwc'] = df4MT['angularDisparity'] - mean_per_pid

# 3) Genera el término cuadrático
df4MT['angDis2_cwc'] = df4MT['angDisparity_cwc'] ** 2


import statsmodels.formula.api as smf
coefs = []
for pid, grp in df4MT.groupby('PROLIFIC_PID'):
    m = smf.glm("acc ~ angDisparity_cwc * angDis2_cwc * Accuracy",
                data=grp, family=sm.families.Binomial()).fit()
    coefs.append(m.params['angDis2_cwc'])
    
    
t_statistic, t_p_value = stats.ttest_1samp(
    coefs, 
    popmean=0,
    alternative='greater'
)
print(f'Cuadrático: t = {t_statistic:.3f}, p = {t_p_value:.3f}')