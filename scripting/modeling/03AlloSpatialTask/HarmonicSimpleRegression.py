# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:38:32 2025

@author: aramendi
"""


##############################################################################################################################################
## T a harmonic regression model 
import numpy as np, pandas as pd
import statsmodels.api as sm
df4MT = pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\allotask.csv")
df4mtMeans= df4MT.groupby(['PROLIFIC_PID','angularDisparity'])['key_resp_3.corr'].mean().reset_index()
# 1) Mapea 1..8 a grados "normales"
deg_map = {1:40,2:80,3:120,4:160,5:200,6:240,7:280,8:320}

df4mtMeans = df4mtMeans.copy()
df4mtMeans['deg_mapped'] = df4mtMeans['angularDisparity'].map(deg_map).astype(float)

# 3) Ahora sí, a radianes sin problema
df4mtMeans['θ'] = np.deg2rad(df4mtMeans['deg_mapped'])
df4mtMeans['Accuracy'] = df4mtMeans['key_resp_3.corr'] * 100
results = []
for pid, grp in df4mtMeans.groupby('PROLIFIC_PID'):
    X = pd.DataFrame({
        'cosθ': np.cos(grp['θ']),
        'sinθ': np.sin(grp['θ'])
    })
    X = sm.add_constant(X)
    y = grp['Accuracy']
    fit = sm.OLS(y, X).fit()
    β0, βc, βs = fit.params
    amp   = np.hypot(βc, βs)
    phase = np.arctan2(βs, βc)
    results.append({
        'PROLIFIC_PID': pid,
        'intercept':   β0,
        'beta_cos':    βc,
        'beta_sin':    βs,
        'amplitude':   amp,
        'phase_rad':   phase,
        'R2':          fit.rsquared
    })

df_harm = pd.DataFrame(results)
print(df_harm)
##############################################################################################################################################


import numpy as np, pandas as pd
import statsmodels.api as sm

# 1) Mapea 1..8 a grados "normales"
deg_map = {1:40,2:80,3:120,4:160,5:200,6:240,7:280,8:320}

anova = anova.copy()
anova['deg_mapped'] = anova['angularDisparity'].map(deg_map).astype(float)

# 3) Ahora sí, a radianes sin problema
anova['θ'] = np.deg2rad(anova['deg_mapped'])
anova['Accuracy'] = anova['key_resp_3.corr'] * 100
results = []
for pid, grp in anova.groupby('PROLIFIC_PID'):
    X = pd.DataFrame({
        'cosθ': np.cos(grp['θ']),
        'sinθ': np.sin(grp['θ'])
    })
    X = sm.add_constant(X)
    y = grp['Accuracy']
    fit = sm.OLS(y, X).fit()
    β0, βc, βs = fit.params
    amp   = np.hypot(βc, βs)
    phase = np.arctan2(βs, βc)
    results.append({
        'participant': pid,
        'intercept':   β0,
        'beta_cos':    βc,
        'beta_sin':    βs,
        'amplitude':   amp,
        'phase_rad':   phase,
        'R2':          fit.rsquared
    })

df_harm = pd.DataFrame(results)
print(df_harm)
