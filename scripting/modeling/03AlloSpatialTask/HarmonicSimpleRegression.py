# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:38:32 2025

@author: aramendi
"""


##############################################################################################################################################
## T a harmonic regression model 
import numpy as np, pandas as pd
import statsmodels.api as sm
path_data= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"

df4MT= pd.read_csv(path_data + "\\processed\\AlloTask_SpatialScore.csv")
df4mtMeans= df4MT.groupby(['PROLIFIC_PID','angularDisparity'])['key_resp_3.corr'].mean().reset_index()
# 1) Mapea 1..8 a grados "normales"
deg_map = {1:40,2:80,3:120,4:160,5:200,6:240,7:280,8:320}

df4mtMeans = df4mtMeans.copy()
df4mtMeans['deg_mapped'] = df4mtMeans['angularDisparity'].map(deg_map).astype(float)

# 3) Ahora sí, a radianes sin problema
df4mtMeans['θ'] = np.deg2rad(df4mtMeans['deg_mapped'])
df4mtMeans['Accuracy'] = df4mtMeans['key_resp_3.corr'] 
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



import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Permutation test to see if amplitude is significant above chance
# ---------------------------------------------------------------------


def compute_amp(grp):
    """Compute amplitude for one participant."""
    X = pd.DataFrame({
        'cosθ': np.cos(grp['θ']),
        'sinθ': np.sin(grp['θ'])
    })
    X = sm.add_constant(X)
    y = grp['Accuracy']
    fit = sm.OLS(y, X).fit()
    βc = fit.params['cosθ']
    βs = fit.params['sinθ']
    amplitude = np.sqrt(βc**2 + βs**2)
    return amplitude

# ---------------------------------------------------------
# 1) Compute REAL group mean amplitude
# ---------------------------------------------------------
amps_real = []
for pid, grp in df4mtMeans.groupby('PROLIFIC_PID'):
    amps_real.append(compute_amp(grp))
real_group_mean = np.mean(amps_real)

# ---------------------------------------------------------
# 2) Permutation test
# ---------------------------------------------------------
n_permutations = 200   # increase to 1000 for real analysis
null_distribution = []

for perm in range(n_permutations):
    perm_amps = []
    for pid, grp in df4mtMeans.groupby('PROLIFIC_PID'):
        shuffled = grp.copy()
        shuffled['θ'] = np.random.permutation(shuffled['θ'])  # shuffle angles
        amp = compute_amp(shuffled)
        perm_amps.append(amp)
    null_distribution.append(np.mean(perm_amps))

null_distribution = np.array(null_distribution)

# ---------------------------------------------------------
# 3) Compute p-value
# ---------------------------------------------------------
p_value = (np.sum(null_distribution >= real_group_mean) + 1) / (n_permutations + 1)

real_group_mean, p_value




#df_harm.to_csv(paths_results + "\\03AlloSpatialTask\\HarmonicRegressionRT.csv")
##############################################################################################################################################

