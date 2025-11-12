# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:51:00 2025

@author: aramendi
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import statsmodels.api as sm

#########################################################################################################################
df4MT = pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\allotask.csv")
df4MT = pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP5_SocialSpatialTask\Analysis\Datasets\AlloSpatialTask_WP5.csv")

paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"

#########################################################################################################################
participant_ID= 'DNI' # or "DNI"
##################################
# harmonic regresions
df4mtMeans= df4MT.groupby([participant_ID,'angularDisparity']).mean().reset_index()
# 1) Mapea 1..8 a grados "normales"
deg_map = {1:40,2:80,3:120,4:160,5:200,6:240,7:280,8:320}

df4mtMeans = df4mtMeans.copy()
df4mtMeans['deg_mapped'] = df4mtMeans['angularDisparity'].map(deg_map).astype(float)

# 3) Ahora sí, a radianes sin problema
df4mtMeans['θ'] = np.deg2rad(df4mtMeans['deg_mapped'])
df4mtMeans['Accuracy'] = df4mtMeans['key_resp_3.corr'] 
df4mtMeans['RT'] = df4mtMeans['key_resp_3.rt'] 
results = []
resultsRT= []

for pid, grp in df4mtMeans.groupby(participant_ID):
    X = pd.DataFrame({
        'cosθ': np.cos(grp['θ']),
        'sinθ': np.sin(grp['θ'])
    })
    # MODEL Accuracy
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
    
    # MODEL RT
    X = sm.add_constant(X)
    y = grp['RT']
    fit = sm.OLS(y, X).fit()
    β0, βc, βs = fit.params
    amp   = np.hypot(βc, βs)
    phase = np.arctan2(βs, βc)
    resultsRT.append({
        'PROLIFIC_PID': pid,
        'intercept':   β0,
        'beta_cos':    βc,
        'beta_sin':    βs,
        'amplitude':   amp,
        'phase_rad':   phase,
        'R2':          fit.rsquared
    })
    

df_harm = pd.DataFrame(results)
df_harmRT = pd.DataFrame(resultsRT)

## Save results
if participant_ID=='DNI':
    df_harm.to_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP5_SocialSpatialTask\Analysis\Results\4MT\Coeffs_HarmonicRegressionAcc.csv")
    df_harmRT.to_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP5_SocialSpatialTask\Analysis\Results\4MT\Coeffs_HarmonicRegressionRT.csv")

elif participant_ID=='PROLIFIC_PID':
    df_harm.to_csv(paths_results + "\\03AlloSpatialTask\\HarmonicRegression.csv")
    df_harmRT.to_csv(paths_results + "\\03AlloSpatialTask\\HarmonicRegressionRT.csv")

'''
## PLOTING RESULTS ##
# Angles
deg_map = {1:40, 2:80, 3:120, 4:160, 5:200, 6:240, 7:280, 8:320}
order360 = [40, 80, 120, 160, 200, 240, 280, 320]

df = anova.copy()
df['deg360'] = df['angularDisparity'].map(deg_map)

# Variables
acc_col = 'key_resp_3.corr'
rt_col  = 'key_resp_3.rt'

# Mean by angle
acc_means = df.groupby('deg360')[acc_col].mean().reindex(order360).values
# Si accuracy está en [0,1], pásalo a %
if np.nanmax(acc_means) <= 1.0:
    acc_means = acc_means * 100.0

rt_means  = df.groupby('deg360')[rt_col].mean().reindex(order360).values

#
theta = np.deg2rad(order360)
bar_width = np.deg2rad(45) * 0.8
labels = ['40°','80°','120°','160°','-160°','-120°','-80°','-40°']

# FIGURE #
fig, axes = plt.subplots(1, 2, figsize=(12,6), subplot_kw=dict(polar=True), constrained_layout=True)

# ===================== Accuracy =====================
ax = axes[0]
cmap_acc = plt.cm.YlGnBu
norm_acc = plt.Normalize(vmin=np.nanmin(acc_means), vmax=np.nanmax(acc_means))
colors_acc = cmap_acc(norm_acc(acc_means))

bars = ax.bar(theta, acc_means, width=bar_width, color=colors_acc,
              edgecolor='black', linewidth=1, alpha=0.9)
ax.plot(theta, acc_means, 'k--o', lw=1.2, markersize=4)

# Línea de referencia al 25%
circ = np.linspace(0, 2*np.pi, 200)
ax.plot(circ, np.full_like(circ, 25), linestyle=':', color='red', linewidth=1)

ax.set_title('Accuracy', pad=12)
ax.set_theta_zero_location('S')
ax.set_theta_direction(1)
ax.set_xticks(theta)
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels([])

# Colorbar para Accuracy
sm_acc = plt.cm.ScalarMappable(cmap=cmap_acc, norm=norm_acc)
sm_acc.set_array([])
cbar_acc = plt.colorbar(sm_acc, ax=ax, pad=0.12, shrink=0.8)
cbar_acc.set_label('Performance (%Correct)')

# ===================== RT =====================
ax = axes[1]
cmap_rt = plt.cm.YlGnBu
norm_rt = plt.Normalize(vmin=np.nanmin(rt_means), vmax=np.nanmax(rt_means))
colors_rt = cmap_rt(norm_rt(rt_means))

ax.bar(theta, rt_means, width=bar_width, color=colors_rt,
       edgecolor='black', linewidth=1, alpha=0.9)
ax.plot(theta, rt_means, 'k--o', lw=1.2, markersize=4)

circ = np.linspace(0, 2*np.pi, 200)
ax.plot(circ, np.full_like(circ, rt_means.mean()), linestyle=':', color='red', linewidth=1)

ax.set_title('RT(s)', pad=12)
ax.set_theta_zero_location('S')
ax.set_theta_direction(1)
ax.set_xticks(theta)
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels([])

# Colorbar para RT
sm_rt = plt.cm.ScalarMappable(cmap=cmap_rt, norm=norm_rt)
sm_rt.set_array([])
cbar_rt = plt.colorbar(sm_rt, ax=ax, pad=0.12, shrink=0.8)
cbar_rt.set_label('RT (s)')

plt.show()

'''
