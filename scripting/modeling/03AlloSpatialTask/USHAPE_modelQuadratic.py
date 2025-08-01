# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 15:11:40 2025

@author: aramendi
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf 
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Open df
dataset_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\processed"
df4MT = pd.read_csv(dataset_path + "//AlloTask_SpatialScore.csv")


# QUadratic regression 
results = []
predictors= ['angDisparity_cwc','angDis2_cwc']

for participant in df4MT['PROLIFIC_PID'].unique():
    data_subset = df4MT[df4MT['PROLIFIC_PID'] == participant]
    data_subset = data_subset.dropna(subset=['key_resp_3.corr', 'key_resp_3.rt', 'angularDisparity']) # Eliminate Nans
    
    data_subset['logRT'] = np.log(data_subset['key_resp_3.rt'])
    
    data_subset['angDisparity_cwc'] = data_subset['angularDisparity'] - data_subset['angularDisparity'].mean()
    data_subset['angDis2_cwc']      = data_subset['angDisparity_cwc'] ** 2

    X = data_subset[predictors]
    y = data_subset['logRT']
    yAcc= data_subset['key_resp_3.corr']
    
    # Reaction time regression
    modelRT= LinearRegression().fit(X,y)
    coefs_RT= modelRT.coef_
    
    modelAcc= LinearRegression().fit(X,yAcc)
    coefs_Acc= modelAcc.coef_
    
    results.append({'PROLIFIC_PID': participant, 
                    'coeficients_quadraticRT':coefs_RT[1],'coeficients_linearRT':coefs_RT[0],
                    'coeficients_quadraticAcc': coefs_Acc[1], 'coeficients_linearAcc': coefs_Acc[0]})

    
    
quadratic4MT= pd.DataFrame(results)

output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results\03AlloSpatialTask"
quadratic4MT.to_csv(output_path + "\\allocentric_slopesQuatraticRegressions.csv")

from scipy.stats import ttest_1samp, wilcoxon


##########################################################################################################
#◘ Stats and graphs
tests = [
    ("coeficients_quadraticRT", "less", "Quadratic RTs", "Slopes RTs"),
    ("coeficients_quadraticAcc", "greater", "Quadratic Acc", "Slopes Acc")
]

print("=== T-tests and Wilcoxon signed-rank tests ===")

plt.figure(figsize=(8, 5))
y_lim_min=-0.1
y_lim_max=0.11
y_ticks = np.arange(y_lim_min, y_lim_max, 0.05)
for i, (col, alt, label, ylabel) in enumerate(tests):
    print(f"\n--- {label} ---")

    # T-test
    t_stat, t_p = ttest_1samp(quadratic4MT[col], popmean=0, alternative=alt)
    print(f"T-test:       t = {t_stat:.3f}, p = {t_p:.3f}")

    # Wilcoxon test
    try:
        w_stat, w_p = wilcoxon(quadratic4MT[col], alternative=alt)
        print(f"Wilcoxon test: W = {w_stat:.3f}, p = {w_p:.3f}")
    except ValueError as e:
        w_p = None
        print(f"Wilcoxon test could not be computed: {e}")

    # Subplot
    ax = plt.subplot(1, 2, i + 1)
    sns.set_theme(style="white")
    sns.barplot(y=quadratic4MT[col], color='lightgrey', ci=None, ax=ax)
    sns.stripplot(y=quadratic4MT[col], color='black', dodge=True, alpha=0.8, size=3, ax=ax)
    ax.axhline(0.00, linestyle='--', color='red', linewidth=1)
    ax.set_xlabel('')
    ax.set_ylim(y_lim_min,y_lim_max)
    ax.set_yticks(y_ticks)
    ax.set_ylabel(ylabel, color='black', size=14, fontweight='bold')

    # P-value text in upper right
    if w_p is not None:
        p_text = f"p = {w_p:.3f}" if w_p >= 0.001 else "p < .001"
        ax.text(0.95, 0.95, p_text,
                ha='right', va='top', fontsize=12, fontweight='bold', transform=ax.transAxes)

plt.subplots_adjust(wspace=0.4)
plt.show()

##########################################################################################################
### CORRELATIONS 4MT WITH PERFORMANCE ##
df4MT_mean= df4MT.groupby(['PROLIFIC_PID']).mean().reset_index()
union= pd.merge(df4MT_mean, quadratic4MT, on='PROLIFIC_PID')
from scipy import stats

variables=['coeficients_quadraticRT','coeficients_quadraticAcc']
for variable in variables:
    res = stats.spearmanr(union['key_resp_3.corr'], union[variable])
    print("Correlation with accuracy 4mt and ushape")
    print(res)

##########################################################################################################
pair_map = {
    1: 1, 8:1,
    2: 1, 7: 1,
    3: 2, 6:2,
    4: 2, 5:2,
}

# create the new bin column
df4MT['bin'] = df4MT['angularDisparity'].map(pair_map)
df4MT_mean= df4MT.groupby('PROLIFIC_PID').mean().reset_index()

bin1= df4MT.loc[df4MT['bin']==1]
bin1mean= bin1.groupby('PROLIFIC_PID').mean().reset_index()

bin2= df4MT.loc[df4MT['bin']==2]
bin2mean= bin2.groupby('PROLIFIC_PID').mean().reset_index()

union= pd.merge(bin1mean,bin2mean, on='PROLIFIC_PID')
union['differences']= union['key_resp_3.rt_x'] - union['key_resp_3.rt_y']

plt.figure(1)
#sns.lineplot(x='bin', y='key_resp_3.rt', data= df4MT, hue='PROLIFIC_PID',legend=False,linewidth=0.2, ci=False)
sns.barplot(y=union['differences'], data=union,color='lightgrey')   
sns.stripplot(y=union['differences'], data=union,color='black', dodge=True,alpha=0.8, size=3)

plt.show()

plt.axhline(0, linestyle='--', color='red', linewidth=1)

union_posti= union.loc[union['differences']>=0]
##########################################################################################################

