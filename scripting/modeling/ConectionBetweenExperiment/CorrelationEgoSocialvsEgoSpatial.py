# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:02:37 2025

@author: aramendi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import spearmanr, pearsonr



################################################################################################
# Open datasets
paths_cleanedData= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
egoSocial= pd.read_csv(paths_cleanedData + "\\egosocialCleanedRT.csv")
egoSpatial= pd.read_csv(paths_cleanedData + "\\egospatialCleanedRT.csv")

# Open reuslts from models EgoSpatial
path_modelsR= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results\02EgoSpatialTask"
slopesSpatialRT= pd.read_csv(path_modelsR + "\\CoefsMixedLinearRT_meanDistance_egoTask.csv")
slopesSpatialAcc= pd.read_csv(path_modelsR + "\\CoefsMixedLogisticAcc_meanDistance_egoTask.csv")

########################## CORRELATION SLOPES WITH ACC ################################################

def compute_slopes(df, x_col, y_col, pid_col='PROLIFIC_PID', suffix=''):
    '''Perform linear regression for each participant'''
    results = []
    for pid in df[pid_col].unique():
        subset = df[df[pid_col] == pid]
        if len(subset) > 1:
            slope, _, rvalue, pvalue, _ = linregress(subset[x_col], subset[y_col])
            results.append({'PROLIFIC_PID': pid, f'slope_{suffix}': slope, f'rvalue_{suffix}': rvalue})
    return pd.DataFrame(results)

# Compute slopes
slopes_social = compute_slopes(egoSocial, 'RTlog_cwc', 'RD', suffix='social')
slopes_spatial = compute_slopes(egoSpatial, 'RTlog_cwc', 'meanDistance', suffix='spatial')


slopes= pd.merge(slopes_social,slopes_spatial, on='PROLIFIC_PID')
slopes.to_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results\anchoring_biasSpatialSocialslopes.csv")

# Differences from zero.
slopes_long= pd.melt(slopes, id_vars='PROLIFIC_PID', value_vars=['slope_spatial', 'slope_social'], value_name='Slope')
from scipy.stats import ttest_1samp
t_stat, p_val = ttest_1samp(slopes['slope_spatial'], alternative='greater', popmean=0)
print(f"Slopes EgocentricSpatial t={t_stat:.2f}, p={p_val:.3f}")

t_stat, p_val = ttest_1samp(slopes['slope_social'], alternative='greater', popmean=0)
print(f"Slopes Egocentric Anchoring t={t_stat:.2f}, p={p_val:.3f}")

# Correlations between slopes with a individual specfici model
plt.figure(1, figsize=(5,5))
sns.barplot(y='Slope',x='variable', data=slopes_long,color='lightgrey')
sns.stripplot(y='Slope',x='variable', data=slopes_long,color='black', dodge=True,alpha=0.8, size=3)
plt.axhline(0.00, linestyle='--', color='red', linewidth=1)
plt.xlabel('')
plt.ylabel('Slopes', color='black',size=16, fontweight='bold')
plt.show()


## Correlation spatial vs social slopes ## 
x= slopes['slope_social']
y= slopes['slope_spatial']
#Estadistica
correlation, p_value = spearmanr(x, y)

#Grafico
plt.figure ( 1, figsize=(5,5))
sns.regplot(x=x, y=y,data=slopes, color='black')

plt.text(0.05, 0.95, f'SpearmanR: {correlation:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')
plt.text(0.05, 0.9, f'p-value: {p_value:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')
plt.xlabel('Slopes Social')
plt.ylabel('Slopes EgoSpatial')
plt.show()



#################################################################################################################
# From here we star using the LMixed MOdel from R results:
    # A. RT meanDistance_z with social slopes # 
    # B. Acc meanDistance_z with social slopes # 
    # C. RTmeanDistance vs egocentric performance
    # D. AccmeanDistance vs egocentric performance
    
#################################################################################################################
# A.meanDistance_z RT with social slopes #
#################################################################################################################
dfegoSpatialSocial= pd.merge(slopes, slopesSpatialRT, on='PROLIFIC_PID')
#Grafico
x=dfegoSpatialSocial['slope_social']
y=dfegoSpatialSocial['slope_random_meanDistance']
plt.figure ( 1, figsize=(5,5))
sns.regplot(x=x, y=y,data=dfegoSpatialSocial, color='black')

correlation, p_value = spearmanr(x, y)

plt.text(0.05, 0.95, f'SpearmanR: {correlation:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')
plt.text(0.05, 0.9, f'p-value: {p_value:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')

plt.xlabel('Slopes Social')
plt.ylabel('Coeff EgoSpatial')

plt.show()
#################################################################################################################
# B.meanDistance_z Acc with social slopes
#################################################################################################################

dfegoSpatialSocial= pd.merge(slopes, slopesSpatialAcc, on='PROLIFIC_PID')
#Grafico
x=dfegoSpatialSocial['slope_social']
y=dfegoSpatialSocial['meanDistance_z']
plt.figure ( 1, figsize=(5,5))
sns.regplot(x=x, y=y,data=dfegoSpatialSocial, color='black')

correlation, p_value = spearmanr(x, y)

plt.text(0.05, 0.95, f'SpearmanR: {correlation:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')
plt.text(0.05, 0.9, f'p-value: {p_value:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')

plt.xlabel('Slopes Social')
plt.ylabel('Coeff EgoSpatial Accuracy')

plt.show()

#################################################################################################################
#C. meanDistance_z con su propia performance EgoTask
#################################################################################################################
dfmeans= egoSpatial.groupby('PROLIFIC_PID').mean().reset_index()
dfAlloSpatialSocial= pd.merge(slopesSpatialRT, dfmeans, on='PROLIFIC_PID')
#Grafico
x=dfAlloSpatialSocial['slope_random_meanDistance']
y=dfAlloSpatialSocial['Accuracy']

plt.figure ( 1, figsize=(5,5))
sns.regplot(x=x, y=y,data=dfAlloSpatialSocial, color='black')

correlation, p_value = spearmanr(x, y)

plt.text(0.05, 0.95, f'SpearmanR: {correlation:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')
plt.text(0.05, 0.9, f'p-value: {p_value:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')

plt.xlabel('Slopes EgoSpatial Task')
plt.ylabel('Performance EgoSpatial Task')
plt.show()

##################################################################################################################

#D. meanDistance_z Accuracy con su propia performance EgoTask
#################################################################################################################
dfAlloSpatialSocial= pd.merge(slopesSpatialAcc, dfmeans, on='PROLIFIC_PID')
#Grafico
x=dfAlloSpatialSocial['meanDistance_z']
y=dfAlloSpatialSocial['Accuracy']

plt.figure ( 1, figsize=(5,5))
sns.regplot(x=x, y=y,data=dfAlloSpatialSocial, color='black')

correlation, p_value = spearmanr(x, y)

plt.text(0.05, 0.95, f'SpearmanR: {correlation:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')
plt.text(0.05, 0.9, f'p-value: {p_value:.2f}', fontsize=10, transform=plt.gca().transAxes,va='top', ha='left')

plt.xlabel('Slopes EgoSpatial Task Acc')
plt.ylabel('Performance EgoSpatial Task')
plt.show()

##################################################################################################################

