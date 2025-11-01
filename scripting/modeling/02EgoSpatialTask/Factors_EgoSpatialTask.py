# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 13:11:37 2025

@author: aramendi
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats


## Factors
paths_cleanedData= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
dfegoSpatial= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\egospatialtask.csv")
#dfegoSpatial= pd.read_csv(paths_cleanedData + "\\egospatialCleanedRT.csv")

participant_ID='PROLIFIC_PID'
## TYPE ENCONDED TRIAL

## Tryal type enconded

fig= plt.figure(5, figsize=(8,8))
dfegoSpatial_mean= dfegoSpatial.groupby([participant_ID,'Type_trialEncoded'])[['Accuracy','Response.rt']].mean().reset_index()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

plt.subplot(2,2,1)
sns.barplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.25, size=7,marker='o')
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')

plt.subplot(2,2,2)
sns.barplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.25, size=7,marker='o')
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')

fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


from scipy import stats
variables=['Accuracy','Response.rt']
for i in variables:
    acc_self     = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Type_trialEncoded'] == 'Self']
    acc_landmark = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Type_trialEncoded'] == 'Landmark']
    
    # paired t‑test
    t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)
    
    print(f"{i}: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")
    
    
 
'''   
## DIFFICULTY

dfegoSpatial_mean= dfegoSpatial.groupby([participant_ID,'Difficulty',])[['Accuracy','Response.rt']].mean().reset_index()

#### DIFFICULTY ####
fig= plt.figure(4, figsize=(8,8))
plt.subplot(2,2,1)
sns.pointplot(x='Difficulty', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black')
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Difficulty', color='black',size=15, fontweight='bold')
plt.xticks(size=14)

plt.subplot(2,2,2)
sns.pointplot(x='Difficulty', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black')
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
plt.xlabel('Difficulty', color='black',size=15, fontweight='bold')
plt.xticks(size=14)
fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

from scipy import stats
variables=['Response.rt','Accuracy']
for i in variables:
    acc_self     = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Difficulty'] == 'Easy']
    acc_landmark = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Difficulty'] == 'Hard']
    
    # paired t‑test
    t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)
    
    print(f"{i}: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")
'''

### SELF VS LANDMARK PROXIMITY
dfegoSpatial_mean= dfegoSpatial.groupby([participant_ID,'SelfvsLandmark_proximity',])[['Accuracy','Response.rt']].mean().reset_index()

fig=plt.figure(3, figsize=(8,8))
plt.subplot(2,2,1)
sns.barplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean, color='lightgrey')
sns.stripplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.25, size=7,marker='o')
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Self-Landmark proximity', color='black',size=15, fontweight='bold')
plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Landmark", "Self"],       # but show these texts instead
    rotation=0
)

plt.subplot(2,2,2)
sns.barplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.25, size=7,marker='o')
plt.xlabel('Self-Landmark proximity', color='black',size=15, fontweight='bold')
plt.ylabel('RT(ms)', color='black',size=16, fontweight='bold')
fig.subplots_adjust(wspace=0.5, hspace=0.25)
plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Landmark", "Self"],       # but show these texts instead
    rotation=0
)

plt.show()

from scipy import stats
variables=['Response.rt','Accuracy']
for i in variables:
    acc_self     = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['SelfvsLandmark_proximity'] == 'correct_landmark']
    acc_landmark = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['SelfvsLandmark_proximity'] == 'correct_self']
    
    # paired t‑test
    t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)
    
    print(f"{i}: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")

####################################################################################################################
######################################################## SELF PROXIMITY ############################################
####################################################################################################################
fig= plt.figure(2, figsize=(8,7))
plt.subplot(2,2,1)

dfegoSpatial_mean= dfegoSpatial.groupby([participant_ID,'Self_proximity',])[['Accuracy','Response.rt','distCorrSelf','AD']].mean().reset_index()
sns.barplot(x='Self_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='lightgrey')     
sns.stripplot(x='Self_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.25, size=7)
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Self proximity ', color='black',size=16, fontweight='bold')
plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Close", "Far"],       # but show these texts instead
    rotation=0
)


plt.subplot(2,2,2)
sns.barplot(x='Self_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Self_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.25, size=7)
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
plt.xlabel('Self proximity ', color='black',size=16, fontweight='bold')
fig.subplots_adjust(wspace=0.45)

plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Close", "Far"],       # but show these texts instead
    rotation=0
)


plt.show()
from scipy import stats
variables=['Response.rt','Accuracy']
for i in variables:
    acc_self     = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Self_proximity'] == 'self_close']
    acc_landmark = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Self_proximity'] == 'self_far']
    
    # paired t‑test
    t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)
    
    print(f"{i}: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")
####################################################################################################################