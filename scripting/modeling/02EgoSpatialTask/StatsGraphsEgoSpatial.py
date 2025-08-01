# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:34:22 2025

@author: aramendi
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

dfegoSpatial= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\egospatialtask.csv")
dfegoSpatial_mean= dfegoSpatial.groupby(['PROLIFIC_PID'])[['Accuracy','Response.rt']].mean().reset_index()

dfegoSpatial['position'] = (
    dfegoSpatial['n_objeto']
      .str.extract(r'^([LR])', expand=False)          # grab leading L or R
      .map({'L': 'left', 'R': 'right'})               # map to words
)

fig= plt.figure(1, figsize=(5,8))
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

plt.subplot(2,2,1)
sns.barplot(y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='lightgrey')     
sns.stripplot(y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.4, size=4.5,marker='o')
plt.axhline(50, linestyle='--', color='red', linewidth=1)
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')

plt.subplot(2,2,2)
sns.barplot(y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.4, size=4.5,marker='o')
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
fig.subplots_adjust(wspace=0.55, hspace=0.25)
plt.show()

##########################################################################################################################################################
### CLENING DATABASE  EGO-SPATIAL
### Step 1: transformation RT to log
dfegoSpatial['meanDistance']= (dfegoSpatial['distCorrSelf'] + dfegoSpatial['distIncorrSelf'])/2
dfegoSpatial['RTlog_pre']= np.log(dfegoSpatial['Response.rt'])
dfegoSpatial['RTlog_pre']= np.log(dfegoSpatial['Response.rt'])
### Step 2: ELiminate outlier
out_sd_lo= dfegoSpatial['RTlog_pre'].mean() - (2.5 *  (dfegoSpatial['RTlog_pre'].std()))
out_sd_hi= dfegoSpatial['RTlog_pre'].mean()  + (2.5 *  (dfegoSpatial['RTlog_pre'].std()))
dfegoSpatial= dfegoSpatial.loc[(dfegoSpatial.RTlog_pre> out_sd_lo) & (dfegoSpatial.RTlog_pre< out_sd_hi)]
dfegoSpatial['RTlog_pre_mean'] = dfegoSpatial.groupby('PROLIFIC_PID')['RTlog_pre'].transform('mean')
dfegoSpatial['RTlog_pre_cwc']= dfegoSpatial['RTlog_pre'] - dfegoSpatial['RTlog_pre_mean']

##########################################################################################################################################################
                                                                    #### SELF PROXIMITY ####
##########################################################################################################################################################
# Bins
fig= plt.figure(2, figsize=(8,7))
plt.subplot(2,2,1)
dfegoSpatial_mean= dfegoSpatial.groupby(['PROLIFIC_PID','Self_proximity',])[['Accuracy','Response.rt','distCorrSelf','AD']].mean().reset_index()
sns.barplot(x='Self_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='lightgrey')     
sns.stripplot(x='Self_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Self proximity ', color='black',size=16, fontweight='bold')

plt.subplot(2,2,2)
sns.barplot(x='Self_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Self_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
plt.xlabel('Self proximity ', color='black',size=16, fontweight='bold')
fig.subplots_adjust(wspace=0.45)

plt.show()

################### Continuos graphs ################### Self proximity define as distance to the correct ball 
fig= plt.figure(20000, figsize=(10,10))
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.lmplot(x='distCorrSelf', y='Accuracy', data=dfegoSpatial, hue='PROLIFIC_PID', logistic=True, legend=False,scatter_kws={'s': 5,'alpha': 0.1},
           line_kws={'linewidth': 0.9}, ci=None,palette='Greys')

sns.regplot(x='distCorrSelf', y='Accuracy', data=dfegoSpatial,logistic=True, scatter=False, color='red')
plt.ylabel('Accuracy', color='black',size=16, fontweight='bold')
plt.xlabel('Self - CorrectObject distance ', color='black',size=16, fontweight='bold')
plt.show()


# By self close and self far
fig= plt.figure(20000, figsize=(10,10))
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

sns.regplot(x='distCorrSelf', y='Accuracy', data=dfegoSpatial.loc[dfegoSpatial['Self_proximity']=='self_close'],logistic=True, scatter=False, color='green')
sns.regplot(x='distCorrSelf', y='Accuracy', data=dfegoSpatial.loc[dfegoSpatial['Self_proximity']=='self_far'],logistic=True, scatter=False, color='blue')
plt.ylabel('Accuracy', color='black',size=20, fontweight='bold')
plt.xlabel('Self - CorrectObject distance ', color='black',size=20, fontweight='bold')
plt.show()


# Continuos
fig= plt.figure(20000, figsize=(10,10))
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


## REACTON TIME
sns.regplot(x='distCorrSelf', y='Response.rt', data=dfegoSpatial.loc[dfegoSpatial['Self_proximity']=='self_close'], scatter=False, color='green')
sns.regplot(x='distCorrSelf', y='Response.rt', data=dfegoSpatial.loc[dfegoSpatial['Self_proximity']=='self_far'], scatter=False, color='blue')
plt.ylabel('RT', color='black',size=20, fontweight='bold')
plt.xlabel('Self - CorrectObject distance ', color='black',size=20, fontweight='bold')
plt.show()



############## ACCURACY ####################### 
''' Este pack es para dibujar la grafica pero 
con distribucion de la variables tambien '''

pids    = dfegoSpatial['PROLIFIC_PID'].unique()
palette = sns.color_palette("Greys", n_colors=len(pids))

# 1) Figure + 2×1 GridSpec (top histogram, bottom joint plot)
fig = plt.figure(figsize=(6, 7))
gs  = gridspec.GridSpec(
    nrows=2, ncols=1,
    height_ratios=[1, 4],
    hspace=0
)

ax_histx = fig.add_subplot(gs[0, 0])  # top: distCorrSelf histogram
ax_joint = fig.add_subplot(gs[1, 0])  # bottom: scatter + logistic fits

# 2) Per‐PID logistic fits + points (Accuracy)
for pid, color in zip(pids, palette):
    sub = dfegoSpatial[dfegoSpatial['PROLIFIC_PID'] == pid]
    sns.regplot(
        x='meanDistance',
        y='Response.rt',
        data=sub,
        logistic=True,
        scatter_kws={'s': 10, 'alpha': 0.1, 'color': color},
        line_kws   ={'linewidth': 0.7, 'color': color},
        ci=None, 
        ax=ax_joint
    )

# 3) Overall logistic curve in black
sns.regplot(
    x='meanDistance', y='Accuracy',
    data=dfegoSpatial,
    scatter=False,
    color='red',
    ax=ax_joint
)
# 4) Top marginal histogram for distCorrSelf
sns.histplot(
    x=dfegoSpatial['meanDistance'],
    bins=30,
    fill=True,
    alpha=0.3,color='grey',
    linewidth=0.5,
    ax=ax_histx
)

# 5) Tidy up:
ax_histx.set_axis_off()  # hide ticks/labels on top
ax_joint.set_xlabel('Mean Distance choices', fontsize=16, fontweight='bold')
ax_joint.set_ylabel('RT(s)',            fontsize=16, fontweight='bold')

plt.show()

##########################################################################################################
results_df=pd.DataFrame()
results_df2=pd.DataFrame()
results_modelfull_df= pd.DataFrame()
participants= dfegoSpatial['PROLIFIC_PID'].unique() 

import statsmodels.formula.api as smf
for participant in participants:
        ## Fit logistic model for each participant:
        df_participant= dfegoSpatial.loc[dfegoSpatial['PROLIFIC_PID']== participant]

        # Fit logistic regresion
        model = smf.logit("Accuracy ~ AD", data=df_participant)
        model = model.fit_regularized(method='l1', alpha=0.1)
        
        model1 = smf.logit("Accuracy ~ meanDistance", data=df_participant)
        model1= model1.fit_regularized(method='l1', alpha=0.1)
        
        model_full= smf.logit("Accuracy ~ meanDistance +  AD", data=df_participant)
        model_full= model_full.fit_regularized(method='l1', alpha=0.1)
        
        #Results
        results= pd.DataFrame({'PROLIFIC_PID': [participant], 
                               'b1_AD': [model.params['AD']],
                               'oddRatioAD':np.exp([model.params['AD']])})
        results_df=pd.concat([results,results_df])
        
        results_model2= pd.DataFrame({'PROLIFIC_PID': [participant], 
                                      'b1_MeanDistance': [model1.params['meanDistance']],
                                      'oddRatioMeanDistance':np.exp([model1.params['meanDistance']])})
        results_df2=pd.concat([results_model2,results_df2])
        
        results_modelfull= pd.DataFrame({'PROLIFIC_PID': [participant], 
                                         'coefs_MeanDistance': [model_full.params['meanDistance']],
                                         'coefs_AD': [model_full.params['AD']],
                                         'oddRatioAD':np.exp([model_full.params['AD']]),
                                         'oddRatioMeanDistance':np.exp([model_full.params['meanDistance']])})
        
        results_modelfull_df=pd.concat([results_modelfull_df,results_modelfull])
        


cofs= pd.merge(results_df,results_df2, on='PROLIFIC_PID')  
cofs.to_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\coefficients_EachParticipants_logisticEgoSpatialTask.csv")


#### RESULTS FROM THE REGRESION MODEL IN R MIXED LOGISTIC REGRESION MODEL 

############################################################################################################
#LOGISTIC REGRESSIONS: Coefs different from zero?
############################################################################################################
from scipy.stats import wilcoxon
stat, p = wilcoxon(results_modelfull_df['coefs_MeanDistance'], zero_method='wilcox', alternative='two-sided', mode='auto')
print(f"Wilcoxon signed‐rank for meanDistance: W = {stat:.2f}, p = {p:.4f}")

stat, p = wilcoxon(results_modelfull_df['coefs_AD'], zero_method='wilcox', alternative='two-sided', mode='auto')
print(f"Wilcoxon signed‐rank for AD: W = {stat:.2f}, p = {p:.4f}")

fig= plt.figure(5, figsize=(5,5))
sns.set_theme(style="white", palette=None)
plt.subplot(2,2,1)
sns.barplot(data=cofs, y="coefs_MeanDistance", ci=68, color='grey')
sns.stripplot(y=cofs['meanDistance'],color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('Beta coefficients', color='black',size=15, fontweight='bold')
plt.xlabel('coefs MeanDistance', color='black',size=15, fontweight='bold')

plt.subplot(2,2,2)
sns.barplot(data=cofs, y="AD", ci=68, color='grey')
sns.stripplot(y=cofs['AD'],color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('Beta coefficients', color='black',size=15, fontweight='bold')
plt.xlabel('coefs AD', color='black',size=15, fontweight='bold')
fig.subplots_adjust(wspace=0.5, hspace=0.25)
plt.show()


############################################################################################################
                    ############## REACTION TIMES #######################
                    
# Prepare your data & palette
pids     = dfegoSpatial['PROLIFIC_PID'].unique()
palette  = sns.color_palette("Greys", n_colors=len(pids))

# Create figure + GridSpec layout
fig = plt.figure(figsize=(8,8))
gs  = gridspec.GridSpec(
    nrows=2, ncols=2,
    width_ratios =[4, 1],
    height_ratios=[1, 4],
    hspace=0, wspace=0
)

# Axes definitions
ax_histx = fig.add_subplot(gs[0, 0])   # top histogram
ax_joint = fig.add_subplot(gs[1, 0])   # main scatter+regression
ax_histy = fig.add_subplot(gs[1, 1])   # right histogram

# 1) Per-PID regressions + points
for pid, color in zip(pids, palette):
    sub = dfegoSpatial[dfegoSpatial['PROLIFIC_PID'] == pid]
    sns.regplot(
        x='meanDistance', y='Response.rt',
        data=sub,
        scatter_kws={'s': 10, 'alpha': 0.1, 'color': color},
        line_kws   ={'linewidth': 0.7, 'color': color},
        ci=None,
        ax=ax_joint
    )

# 2) Overall regression in black
sns.regplot(
    x='meanDistance', y='Response.rt',
    data=dfegoSpatial,
    scatter=False,
    color='red',
    ax=ax_joint
)

# 3) Top marginal: histogram of distCorrSelf
sns.histplot(
    x=dfegoSpatial['meanDistance'],
    bins=30,
    fill=True,
    alpha=0.3,
    ax=ax_histx, color='grey'
)

# 4) Right marginal: histogram of Response.rt (horizontal)
sns.histplot(
    y=dfegoSpatial['Response.rt'],
    bins=30,
    fill=True,
    alpha=0.3,
    ax=ax_histy,color='grey'
)

# 5) Clean up marginals
ax_histx.set_axis_off()
ax_histy.set_axis_off()

# 6) Labels on main plot
ax_joint.set_xlabel('MeanDistance Choices', fontsize=16, fontweight='bold')
ax_joint.set_ylabel('RT (s)',                    fontsize=16, fontweight='bold')

plt.show()

############################################################################################################
import numpy as np
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(5, 5))
gs = GridSpec(2, 2, figure=fig)

###Step 1: transformation RT to log
dfegoSpatial['RTlog']= np.log(dfegoSpatial['Response.rt'])

### ELiminate outlier
out_sd_lo= dfegoSpatial['RTlog'].mean() - (2.5*  (dfegoSpatial['RTlog'].std()))
out_sd_hi= dfegoSpatial['RTlog'].mean()  + (2.5*  (dfegoSpatial['RTlog'].std()))
dfegoSpatial= dfegoSpatial.loc[(dfegoSpatial.RTlog> out_sd_lo) & (dfegoSpatial.RTlog< out_sd_hi)]
dfegoSpatial['RTlog_mean'] = dfegoSpatial.groupby('PROLIFIC_PID')['RTlog'].transform('mean')
dfegoSpatial['std'] = dfegoSpatial.groupby('PROLIFIC_PID')['RTlog'].transform('std')
dfegoSpatial['RTlog_cwc']= (dfegoSpatial['RTlog'] - dfegoSpatial['RTlog_mean'])
dfegoSpatial['RTlog_std']= (dfegoSpatial['RTlog'] - dfegoSpatial['RTlog_mean'])/(dfegoSpatial['std'] )


ax1 = fig.add_subplot(gs[0, 0])
sns.set_theme(style="white", palette=None)
jg1 = sns.JointGrid(data=dfegoSpatial, x="RTlog_cwc", y="meanDistance")
sns.regplot(data=dfegoSpatial, x="RTlog_cwc", y="meanDistance", color="red", scatter=False, ax=jg1.ax_joint)
jg1.ax_marg_x.hist(dfegoSpatial["RTlog_cwc"], color="grey")
jg1.ax_marg_y.hist(dfegoSpatial["meanDistance"], color="grey", orientation="horizontal")
# Set axis labels
jg1.ax_joint.set_ylabel("MeanDistance choices", fontsize=16,fontweight='bold')  # Change "Custom X-axis Label" to your desired label
jg1.ax_joint.set_xlabel("RT (log-transformed)", fontsize=16,fontweight='bold')  # Change "Custom Y-axis Label" to your desired label

jg1.ax_joint.set_xlim(-2, +2)
jg1.ax_joint.set_ylim(0, 10)
plt.show()

rt = dfegoSpatial.groupby(['PROLIFIC_PID'])['Response.rt'].agg(['mean','sem','std']).reset_index()
print('Participants mean anchor rt:' + str(rt['mean'].mean()) +  ' with a SEM of: '+ str(rt['sem'].mean()))

#############################################################################################################
############################################################################################################
import numpy as np
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(5, 5))
gs = GridSpec(2, 2, figure=fig)

###Step 1: transformation RT to log
dfegoSpatial['RTlog']= np.log(dfegoSpatial['Response.rt'])

### ELiminate outlier
out_sd_lo= dfegoSpatial['RTlog'].mean() - (2.5*  (dfegoSpatial['RTlog'].std()))
out_sd_hi= dfegoSpatial['RTlog'].mean()  + (2.5*  (dfegoSpatial['RTlog'].std()))
dfegoSpatial= dfegoSpatial.loc[(dfegoSpatial.RTlog> out_sd_lo) & (dfegoSpatial.RTlog< out_sd_hi)]
dfegoSpatial['RTlog_mean'] = dfegoSpatial.groupby('PROLIFIC_PID')['RTlog'].transform('mean')
dfegoSpatial['std'] = dfegoSpatial.groupby('PROLIFIC_PID')['RTlog'].transform('std')
dfegoSpatial['RTlog_cwc']= (dfegoSpatial['RTlog'] - dfegoSpatial['RTlog_mean'])
dfegoSpatial['RTlog_std']= (dfegoSpatial['RTlog'] - dfegoSpatial['RTlog_mean'])/(dfegoSpatial['std'] )


ax1 = fig.add_subplot(gs[0, 0])
sns.set_theme(style="white", palette=None)
jg1 = sns.JointGrid(data=dfegoSpatial, x="RTlog_cwc", y="AD")
sns.regplot(data=dfegoSpatial, x="RTlog_cwc", y="AD", color="red", scatter=False, ax=jg1.ax_joint)
jg1.ax_marg_x.hist(dfegoSpatial["RTlog_cwc"], color="grey")
jg1.ax_marg_y.hist(dfegoSpatial["AD"], color="grey", orientation="horizontal")
# Set axis labels
jg1.ax_joint.set_ylabel("Absolute distance", fontsize=16,fontweight='bold')  # Change "Custom X-axis Label" to your desired label
jg1.ax_joint.set_xlabel("RT (log-transformed)", fontsize=16,fontweight='bold')  # Change "Custom Y-axis Label" to your desired label

jg1.ax_joint.set_xlim(-2, +2)
jg1.ax_joint.set_ylim(0, 5)
plt.show()

rt = dfegoSpatial.groupby(['PROLIFIC_PID'])['Response.rt'].agg(['mean','sem','std']).reset_index()
print('Participants mean anchor rt:' + str(rt['mean'].mean()) +  ' with a SEM of: '+ str(rt['sem'].mean()))

###############################################################################################################
#### SELF VS LANDMARK PROXIMITY ####
dfegoSpatial_mean= dfegoSpatial.groupby(['PROLIFIC_PID','SelfvsLandmark_proximity',])[['Accuracy','Response.rt']].mean().reset_index()

fig=plt.figure(3, figsize=(8,8))
plt.subplot(2,2,1)
sns.barplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean, color='lightgrey')
sns.stripplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.4, size=4.5,marker='o')
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Self-Landmark proximity', color='black',size=15, fontweight='bold')
plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Landmark", "Self"],       # but show these texts instead
    rotation=0
)

plt.subplot(2,2,2)
sns.barplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='SelfvsLandmark_proximity', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.4, size=4.5,marker='o')
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
    
    print(f"str(i)i: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")


dfegoSpatial_mean= dfegoSpatial.groupby(['PROLIFIC_PID','Difficulty',])[['Accuracy','Response.rt']].mean().reset_index()

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
    
    print(f"str(i)i: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")

## Tryal tupe encond

fig= plt.figure(5, figsize=(8,8))
dfegoSpatial_mean= dfegoSpatial.groupby(['PROLIFIC_PID','Type_trialEncoded','Difficulty'])[['Accuracy','Response.rt','AD']].mean().reset_index()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

plt.subplot(2,2,1)
sns.barplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.4, size=4.5,marker='o')
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')

plt.subplot(2,2,2)
sns.barplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Type_trialEncoded', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.4, size=4.5,marker='o')
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')

fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


from scipy import stats
variables=['Response.rt','AD']
for i in variables:
    acc_self     = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Type_trialEncoded'] == 'Self']
    acc_landmark = dfegoSpatial_mean[i].loc[dfegoSpatial_mean['Type_trialEncoded'] == 'Landmark']
    
    # paired t‑test
    t_stat_scipy, p_value_scipy = stats.ttest_rel(acc_self, acc_landmark)
    
    print(f"str(i)i: t = {t_stat_scipy:.3f}, p = {p_value_scipy:.3f}")

#############################
fig= plt.figure(5, figsize=(8,8))
dfegoSpatial_mean= dfegoSpatial.groupby(['PROLIFIC_PID','Angle'])[['Accuracy','Response.rt','AD']].mean().reset_index()

plt.subplot(2,3,1)
sns.barplot(x='Angle', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Angle', y=dfegoSpatial_mean['Accuracy']*100, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')

plt.subplot(2,3,2)
sns.barplot(x='Angle', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Angle', y=dfegoSpatial_mean['Response.rt']*1000, data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')

plt.subplot(2,3,3)
sns.barplot(x='Angle', y='AD', data=dfegoSpatial_mean,color='lightgrey')
sns.stripplot(x='Angle', y='AD', data=dfegoSpatial_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('AD', color='black',size=16, fontweight='bold')
plt.xlabel('Encoded Trial', color='black',size=15, fontweight='bold')


fig.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=dfegoSpatial, x="Self_proximity", y="Response.rt", hue="Difficulty", col="SelfvsLandmark_proximity",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)


#################################################################################################################
# Speed Accuracy ##

import pandas as pd

summary = (
    dfegoSpatial
    .groupby(['PROLIFIC_PID','Self_proximity', 'Difficulty', 'SelfvsLandmark_proximity'])
    .apply(lambda g: pd.Series({
        'accuracy'   : g['Accuracy'].mean(),                              
        'mean_rt_s'  : g.loc[g['Accuracy']==1, 'Response.rt'].mean(),             
    }))
    .reset_index()
)

# now compute Efficiency = accuracy / mean_rt_s
summary['efficiency'] = summary['accuracy'] / summary['mean_rt_s']


g = sns.catplot(
    data=summary, x="Self_proximity", y="efficiency", hue="Difficulty", col="SelfvsLandmark_proximity",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)


############################################################################################################


