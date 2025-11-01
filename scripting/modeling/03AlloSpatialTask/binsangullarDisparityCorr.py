# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:22:50 2025

@author: aramendi
"""
import pandas as pd
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression

dataset_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\processed"

df4MT = pd.read_csv(dataset_path + "//AlloTask_SpatialScore.csv")
########################################################################################################################
# Graphs with angular dispairty by bin angle
anova= df4MT.groupby(['PROLIFIC_PID','angularDisparity']).mean().reset_index()
anova = anova.copy()
anova['group'] = anova['angularDisparity'].replace({
    1:'40º',
    8:'40º',
    2:'80º',
    7:'80º',
    3: '120º',
    6: '120º',
    4: '160º',
    5: '160º'})

pair_map = {
    1: 1, 8:1,
    2: 2, 7: 2,
    3: 3, 6: 3,
    4: 4, 5:4 }

# create the new bin column
df4MT['bin'] = df4MT['angularDisparity'].map(pair_map)

anova['Accuracy'] = anova['key_resp_3.corr'] * 100

order = ['40º','80º','120º','160º']
pastel_colors = sns.color_palette("Blues", 4)


# 6) Dibuja el gráfico
plt.figure(figsize=(8, 6))

# Barras con paleta pastel
sns.barplot(
    data=anova,
    x='group',
    y='Accuracy',
    order=order,
    ci=None,
    palette=pastel_colors
)

# Línea y puntos de medias+CI
sns.pointplot(
    data=anova,
    x='group',
    y='Accuracy',
    order=order,
    ci=68,
    color='black',
    linestyles='dotted',
    markers='o',
    capsize=.09,
    errwidth=1,
    dodge=False
)

plt.xlabel('Angular Disparity', fontsize=16, fontweight='bold')
plt.ylabel('Performance', fontsize=16, fontweight='bold')
plt.ylim(0, None)
plt.tight_layout()
plt.show()
##########################################################################################################################

df4mtMeans= df4MT.groupby(['PROLIFIC_PID','bin'])['key_resp_3.corr','key_resp_3.rt'].mean().reset_index()
plt.figure(1, figsize=(10,15))
plt.subplot(2,2,1)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.barplot(data= df4mtMeans, x='bin', y=df4mtMeans['key_resp_3.corr']*100,color='lightgrey', ci=None)
sns.pointplot(
    data=df4mtMeans,
    x='bin',
    y=df4mtMeans['key_resp_3.corr']*100,
    ci=68,
    color='black',
    linestyles='dotted',
    markers='o',
    capsize=.09,
    errwidth=1,
    dodge=False, legend=False
)

plt.xlabel('Angular Disparity bins', color='black',size=16, fontweight='bold')
plt.ylabel('Performance (%Correct)', color='black',size=16, fontweight='bold')

plt.subplot(2,2,2)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.barplot(data= df4mtMeans, x='bin', y=df4mtMeans['key_resp_3.rt']*1000,color='lightgrey', ci=None)
sns.pointplot(
    data=df4mtMeans,
    x='bin',
    y=df4mtMeans['key_resp_3.rt']*1000,
    ci=68,
    color='black',
    linestyles='dotted',
    markers='o',
    capsize=.09,
    errwidth=1,
    dodge=False, legend=False
)

plt.xlabel('Angular Disparity bins', color='black',size=16, fontweight='bold')
plt.ylabel('RT(ms)', color='black',size=16, fontweight='bold')
plt.savefig(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\plots\AlloSpatial_LinearRTAcc.png", dpi=300, bbox_inches='tight')
plt.show()

##########################################################################################################################
# Linear regression 
results = []
predictors= ['bin']
df4MT["bin_z"] = df4MT.groupby("PROLIFIC_PID")["bin"].transform(lambda x: (x - x.mean()) / x.std())
for participant in df4MT['PROLIFIC_PID'].unique():
    data_subset = df4MT[df4MT['PROLIFIC_PID'] == participant]
    data_subset = data_subset.dropna(subset=['key_resp_3.corr', 'key_resp_3.rt', 'bin']) # Eliminate Nans
    
    data_subset['logRT'] = np.log(data_subset['key_resp_3.rt'])

    X = data_subset[predictors]
    X_acc= data_subset[['bin_z']]
    y = data_subset['logRT']
    yAcc= data_subset['key_resp_3.corr']
    
    # Reaction time regression
    modelRT= LinearRegression().fit(X,y)
    coefs_RT= modelRT.coef_
    
    modelAcc= LinearRegression().fit(X_acc,yAcc)
    coefs_Acc= modelAcc.coef_
    
    results.append({'PROLIFIC_PID': participant, 'slopesRT_4MT':coefs_RT[0],'intercept_Acc': modelAcc.intercept_, 'slope_4MT': coefs_Acc[0]})

    
    
linear4MT= pd.DataFrame(results)

output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results\03AlloSpatialTask"
linear4MT.to_csv(output_path + "\\LinearRegressionAngDisp_4MT.csv")
##########################################################################################################################
from scipy import stats

t_statistic, t_p_value = stats.ttest_1samp(
    linear4MT['slope_4MT'], 
    popmean=0,
    alternative='less'
)
print(f'LinearAcc: t = {t_statistic:.3f}, p = {t_p_value:.3f}')

##########################################################################################################################

##########################################################################################################################
from scipy import stats

t_statistic, t_p_value = stats.ttest_1samp(
    linear4MT['slopesRT_4MT'],  # Remeber LogRTs slopes
    popmean=0,
    alternative='greater'
)
print(f'LinearRT: t = {t_statistic:.3f}, p = {t_p_value:.3f}')

##########################################################################################################################

