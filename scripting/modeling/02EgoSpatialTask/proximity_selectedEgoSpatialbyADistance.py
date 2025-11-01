# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:08:35 2025

@author: aramendi
"""

import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


'''This script will calculate the proximity select in the egocentric spatial task:
    Trials where participants select the closer option and how it is modulate by dificulty.'''
    
paths_cleanedData= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"

df= pd.read_csv(paths_cleanedData + "\\egospatialCleanedRT.csv")



# Open datasets
path_data= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"

## Database
egoSocial= pd.read_csv(path_data + "\\cleanedData\\egosocialCleanedRT.csv")
dfegoSpatial= pd.read_csv(path_data + "\\cleanedData\\egospatialCleanedRT.csv")
df4MT= pd.read_csv(path_data + "\\processed\\AlloTask_SpatialScore.csv")

# Open reuslts from models EgoSpatial
path_modelsR= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"
slopesSpatialRT= pd.read_csv(path_modelsR + "\\02EgoSpatialTask\\CoefsMixedLinearRT_meanDistance_egoTask.csv")
slopesSpatialAcc= pd.read_csv(path_modelsR + "\\02EgoSpatialTask\\CoefsMixedLogisticAcc_meanDistance_egoTask.csv")

# Open results from 4MT models
df4MTLinear= pd.read_csv(path_modelsR + "\\03AlloSpatialTask\\LinearRegressionAngDisp_4MT.csv")
df_slopes= pd.read_csv(path_modelsR + "\\03AlloSpatialTask\\allocentric_slopesQuatraticRegressions.csv")


# Open social slopes
social_slopesMixed= pd.read_csv(path_modelsR + "\\01SocialAnchor\\tables\\AnchorSlopes_RTlog_byParticipant.csv")
social_slopes= pd.read_csv(path_modelsR + "\\anchoring_biasSpatialSocialslopes.csv")

# Mean Dfs
df4mtmean_disparity= df4MT.groupby(['PROLIFIC_PID','angularDisparity'])['key_resp_3.corr'].mean().reset_index()
df4mtmean= df4MT.groupby('PROLIFIC_PID').mean().reset_index()
dfegoSpatial_mean= dfegoSpatial.groupby('PROLIFIC_PID').mean().reset_index()

# -----------------------------------------------------------------------------------------------------# 
# ------------------------------------PROXIMITY SELECTED ----------------------------------------------#

dfegoSpatial['dist_selected'] = np.nan  # creamos la columna vacía
dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 1, 'dist_selected'] = dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 1, 'distCorrSelf']
dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 0, 'dist_selected'] = dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 0, 'distIncorrSelf']

dfegoSpatial['proximity_selected'] = np.where(
    dfegoSpatial['Accuracy']==1,
    np.where(dfegoSpatial['distCorrSelf'] < dfegoSpatial['distIncorrSelf'], 1,0),
    np.where(dfegoSpatial['distIncorrSelf'] < dfegoSpatial['distCorrSelf'], 1,0)
)

dfegoSpatial["AD_z"] = dfegoSpatial.groupby("PROLIFIC_PID")["AD"].transform(lambda x: (x - x.mean()) / x.std())


results=[]
for pid, grp in dfegoSpatial.groupby('PROLIFIC_PID'):
    m = smf.glm("proximity_selected ~ AD_z ",
                data=grp, family=sm.families.Binomial()).fit()
    
    results.append({'PROLIFIC_PID': pid, 'intercept': m.params['Intercept'], 'coefs_AD':m.params['AD_z']})   
    
egospatialBias= pd.DataFrame(results)    


t_statistic, t_p_value = stats.ttest_1samp(
    egospatialBias['coefs_AD'], 
    popmean=0,
    alternative='greater'
)

print(f'Select the close option dependiend on AD": t = {t_statistic:.3f}, p = {t_p_value:.3f}')

egospatialBias.to_csv(paths_results +"\\02EgoSpatialTask\\coeff_select_closerOptionToSelf_AD.csv")
#-----------------------------------------------------------------------------------------------------------#



def Correlations_across_parameters(df1,df2, y_variable, x_variable): 
    '''This function create the fig and the correlation between the merge dfs
     depending on the variables of interes'''
     
    union= pd.merge(df1,df2, on='PROLIFIC_PID')
    plt.figure(figsize=(5,4))
    sns.regplot(data=union, y=y_variable, x=x_variable,ci=99, marker="o", color=".3", line_kws=dict(color="r"))
    
    if x_variable=='slopesRT_4MT' or x_variable=='slope_4MT':
        plt.xlabel('Spatial slopes', color='black',size=16, fontweight='bold')
    elif x_variable=='key_resp_3.corr':
        plt.xlabel('4MT Performance', color='black',size=16, fontweight='bold')
    elif x_variable=='key_resp_3.rt':
        plt.xlabel('4MT RT', color='black',size=16, fontweight='bold')
    
    if y_variable=='slope_spatial':
        plt.ylabel('Spatial slopes', color='black',size=16, fontweight='bold')
    elif y_variable=='slope_social':
            plt.ylabel('Social slopes', color='black',size=16, fontweight='bold')
    plt.show()
    
    print("------------------------------------------------------------------")
    print("----- CORRELATION between "+str(y_variable)+ " and "+str(x_variable)+" : ")
    print("Spearman Correlation: " +str(y_variable)+ " and "+str(x_variable)+":")
    res = stats.spearmanr(union[y_variable], union[x_variable])
    print(res)
    
    print("Pearson Correlation: " +str(y_variable)+ " and "+str(x_variable)+":")
    res = stats.pearsonr(union[y_variable], union[x_variable])
    print(res)
    print("------------------------------------------------------------------")
    print("    ")
    

Correlations_across_parameters(social_slopes,egospatialBias, 'slope_social','intercept')
Correlations_across_parameters(social_slopes,egospatialBias, 'slope_spatial','intercept')

egospatialBias['coefs_ADIndex']=  egospatialBias['coefs_AD']
df4MTLinear['slope_4MTIndex']= - df4MTLinear['slope_4MT']
Correlations_across_parameters(df4MTLinear,egospatialBias, 'intercept_Acc','intercept')
Correlations_across_parameters(df4MTLinear,egospatialBias, 'slope_4MT','coefs_AD')

Correlations_across_parameters(df4mtmean,df4MTLinear, 'key_resp_3.corr','slope_4MTIndex')

'''

## 

sns.lmplot(
    data=df,
    x="AD",
    y="proximity_selected",
    logistic=True,
    ci=None,
    scatter=False,
    line_kws={"alpha":0.08, "color":"#a40028"},hue='PROLIFIC_PID',legend=False
)
sns.regplot(data=df,
x="AD",
y="proximity_selected",color='black')
plt.xlabel("Difficulty")
plt.ylabel("p(selfClose_choseen)")
plt.tight_layout()
plt.show()


sns.lmplot(
    data=df,
    x="AD",
    y="proximity_selected",
    logistic=True,
    ci=None,
    scatter=False,
    line_kws={"alpha":1},hue='Self_proximity',legend=True
)

plt.xlabel("Difficulty")
plt.ylabel("p(selfClose_choseen)")
plt.tight_layout()
plt.show()
'''