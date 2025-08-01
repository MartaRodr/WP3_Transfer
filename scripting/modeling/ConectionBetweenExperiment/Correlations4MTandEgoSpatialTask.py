# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:55:31 2025

@author: aramendi
"""
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' 
In this script I will perform the correlation between differents coefficients
between 4MT Linear models ( Acc and RTs, single participants models) 
and the egospatial models ( Acc and RTs), extracted from the mixed models done in R
'''


################################################################################################
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
df_slopes= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\allocentric_slopesQuatraticRegressions.csv")

# Open social slopes
social_slopesMixed= pd.read_csv(path_modelsR + "\\01SocialAnchor\\tables\\AnchorSlopes_RTlog_byParticipant.csv")
social_slopes= pd.read_csv(path_modelsR + "\\anchoring_biasSpatialSocialslopes.csv")

# Mean Dfs
df4mtmean_disparity= df4MT.groupby(['PROLIFIC_PID','angularDisparity'])['key_resp_3.corr'].mean().reset_index()
df4mtmean= df4MT.groupby('PROLIFIC_PID')['key_resp_3.corr'].mean().reset_index()
dfegoSpatial_mean= dfegoSpatial.groupby('PROLIFIC_PID')['Accuracy'].mean().reset_index()

#####################################################################################

def Correlations_across_parameters(df1,df2, y_variable, x_variable): 
    '''This function create the fig and the correlation between the merge dfs
     depending on the variables of interes'''
     
    union= pd.merge(df1,df2, on='PROLIFIC_PID')
    plt.figure(figsize=(5,4))
    sns.regplot(data=union, y=y_variable, x=x_variable,ci=99, marker="o", color=".3", line_kws=dict(color="r"))
    #plt.ylabel('4MT performance', color='black',size=16, fontweight='bold')
    #plt.xlabel('Social slopes', color='black',size=16, fontweight='bold')
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
    
#####################################################################################

# Correlations between 4MT Linear Term and egospatial task
Correlations_across_parameters(df4mtmean, slopesSpatialAcc, 'meanDistance_z','key_resp_3.corr') 
Correlations_across_parameters(df4MTLinear, slopesSpatialAcc, 'slopesRT_4MT','meanDistance_z')
Correlations_across_parameters(df4MTLinear, slopesSpatialAcc, 'slope_4MT','meanDistance_z')

Correlations_across_parameters(slopesSpatialRT, df4mtmean, 'slope_random_meanDistance','key_resp_3.corr') 
Correlations_across_parameters(slopesSpatialRT,df4MTLinear, 'slopesRT_4MT','slope_random_meanDistance')    
Correlations_across_parameters(slopesSpatialRT,df4MTLinear, 'slope_4MT','slope_random_meanDistance')  
