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
slopesSpatialRTdistCorr= pd.read_csv(path_modelsR + "\\02EgoSpatialTask\\CoefsMixedLinearRT_selfCorrectDistance_egoTask.csv")

# Open results from 4MT models
df4MTLinear= pd.read_csv(path_modelsR + "\\03AlloSpatialTask\\LinearRegressionAngDisp_4MT.csv")
df_slopes= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\allocentric_slopesQuatraticRegressions.csv")

# Open social slopes
social_slopesMixed= pd.read_csv(path_modelsR + "\\01SocialAnchor\\tables\\AnchorSlopes_RTlog_byParticipant.csv")
social_slopes= pd.read_csv(path_modelsR + "\\anchoring_biasSpatialSocialslopes.csv")

# Mean Dfs
df4mtmean_disparity= df4MT.groupby(['PROLIFIC_PID','angularDisparity'])['key_resp_3.corr'].mean().reset_index()
df4mtmean= df4MT.groupby('PROLIFIC_PID').mean().reset_index()
dfegoSpatial_mean= dfegoSpatial.groupby('PROLIFIC_PID').mean().reset_index()

#####################################################################################

def Correlations_across_parameters(df1,df2, y_variable, x_variable): 
    '''This function create the fig and the correlation between the merge dfs
     depending on the variables of interes'''
   
        
    union= pd.merge(df1,df2, on='PROLIFIC_PID')
    plt.figure(figsize=(5,4))
    sns.regplot(data=union, y=y_variable, x=x_variable,ci=99, marker="o", color=".3", line_kws=dict(color="r"))
    
    if df2 is slopesSpatialAcc and  y_variable=='key_resp_3.corr':
        
        plt.xlabel('meanDistance slopes Acc', color='black',size=16, fontweight='bold')
        plt.ylabel('4MT Performance', color='black',size=16, fontweight='bold')
        plt.savefig(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\plots\AllovsEgoSpatial_PerformanceVSAccSlopes.png", dpi=300, bbox_inches='tight')
        
    elif df2 is slopesSpatialRT and y_variable=='key_resp_3.corr': 
        plt.xlabel('meanDistance slopes RT', color='black',size=16, fontweight='bold')
        plt.ylabel('4MT Performance', color='black',size=16, fontweight='bold')
        plt.savefig(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\plots\AllovsEgoSpatial_PerformanceVSRTSlopes.png", dpi=300, bbox_inches='tight')
        
    
    elif (df2 is df4mtmean and x_variable=='key_resp_3.corr') and (df1 is dfegoSpatial_mean and y_variable=='Accuracy'): 
            plt.ylabel('EgoSpatialTask Performance', color='black',size=16, fontweight='bold')
            plt.xlabel('4MT Performance', color='black',size=16, fontweight='bold')
            plt.savefig(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\plots\AllovsEgoSpatial_Performance.png", dpi=300, bbox_inches='tight')
            
    elif (df2 is df4mtmean and x_variable=='key_resp_3.rt') and (df1 is dfegoSpatial_mean and y_variable=='Response.rt'): 
            plt.ylabel('EgoSpatialTask RT', color='black',size=16, fontweight='bold')
            plt.xlabel('4MT RT', color='black',size=16, fontweight='bold')
            plt.savefig(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\plots\AllovsEgoSpatial_RTs.png", dpi=300, bbox_inches='tight')

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
Correlations_across_parameters(df4mtmean, slopesSpatialAcc, 'key_resp_3.corr','meanDistance_z') 
Correlations_across_parameters(df4MTLinear, slopesSpatialAcc, 'slopesRT_4MT','meanDistance_z')
Correlations_across_parameters(df4MTLinear, slopesSpatialAcc, 'slope_4MT','meanDistance_z')

Correlations_across_parameters( df4mtmean,slopesSpatialRT,'key_resp_3.corr', 'slope_random_meanDistance') 
Correlations_across_parameters( df4MTLinear,slopesSpatialRTdistCorr,'slopesRT_4MT', 'slope_selfCorrect') 

Correlations_across_parameters(df4MTLinear,slopesSpatialRT, 'slopesRT_4MT','slope_random_meanDistance')    
Correlations_across_parameters(df4MTLinear,slopesSpatialRT, 'df4MTLinear','slope_random_meanDistance')  

Correlations_across_parameters(dfegoSpatial_mean,df4mtmean, 'Accuracy','key_resp_3.corr')  
Correlations_across_parameters(dfegoSpatial_mean,df4mtmean, 'Response.rt','key_resp_3.rt')  

fig, axes = plt.subplots(1, 2, figsize=(7,3))
ymin, ymax = 0.2, 1
yticks = np.arange(0.2, 1.01, 0.2)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

#color
cmap = sns.color_palette("dark:#5A9_r", as_cmap=True)
greenish_color = 'grey'  # ~0.6–0.7 da un tono verde-azulado bonito

all_dfs= pd.merge(df4mtmean, dfegoSpatial_mean, on='PROLIFIC_PID')

for ax, (var, label) in zip(
    axes,
    [
        ('key_resp_3.corr', '4MT Performance'),
        ('Accuracy', 'EgoSpatial Performance')
    ]
):
    sns.barplot(y=var, data=all_dfs, color='lightgrey', ax=ax)
    sns.stripplot(y=var, data=all_dfs, color=greenish_color, alpha=0.6, size=7, marker='o', ax=ax)
    ax.set_ylabel(label, color='black', size=14, fontweight='bold', labelpad=3)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    
    # 🔹 Estilo de los números del eje Y
    ax.tick_params(axis='y', labelsize=12, width=1.2, color='black', direction='out', length=3)
    for label_tick in ax.get_yticklabels():
        label_tick.set_fontweight('bold')
        label_tick.set_fontname('Arial')  # puedes cambiarlo por 'Calibri', 'Times New Roman', etc.
    
    # 🔹 Formato bonito (quita ceros extra, usa un decimal)
    ax.set_yticklabels([f'{y:.1f}' for y in yticks])
    
    # 🔹 Quitar la raya del eje x
    ax.tick_params(bottom=False)

fig.subplots_adjust(wspace=0.6, hspace=0.4)
plt.show()