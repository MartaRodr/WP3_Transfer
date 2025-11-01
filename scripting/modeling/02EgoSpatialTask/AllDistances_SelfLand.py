# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 08:13:08 2025

@author: aramendi
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats

#Models encoding:
output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\processed"
ego= pd.read_csv(output_path + "\\egospatialTask_SpatialScore.csv")



ego['delta_self']=abs( ego['distOrigSelf']- ego['distCorrSelf'])
ego['delta_landmark']=  abs(ego['distOrigLand']- ego['distCorrLand'])

results_acc = []
results_rt  = []


def zscore_safe(s):
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return s*0  # all zeros (centered, no variance)
    return (s - s.mean()) / sd

results_acc, results_rt = [], []

for pid in ego['PROLIFIC_PID'].unique():
    data = ego.loc[ego['PROLIFIC_PID'] == pid].copy()

    # Normalize RT column name (handle both variants)
    if 'Response.rt' in data.columns:
        data = data.rename(columns={'Response.rt': 'Response_RT'})
    if 'Response.RT' in data.columns:
        data = data.rename(columns={'Response.RT': 'Response_RT'})

    # Drop rows with NaNs in what we might use
    needed = ['Accuracy', 'Response_RT',
              'distOrigSelf','distOrigLand','distCorrSelf','distCorrLand']
    data = data.dropna(subset=[c for c in needed if c in data.columns])

    # Skip if too few rows
    if len(data) < 10:
        results_acc.append({'PROLIFIC_PID': pid, 'error_acc': 'too_few_rows'})
        results_rt.append({'PROLIFIC_PID': pid, 'error_rt': 'too_few_rows'})
        continue

    # Safe z-scoring
    for col in ['distOrigSelf','distOrigLand','distCorrSelf','distCorrLand']:
        if col in data.columns:
            data[f'{col}_c'] = zscore_safe(data[col])

    # Build interaction formula with whatever is present
    dist_vars = [v for v in
                 ['distOrigSelf_c','distOrigLand_c','distCorrSelf_c','distCorrLand_c']
                 if v in data.columns]
    if len(dist_vars) < 2:
        results_acc.append({'PROLIFIC_PID': pid, 'error_acc': 'not_enough_predictors'})
        if 'Response_RT' in data.columns:
            results_rt.append({'PROLIFIC_PID': pid, 'error_rt': 'not_enough_predictors'})
        continue

    inter_formula = " * ".join(dist_vars)  # expands to mains + all interactions

    # --------- GLM Binomial (Accuracy) ---------
    try:
        mdl_acc = sm.GLM.from_formula(
            f"Accuracy ~ {inter_formula}",
            data=data,
            family=sm.families.Binomial()
        ).fit()

        coef_acc = mdl_acc.params.to_dict()
        coef_acc['PROLIFIC_PID'] = pid
        results_acc.append(coef_acc)

    except Exception as e:
        results_acc.append({'PROLIFIC_PID': pid, 'error_acc': str(e)})

    # --------- OLS (RT) ---------
    if 'Response_RT' in data.columns:
        try:
            mdl_rt = sm.OLS.from_formula(
                f"Response_RT ~ {inter_formula}",
                data=data
            ).fit()

            coef_rt = mdl_rt.params.to_dict()
            coef_rt['PROLIFIC_PID'] = pid
            results_rt.append(coef_rt)

        except Exception as e:
            results_rt.append({'PROLIFIC_PID': pid, 'error_rt': str(e)})

# Build DataFrames
df_betas_acc = pd.DataFrame(results_acc)
df_betas_rt  = pd.DataFrame(results_rt)

# Optional long format
df_betas_acc_long = df_betas_acc.melt(id_vars='PROLIFIC_PID', var_name='Predictors', value_name='Coefficients')\
                                .dropna(subset=['Coefficients'])
df_betas_acc_long["Coefficients"] = pd.to_numeric(df_betas_acc_long["Coefficients"], errors='coerce')

df_betas_rt_long  = df_betas_rt.melt(id_vars='PROLIFIC_PID', var_name='Predictors', value_name='Coefficients')\
                               .dropna(subset=['Coefficients'])

# Change name 
rename_base = {
    "Intercept": "I",
    "distOrigSelf_c": "SO",
    "distOrigLand_c": "LO",
    "distCorrSelf_c": "SC",
    "distCorrLand_c": "LC",
    "error_acc": "error_acc"
}

def rename_predictor(term):
    if term in rename_base:
        return rename_base[term]
    if ":" in term:
        return ":".join(rename_base.get(p, p) for p in term.split(":"))
    return term

df_betas_acc_long["Predictor_short"] = df_betas_acc_long["Predictors"].apply(rename_predictor)
df_betas_rt_long["Predictor_short"] = df_betas_rt_long["Predictors"].apply(rename_predictor)

variables= df_betas_rt_long['Predictor_short'].unique()

## RT test significant ##
pairs = [
    ("Accuracy model", df_betas_acc_long),
    ("RT model",       df_betas_rt_long),
]

for label, df in pairs:
    if label=="Accuracy model":
        print("Accuracy model")
    elif label=="RT model":
        print ("    ")
        print("RT model")
    for predictor in variables:
        df_betas= df.loc[df['Predictor_short']==predictor]
        vals = df_betas['Coefficients'].dropna()
        
        t, p = stats.ttest_1samp(vals, 0)
        print(f"{predictor}: t={t:.3f}, p={p:.4g}")
        


import matplotlib.pyplot as plt
import seaborn as sns

## PLOT COEFFICIENTS ACCURY MODELS
fig=plt.figure(1, figsize=(4,5))
sns.barplot(x='Predictor_short', y='Coefficients', data=df_betas_acc_long ,color='lightgrey')
sns.stripplot(x='Predictor_short', y='Coefficients', data=df_betas_acc_long,color='black', dodge=True,alpha=0.1, size=5,marker='o')
plt.ylabel('Coefficients', color='black',size=16, fontweight='bold')
plt.xlabel('Predictors', color='black',size=15, fontweight='bold')

plt.xticks(rotation=90)

plt.axhline(y=0, color="black", linewidth=0.9,linestyle='--',)
plt.show()


## PLOT COEFFICIENTS REACTION TIME  MODELS

fig=plt.figure(1, figsize=(4,5))
sns.barplot(x='Predictor_short', y='Coefficients', data=df_betas_rt_long ,color='lightgrey')
sns.stripplot(x='Predictor_short', y='Coefficients', data=df_betas_rt_long,color='black', dodge=True,alpha=0.1, size=5,marker='o')
plt.ylabel('Coefficients', color='black',size=16, fontweight='bold')
plt.xlabel('Predictors', color='black',size=15, fontweight='bold')
plt.xticks(rotation=90)
plt.axhline(y=0, color="black", linewidth=0.9,linestyle='--',)
plt.show()


## Correlation with 4MT performance
path_data= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"

## Database
egoSocial= pd.read_csv(path_data + "\\cleanedData\\egosocialCleanedRT.csv")
dfegoSpatial= pd.read_csv(path_data + "\\cleanedData\\egospatialCleanedRT.csv")
slopes= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results\anchoring_biasSpatialSocialslopes.csv") # Save results
df4MT= pd.read_csv(path_data + "\\processed\\AlloTask_SpatialScore.csv")
df4MTmean= df4MT.groupby('PROLIFIC_PID').mean().reset_index()

path_modelsR= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"
df4MTLinear= pd.read_csv(path_modelsR + "\\03AlloSpatialTask\\LinearRegressionAngDisp_4MT.csv")

def Correlations_across_parameters(df1,df2, y_variable, x_variable): 
    '''This function create the fig and the correlation between the merge dfs
     depending on the variables of interes'''
   
        
    union= pd.merge(df1,df2, on='PROLIFIC_PID')
    plt.figure(figsize=(5,4))
    sns.regplot(data=union, y=y_variable, x=x_variable,ci=99, marker="o", color=".3", line_kws=dict(color="r"))
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

Correlations_across_parameters(df4MTmean, df_betas_acc, 'key_resp_3.corr','beta_interaction') 
Correlations_across_parameters(df4MTLinear, df_betas_acc, 'slopesRT_4MT','beta_interaction') 
Correlations_across_parameters(df4MTLinear, df_betas_acc, 'slope_4MT','beta_interaction') 


Correlations_across_parameters(df4MTmean, df_betas_rt, 'key_resp_3.corr','beta_interaction') 
Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slopesRT_4MT','beta_interaction') 
Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slope_4MT','beta_interaction') 

Correlations_across_parameters(df4MTmean, df_betas_rt, 'key_resp_3.corr','beta_land') 
Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slopesRT_4MT','beta_land') 
Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slope_4MT','beta_land') 
 
 
Correlations_across_parameters(df4MTmean, df_betas_acc, 'key_resp_3.corr','beta_self') 
Correlations_across_parameters(df4MTLinear, df_betas_acc, 'slopesRT_4MT','beta_self') 
Correlations_across_parameters(df4MTLinear, df_betas_acc, 'slope_4MT','beta_self') 



# Correlations model ACC
Correlations_across_parameters(df4MTmean, df_betas_rt, 'key_resp_3.corr','distOrigSelf_c:distOrigLand_c:distCorrSelf_c:distCorrLand_c') 


'distOrigSelf_c', 'distOrigLand_c',
       'distOrigSelf_c:distOrigLand_c', 'distCorrSelf_c',
       'distOrigSelf_c:distCorrSelf_c', 'distOrigLand_c:distCorrSelf_c',
       'distOrigSelf_c:distOrigLand_c:distCorrSelf_c', 'distCorrLand_c',
       'distOrigSelf_c:distCorrLand_c', 'distOrigLand_c:distCorrLand_c',
       'distOrigSelf_c:distOrigLand_c:distCorrLand_c',
       'distCorrSelf_c:distCorrLand_c',
       'distOrigSelf_c:distCorrSelf_c:distCorrLand_c',
       'distOrigLand_c:distCorrSelf_c:distCorrLand_c',
       'distOrigSelf_c:distOrigLand_c:distCorrSelf_c:distCorrLand_c'




