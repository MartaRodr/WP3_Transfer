# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 08:13:08 2025

@author: aramendi
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats

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

#Models encoding:
output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\processed"
ego= pd.read_csv(output_path + "\\egospatialTask_SpatialScore.csv")

ego['delta_self']=ego['distOrigSelf']- ego['distCorrSelf']
ego['delta_landmark']= ego['distOrigLand']- ego['distCorrLand']

results_acc = []
results_rt  = []

for pid in ego['PROLIFIC_PID'].unique():
    data = ego.loc[ego['PROLIFIC_PID'] == pid].copy()

    # Keep only rows with necessary columns (avoid NaNs)
    needed_cols = ['Accuracy', 'delta_self', 'delta_landmark', 'Response.rt']
    data = data.dropna(subset=[c for c in needed_cols if c in data.columns])

    # Standardize within participant
    data["distOrigSelf_c"]  = (data["delta_self"]  - data["delta_self"].mean())  / data["delta_self"].std()
    data["distOrigLand_c"]  = (data["delta_landmark"]  - data["delta_landmark"].mean())  / data["delta_landmark"].std()
    data["AD_z"]  = (data["AD"]  - data["AD"].mean())  / data["AD"].std()

    '''
    # Standardize within participant
    data["distOrigSelf_c"]  = (data["distCorrSelf"]  - data["distCorrSelf"].mean())  / data["distCorrSelf"].std()
    data["distOrigLand_c"]  = (data["distCorrLand"]  - data["distCorrLand"].mean())  / data["distCorrLand"].std()
    data["AD_z"]  = (data["AD"]  - data["AD"].mean())  / data["AD"].std()
    '''

    # -------- Logistic (Accuracy) --------
    try:
        mdl_acc = sm.Logit.from_formula(
            "Accuracy ~ distOrigSelf_c * distOrigLand_c",
            data=data
        ).fit(disp=False)

        p = mdl_acc.params
        results_acc.append({
            "PROLIFIC_PID": pid,
            "beta_self":        p.get("distOrigSelf_c", np.nan),
            "beta_land":        p.get("distOrigLand_c", np.nan),
            "beta_interaction": p.get("distOrigSelf_c:distOrigLand_c", np.nan)
        })
    except Exception as e:
        results_acc.append({
            "PROLIFIC_PID": pid,
            "beta_self":        np.nan,
            "beta_land":        np.nan,
            "beta_interaction": np.nan,
            "error_acc":        str(e)
        })

    # -------- RT -------
    # Avoid dot in column name for formula
    if 'Response.rt' in data.columns:
        data = data.rename(columns={'Response.rt': 'Response_RT'})

    try:
        mdl_rt = sm.OLS.from_formula(
            "Response_RT ~ distOrigSelf_c * distOrigLand_c",
            data=data
        ).fit()

        p = mdl_rt.params
        results_rt.append({
            "PROLIFIC_PID": pid,
            "beta_self":        p.get("distOrigSelf_c", np.nan),
            "beta_land":        p.get("distOrigLand_c", np.nan),
            "beta_interaction": p.get("distOrigSelf_c:distOrigLand_c", np.nan)
        })
    except Exception as e:
        results_rt.append({
            "PROLIFIC_PID": pid,
            "beta_self":        np.nan,
            "beta_land":        np.nan,
            "beta_interaction": np.nan,
            "error_rt":         str(e)
        })

# ---- Build the two DataFrames
df_betas_acc = pd.DataFrame(results_acc)
df_betas_rt  = pd.DataFrame(results_rt)


for name, col in [("ACC interaction","beta_interaction"),
                  ("ACC self","beta_self"),
                  ("ACC land","beta_land")]:
    vals = df_betas_acc[col].dropna()
    t, p = stats.ttest_1samp(vals, 0)
    print(f"{name}: t={t:.3f}, p={p:.4g}")
'''
from scipy.stats import wilcoxon
wilcoxon(df_betas_acc["beta_land"])
wilcoxon(df_betas_rt["beta_land"])
wilcoxon(df_betas_acc["beta_interaction"])
wilcoxon(df_betas_rt["beta_interaction"])
wilcoxon(df_betas_acc["beta_self"])
wilcoxon(df_betas_rt["beta_self"])
'''
for name, col in [("RT interaction","beta_interaction"),
                  ("RT self","beta_self"),
                  ("RT land","beta_land")]:
    vals = df_betas_rt[col].dropna()
    t, p = stats.ttest_1samp(vals, 0)
    print(f"{name}: t={t:.3f}, p={p:.4g}")


import matplotlib.pyplot as plt
import seaborn as sns
## PLOT COEFFICIENTS ACCURY MODELS
df_longAcc = df_betas_acc.melt(
    id_vars="PROLIFIC_PID",
    value_vars=['beta_self','beta_land','beta_interaction'],
    var_name="Predictors",
    value_name="Coefficients"
)

fig=plt.figure(1, figsize=(4,5))
sns.barplot(x='Predictors', y='Coefficients', data=df_longAcc ,color='lightgrey')
sns.stripplot(x='Predictors', y='Coefficients', data=df_longAcc,color='black', dodge=True,alpha=0.1, size=5,marker='o')
plt.ylabel('Coefficients', color='black',size=16, fontweight='bold')
plt.xlabel('Predictors', color='black',size=15, fontweight='bold')
plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Self", "Landmark", "Interaction"],       # but show these texts instead
    rotation=0
)
plt.axhline(y=0, color="black", linewidth=0.9,linestyle='--',)
plt.show()


## PLOT COEFFICIENTS REACTION TIME  MODELS
df_longRT = df_betas_rt.melt(
    id_vars="PROLIFIC_PID",
    value_vars=['beta_self','beta_land','beta_interaction'],
    var_name="Predictors",
    value_name="Coefficients"
)

fig=plt.figure(1, figsize=(4,5))
sns.barplot(x='Predictors', y='Coefficients', data=df_longRT ,color='lightgrey')
sns.stripplot(x='Predictors', y='Coefficients', data=df_longRT,color='black', dodge=True,alpha=0.1, size=5,marker='o')
plt.ylabel('Coefficients', color='black',size=16, fontweight='bold')
plt.xlabel('Predictors', color='black',size=15, fontweight='bold')
plt.xticks(
    ticks=plt.xticks()[0],               # keep the same positions
    labels=["Self", "Landmark", "Interaction"],       # but show these texts instead
    rotation=0
)
plt.axhline(y=0, color="black", linewidth=0.9,linestyle='--',)
plt.show()

'''
df_betas_rt = df_betas_rt[df_betas_rt.PROLIFIC_PID != '5a993b151eda410001365a23']
df_betas_acc = df_betas_acc[df_betas_acc.PROLIFIC_PID != '5a993b151eda410001365a23']
'''
## Correlation with 4MT performance
path_data= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"

## Database
slopes= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results\anchoring_biasSpatialSocialslopes.csv") # Save results

df4MT= pd.read_csv(path_data + "\\processed\\AlloTask_SpatialScore.csv")
df4MTmean= df4MT.groupby('PROLIFIC_PID').mean().reset_index()
df_harm= pd.read_csv(paths_results + "\\03AlloSpatialTask\\HarmonicRegression.csv")
egoMean=ego.groupby('PROLIFIC_PID').mean().reset_index()
path_modelsR= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"
df4MTLinear= pd.read_csv(path_modelsR + "\\03AlloSpatialTask\\LinearRegressionAngDisp_4MT.csv")


# Correlations model ACC
Correlations_across_parameters(df4MTmean, df_betas_acc, 'key_resp_3.corr','beta_interaction') 
Correlations_across_parameters(df4MTLinear, df_betas_acc, 'slopesRT_4MT','beta_interaction')
Correlations_across_parameters(df4MTLinear, df_betas_acc, 'slope_4MT','beta_interaction')

# Correlations model rt
Correlations_across_parameters(df4MTmean, df_betas_rt, 'key_resp_3.corr','beta_interaction') 
Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slopesRT_4MT','beta_interaction')
Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slope_4MT','beta_interaction')

# Correlations model rt
Correlations_across_parameters(df_harm, df_betas_rt, 'amplitude','beta_interaction') 
Correlations_across_parameters(df_harm, df_betas_rt, 'amplitude','beta_land') 
Correlations_across_parameters(df_harm, df_betas_rt, 'amplitude','beta_self') 



Correlations_across_parameters(df_harm, df_betas_rt, 'amplitude','beta_interaction') 
Correlations_across_parameters(df_harm, df_betas_rt, 'amplitude','beta_land') 
Correlations_across_parameters(df4MTLinear, df_harm, 'slope_4MT','amplitude') 
Correlations_across_parameters(df4MTLinear, df_harm, 'slopesRT_4MT','amplitude')


Correlations_across_parameters(df_harm, df_betas_acc, 'R2','beta_interaction')

Correlations_across_parameters(df4MTLinear, df_betas_rt, 'slope_4MT','beta_interaction')

egoMean_self=ego.groupby(['PROLIFIC_PID','SelfvsLandmark_proximity']).mean().reset_index()
Correlations_across_parameters(df_harm, a, 'amplitude','Response.rt')


'''
## PCA Y CLUSTER
dfs = [slopes, df4MTLinear,df_betas_acc, df_betas_rt,df4MTmean, df_harm,egoMean]

from functools import reduce
df_all = reduce(lambda left, right: pd.merge(left, right, on="PROLIFIC_PID", how="inner"), dfs) # x_accuracy y_RT


from sklearn.preprocessing import StandardScaler


cols = ['slopesRT_4MT','slope_4MT', 'beta_self_x', 'beta_land_x', 
        'beta_self_y', 'beta_land_y','slope_social','key_resp_3.corr','key_resp_3.rt','amplitude']  

X = df_all[cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = {}
for k in range(2,7):
    km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    scores[k] = silhouette_score(X_scaled, km.labels_)

k = max(scores, key=scores.get)
kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
df_all["cluster"] = kmeans.labels_



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pc = pca.fit_transform(X_scaled)

plt.scatter(pc[:,0], pc[:,1], c=df_all["cluster"], cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Participant clusters")
plt.show()
loadings = pd.DataFrame(pca.components_.T,index=cols,columns=['PC1','PC2'])

'''







