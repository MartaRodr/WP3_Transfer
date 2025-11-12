# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:53:21 2025

@author: aramendi
"""

import numpy as np, pandas as pd
import statsmodels.api as sm
path_data= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\results"

df4MT= pd.read_csv(path_data + "\\processed\\AlloTask_SpatialScore.csv")

df4MT["quartileRTs"] = pd.qcut(df4MT["key_resp_3.rt"], q=4, labels=["Q1","Q2","Q3","Q4"])
df4MT_mean= df4MT.groupby(['PROLIFIC_PID','quartileRTs']).mean().reset_index()

import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(
    data=df4MT,
    x="key_resp_3.rt", y="key_resp_3.corr",
    lowess=True, scatter=False
)
plt.title("Accuracy vs Reaction Time")
plt.xlabel("RT")
plt.ylabel("Accuracy")
plt.show()


sns.histplot(
    df4MT, x="key_resp_3.rt", y="key_resp_3.corr",
    bins=10, discrete=(True, False), log_scale=(False, True),
    cbar=True, cbar_kws=dict(shrink=.75),
)

# Define bin edges (for example from 0 to max+1)
max_rt = np.ceil(df4MT["key_resp_3.rt"].max())
bins = np.arange(0, max_rt + 1, 1)

df4MT["RT_bin"] = pd.cut(
    df4MT["key_resp_3.rt"],
    bins=bins,
    right=True,      # include right edge
    include_lowest=False
)

df4MT_mean= df4MT.groupby(['PROLIFIC_PID','angularDisparity']).mean().reset_index()
df4MT_std= df4MT.groupby(['PROLIFIC_PID','angularDisparity']).std().reset_index()

df4MT_merge= pd.merge(df4MT_mean,df4MT_std, on='PROLIFIC_PID' )


sns.barplot(x='RT_bin', y='key_resp_3.corr', data=df4MT_mean)
plt.xticks(rotation=45)