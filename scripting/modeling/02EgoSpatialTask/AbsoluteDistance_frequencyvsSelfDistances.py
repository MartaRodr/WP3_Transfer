# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:57:59 2025

@author: aramendi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Pilot\Datasets\egospatial.csv")
Ntotal= 64

paths_cleanedData= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"

dfegoSpatial= pd.read_csv(paths_cleanedData + "\\egospatialCleanedRT.csv")

''' Frecuency to select the self option when increase in difficulty '''

dfegoSpatial['dist_selected'] = np.nan  # creamos la columna vacía
dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 1, 'dist_selected'] = dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 1, 'distCorrSelf']
dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 0, 'dist_selected'] = dfegoSpatial.loc[dfegoSpatial['Accuracy'] == 0, 'distIncorrSelf']

dfegoSpatial['proximity_selected'] = np.where(
    dfegoSpatial['Accuracy']==1,
    np.where(dfegoSpatial['distCorrSelf'] < dfegoSpatial['distIncorrSelf'], 1,0),
    np.where(dfegoSpatial['distIncorrSelf'] < dfegoSpatial['distCorrSelf'], 1,0)
)

dfegoSpatial["AD_z"] = dfegoSpatial.groupby("PROLIFIC_PID")["AD"].transform(lambda x: (x - x.mean()) / x.std())


import numpy as np
import pandas as pd

bin_width = 0.5
max_AD = np.ceil(dfegoSpatial["AD"].max())
bins = np.arange(0, max_AD + bin_width, bin_width)

dfegoSpatial["AD_bin"] = pd.cut(dfegoSpatial["AD"], bins=bins, right=True)

dfegoSpatial['AD_bin_q'] = pd.qcut(dfegoSpatial['AD'], q=5)

df_AD = dfegoSpatial.groupby(['PROLIFIC_PID',"AD_bin_q"])["proximity_selected"].mean().reset_index()
df_AD.rename(columns={"proximity_selected": "prop_closer"}, inplace=True)

df_AD["AD_center"] = df_AD["AD_bin_q"].apply(lambda x: (x.left + x.right)/2)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(
    data=df_AD.sort_values('AD_bin_q'),
    x='AD_center', y='prop_closer',
    estimator='mean', ci='sd', marker='o'
)
ax.axhline(0.5, linestyle='--', alpha=0.7)
ax.set(ylabel='Frequency choosing closer option',
       xlabel='Absolute Distance',
       title='Egocentric spatial bias vs Absolute Distance',
       ylim=(0, 1))
plt.show()

dfegoSpatial["correct_choice_is_closer"] = (
    dfegoSpatial["distCorrSelf"] < dfegoSpatial["distIncorrSelf"]
)

n_correctByAD= dfegoSpatial.groupby(['AD_bin', 'correct_choice_is_closer']).size()