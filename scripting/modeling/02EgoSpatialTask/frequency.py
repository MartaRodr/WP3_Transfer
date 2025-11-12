# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 13:10:14 2025

@author: aramendi
"""

'''Frecuencias '''

import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Pilot\Datasets\egospatial.csv")
Ntotal= 64

paths_cleanedData= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"

df= pd.read_csv(paths_cleanedData + "\\egospatialCleanedRT.csv")

import numpy as np
df['dist_selected'] = np.nan  # creamos la columna vacía
df.loc[df['Accuracy'] == 1, 'dist_selected'] = df.loc[df['Accuracy'] == 1, 'distCorrSelf']
df.loc[df['Accuracy'] == 0, 'dist_selected'] = df.loc[df['Accuracy'] == 0, 'distIncorrSelf']

df['proximity_selected'] = np.where(
    df['Accuracy']==1,
    np.where(df['distCorrSelf'] < df['distIncorrSelf'], 'Close','Far'),
    np.where(df['distIncorrSelf'] < df['distCorrSelf'], 'Close','Far')
)




df_size= df.groupby(['PROLIFIC_PID','Self_proximity','Accuracy']).size().reset_index()
df_size['A']=0.50 ## 
df_size= df_size.rename(columns={0:'N'})
df_size= df_size.loc[df_size['Accuracy']==1]
df_size['C']= df_size['N']/32
df_size['Differences']=  df_size['C'] - df_size['A']

import seaborn as sns
viridis_colors = sns.color_palette("viridis", 2)

# Asigna manualmente
custom_palette = {
    'self_far': viridis_colors[0],     # el primer color para 'self_far'
    'self_close': viridis_colors[1] }  # el segundo color para 'self_close'}

sns.barplot(x='Self_proximity', y='Differences', data=df_size,palette= custom_palette)
sns.stripplot(x='Self_proximity', y='Differences', data=df_size,color= 'black', size=3)
plt.axhline(0.00, linestyle='--', color='red', linewidth=1)
plt.axhline(-0.50, linestyle='--', color='black', linewidth=1)
#plt.axhline(0.50, linestyle='--', color='black', linewidth=1)
plt.show()


from scipy.stats import ttest_rel
stat, p_val = ttest_rel(df_size['Differences'].loc[(df_size['Self_proximity']=='self_close')],
                        df_size['Differences'].loc[(df_size['Self_proximity']=='self_far')])
print("Ttest, p-value:", stat, p_val)




# 1. Crear los cuartiles
df['bins_SelfObjectCorrectDistance'] = pd.cut(
    df['distCorrSelf'],
    bins=[
        df['distCorrSelf'].min(),
        df['distCorrSelf'].quantile(0.25),
        df['distCorrSelf'].quantile(0.5),
        df['distCorrSelf'].quantile(0.75),
        df['distCorrSelf'].max()
    ],
    labels=['Q1', 'Q2', 'Q3', 'Q4'],
    include_lowest=True
)

# 1. Crear los cuartiles
df['bin_AD'] = pd.cut(
    df['distCorrSelf'],
    bins=[
        df['AD'].min(),
        df['AD'].quantile(0.25),
        df['AD'].quantile(0.5),
        df['AD'].quantile(0.75),
        df['AD'].max()
    ],
    labels=['Q1', 'Q2', 'Q3', 'Q4'],
    include_lowest=True
)


df_total = df.groupby(['PROLIFIC_PID', 'Self_proximity', 'bin_AD']).size().reset_index(name='N_total')

# 3. Contar número de aciertos por grupo (accuracy = 1)
df_correct = df[df['Accuracy'] == 1].groupby(['PROLIFIC_PID', 'Self_proximity', 'bin_AD']).size().reset_index(name='N_correct')

# 4. Unir ambos para calcular proporción
df_merged = pd.merge(df_total, df_correct, on=['PROLIFIC_PID', 'Self_proximity', 'bin_AD'], how='left')
df_merged['N_correct'] = df_merged['N_correct'].fillna(0)  # En caso de que haya 0 aciertos

df_merged['C'] = df_merged['N_correct'] /df_merged['N_total'] 
df_merged['A']= 0.50 ##
df_merged['Differences']= df_merged['C'] - df_merged['A']



plt.figure(figsize=(6,6))

sns.lineplot(x='bin_AD', y='Differences', data=df_merged, palette=custom_palette)
plt.axhline(0.00, linestyle='--', color='red', linewidth=1)
plt.axhline(-0.50, linestyle='--', color='black', linewidth=1)
#plt.axhline(0.50, linestyle='--', color='black', linewidth=1)
plt.xlabel("Self–CorrectObject distance")
plt.ylabel("Differences")
plt.show()



df['distCorrSelf_bins'] = pd.cut(
    df['distCorrSelf'],
    bins=8,  # o usa tus cuartiles si prefieres
    labels=['Q1', 'Q2', 'Q3', 'Q4','Q5','Q6','Q7','Q8']
)

df_total = df.groupby(['PROLIFIC_PID', 'Self_proximity', 'distCorrSelf_bins']).size().reset_index(name='N_total')

# 3. Contar número de aciertos por grupo (accuracy = 1)
df_correct = df[df['Accuracy'] == 1].groupby(['PROLIFIC_PID', 'Self_proximity', 'distCorrSelf_bins']).size().reset_index(name='N_correct')

# 4. Unir ambos para calcular proporción
df_merged = pd.merge(df_total, df_correct, on=['PROLIFIC_PID', 'Self_proximity', 'distCorrSelf_bins'], how='left')
df_merged['N_correct'] = df_merged['N_correct'].fillna(0)  # En caso de que haya 0 aciertos

df_merged['C'] = df_merged['N_correct'] / df_merged['N_total'] 
df_merged['A']=0.50 ##
df_merged['Differences']= df_merged['C'] - df_merged['A']

sns.lineplot(x='distCorrSelf_bins', y='Differences', hue='Self_proximity', data=df_merged, palette=custom_palette, ci=68)
plt.axhline(0.00, linestyle='--', color='red', linewidth=1)
plt.axhline(-0.50, linestyle='--', color='black', linewidth=1)
plt.axhline(0.50, linestyle='--', color='black', linewidth=1)
plt.xlabel("Self–CorrectObject distance")
plt.ylabel("Differences")
plt.show()





'''

import numpy as np

# 1) mark “chose_closer” (1 if they picked the option with the smaller distance, 0 otherwise)
df['dist_selected'] = np.where(
    df['Accuracy']==1,
    df['distCorrSelf'],
    df['distIncorrSelf']
)
df['chose_closer'] = (
    df['dist_selected']
    == df[['distCorrSelf','distIncorrSelf']].min(axis=1)
).astype(int)


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df_mean= df.groupby(['PROLIFIC_PID','Self_proximity','distCorrSelf']).mean().reset_index()
plt.figure(figsize=(6,5))
sns.lineplot(
    data=df_mean,
    
    x='distCorrSelf',   # your x variable
    y='chose_closer',                   
    hue='Self_proximity',              
    ci=68,                              
    marker="o",                         
    palette=custom_palette                      
)
plt.axhline(.5, ls="--", c="red")      # chance level
plt.ylim(-0.5,1.5)
plt.ylabel("P(choose closer)")
plt.xlabel("Self–CorrectObject distance")
plt.title("Bias to choose the closer option")
plt.legend(title="Self proximity ")
plt.tight_layout()
plt.show()


# 1a. Create a “chose_farther” column by inverting chose_closer
df['chose_farther'] = 1 - df['chose_closer']

# 1b. Then plot exactly the same way, but with y='chose_farther'
plt.figure(figsize=(6,5))
sns.lineplot(
    data=df,
    x='distCorrSelf',   
    y='chose_farther',                   
    hue='Self_proximity',              
    ci=68,                             
    marker="o",                         
    n_boot=2000, 
    palette=custom_palette
)
plt.axhline(.5, ls="--", c="red")
plt.ylim(-0.5, 1.5)
plt.ylabel("P(choose farther)")
plt.xlabel("Self–CorrectObject distance")
plt.title("Bias to choose the farther option")
plt.legend(title="Self proximity")
plt.tight_layout()
plt.show()


# Crear bins para distCorrSelf (ejemplo con 4 bins iguales)
df['distCorrSelf_bins'] = pd.cut(
    df['distCorrSelf'],
    bins=8,  # o usa tus cuartiles si prefieres
    labels=['Q1', 'Q2', 'Q3', 'Q4','Q5','Q6','Q7','Q8']
)

sns.lineplot(
    data=df,
    x='distCorrSelf_bins',
    y='chose_closer',
    hue='Self_proximity',
    ci=68,
    marker='o',
    n_boot=2000,
    palette=custom_palette
)
plt.axhline(0.5, ls='--', c='red')  # chance level
plt.ylim(-0.1, 1.1)
plt.xlabel("Self–CorrectObject distance")
plt.ylabel("P(choose closer)")
plt.title("Bias to choose the closer option")
plt.legend(title="Self proximity")
plt.tight_layout()
plt.show()


sns.lineplot(
    data=df,
    x='distCorrSelf_bins',      # your binned distance variable
    y='chose_farther',          # now plotting the probability of choosing the farther option
    hue='Self_proximity',
    ci=68,
    marker='o',
    n_boot=2000,
    palette=custom_palette
)



plt.axhline(0.5, ls='--', c='red')  # chance level
plt.ylim(-0.1, 1.1)
plt.xlabel("Self–CorrectObject distance")
plt.ylabel("P(choose farther)")
plt.title("Bias to choose the farther option")
plt.legend(title="Self proximity")
plt.tight_layout()
plt.show()

df_closer= df.loc[df['Self_proximity']=='self_close']
dfclosermean= df_closer.groupby(['PROLIFIC_PID','distCorrSelf_bins'])['chose_closer'].mean().reset_index()


df_far= df.loc[df['Self_proximity']=='self_far']
dffarmean= df_far.groupby(['PROLIFIC_PID','distCorrSelf_bins'])['chose_farther'].mean().reset_index()

concat= pd.merge(dfclosermean, dffarmean, on=('PROLIFIC_PID','distCorrSelf_bins'))
concat['Differences']= concat['chose_closer'] - concat['chose_farther']

sns.lineplot(
    data=concat,
    x='distCorrSelf_bins',color='black',
    y='Differences',     
    ci=68,
    marker='o',
)
plt.ylabel("P(choose closer) - P(choose farther)")
plt.axhline(.0, ls="--", c="red")

'''


