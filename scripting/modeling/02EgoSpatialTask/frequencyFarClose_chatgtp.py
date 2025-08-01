# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:40:45 2025

@author: aramendi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 0) Asume que ya tienes tu DataFrame 'df' con estas columnas:
#    - 'PROLIFIC_PID' (ID de participante)
#    - 'Accuracy'      (1 si eligieron la opción correcta, 0 si no)
#    - 'distCorrSelf'  (distancia de la opción “correcta”)
#    - 'distincorrectSelf' (distancia de la opción “incorrecta”)

paths_cleanedData= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"
paths_results= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\cleanedData"

df= pd.read_csv(paths_cleanedData + "\\egospatialCleanedRT.csv")

# 1) Crear una columna con la distancia de la opción que eligió el participante
df['dist_selected'] = np.where(
    df['Accuracy'] == 1,
    df['distCorrSelf'],
    df['distIncorrSelf']
)

# 2) Marcar si esa distancia corresponde a la opción más cercana
df['chose_closer'] = (
    df['dist_selected']
    == df[['distCorrSelf', 'distIncorrSelf']].min(axis=1)
)

# 3) Calcular la proporción global de elecciones de la más cercana
prop_closer_overall = df['chose_closer'].mean()
prop_farther_overall = 1 - prop_closer_overall
print(f"Escogieron la opción más cercana en {prop_closer_overall:.2%} de los ensayos, "
      f"y la más lejana en {prop_farther_overall:.2%}.")

prop_por_part = (
    df
    .groupby('PROLIFIC_PID')['chose_closer']
    .mean()                      # esto es prop. closer
    .reset_index(name='Close')
)
prop_por_part['Far'] = 1 - prop_por_part['Close']


df_melt = prop_por_part.melt(
    id_vars='PROLIFIC_PID',
    value_vars=['Close', 'Far'],
    var_name='Choice',        # nombre para la columna que indicará closer vs farther
    value_name='Proportion'    # nombre para la columna de valores
)


# 5) Gráficos
sns.set_theme(style="whitegrid")
viridis_colors = sns.color_palette("viridis", 2)
custom_palette = {
    'Far': viridis_colors[0],     # el primer color para 'self_far'
    'Close': viridis_colors[1]    # el segundo color para 'self_close'
}

# 5a) Barra global
plt.figure(figsize=(4, 6))
sns.barplot(
    x='Choice',
    y=df_melt['Proportion']*100,
    palette=custom_palette, data= df_melt
)
sns.stripplot(y=df_melt['Proportion']*100, x='Choice',color='black', dodge=True,alpha=0.8, size=3,data= df_melt)

plt.axhline(50, linestyle='--', color='red', linewidth=1)
plt.ylabel('Proportion (%)')
plt.ylim(0, 100)
plt.show()

from scipy.stats import ttest_rel
stat, p_val = ttest_rel(df_melt['Proportion'].loc[df_melt['Choice']=='Close'],
                        df_melt['Proportion'].loc[df_melt['Choice']=='Far'])
print("Ttest, p-value:", stat, p_val)



# 1) Distancia seleccionada
df['dist_selected'] = np.where(
    df['Accuracy'] == 1,
    df['distCorrSelf'],
    df['distIncorrSelf']
)

# 2) ¿Eligieron la opción más cercana?
df['chose_closer'] = (
    df['dist_selected']
    == df[['distCorrSelf', 'distIncorrSelf']].min(axis=1)
)

# 3) ¿La opción CORRECTA era la más cercana?
df['correct_closer'] = (
    df['distCorrSelf']
    == df[['distCorrSelf', 'distIncorrSelf']].min(axis=1)
)

# 4) Etiquetas para condición y para acierto/fallo
df['Condition'] = np.where(df['correct_closer'], 'CorrectClose', 'CorrectFar')
df['Outcome']   = np.where(
    df['chose_closer'] == df['correct_closer'],
    'CorrectChoice',
    'IncorrectChoice'
)

prop_pp = (
    df
    .groupby(['PROLIFIC_PID','Condition','Outcome'])
    .size()
    .reset_index(name='count')
)

tot_pp = (
    df
    .groupby(['PROLIFIC_PID','Condition'])
    .size()
    .reset_index(name='total')
)

prop_pp = prop_pp.merge(tot_pp, on=['PROLIFIC_PID','Condition'])
prop_pp['prop'] = prop_pp['count'] / prop_pp['total']


summary = (
    prop_pp
    .groupby(['Condition','Outcome'])['prop']
    .agg(mean_prop='mean', sem_prop='sem')
    .reset_index()
)

# 7) Gráfico de barras con barras de error y puntos individuales
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8,6))
green_red = ["green", "red"]
# barras
ax = sns.barplot(
    data=summary,
    x='Condition',
    y=summary['mean_prop']*100,
    hue='Outcome',
    palette=green_red,
    capsize=0.1,
    errwidth=1,

)

# puntos individuales (jitterados un poco)
sns.stripplot(
    data=prop_pp,
    x='Condition',
    y=prop_pp['prop']*100,
    hue='Outcome',
    dodge=True,
    marker='o',
    alpha=0.5,
    color='black',
    ax=ax,

)

# línea de referencia al 50%
ax.axhline(50, linestyle='--', color='red', linewidth=1)

# ajustes finales
ax.set_ylim(0,100)
ax.set_ylabel('Propoortion(%)')
ax.set_xlabel('Self proximity')
plt.legend(title='Self proximity', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)

plt.tight_layout()
plt.show()
