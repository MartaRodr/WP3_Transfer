# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 09:42:40 2025

@author: aramendi
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np


#########################################################################################################################
df4MT =  pd.read_csv(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\Analysis\Experiment\Datasets\allotask.csv")
anova = df4MT.groupby(['PROLIFIC_PID', 'angularDisparity']).mean().reset_index()
#########################################################################################################################

fig= plt.figure(1, figsize=(8,8))
df4MT_mean= df4MT.groupby(['PROLIFIC_PID'])[['key_resp_3.corr','key_resp_3.rt']].mean().reset_index()

fig= plt.figure(1, figsize=(4,6))
sns.set_theme(style="whitegrid")
plt.subplot(2,2,1)
sns.set_theme(style="white", palette=None)
sns.barplot(y=df4MT_mean['key_resp_3.corr']*100, data=df4MT_mean,color='lightgrey')     
sns.stripplot(y=df4MT_mean['key_resp_3.corr']*100, data=df4MT_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.axhline(25, linestyle='--', color='red', linewidth=1)
plt.ylabel('Accuracy (%Correct)', color='black',size=16, fontweight='bold')

plt.subplot(2,2,2)
sns.barplot(y=df4MT_mean['key_resp_3.rt']*1000, data=df4MT_mean,color='lightgrey')
sns.stripplot(y=df4MT_mean['key_resp_3.rt']*1000, data=df4MT_mean,color='black', dodge=True,alpha=0.8, size=3)
plt.ylabel('RT (ms)', color='black',size=16, fontweight='bold')
fig.subplots_adjust(wspace=0.5, hspace=0.25)
plt.show()

plt.figure(figsize=(10,10))
ax1=sns.barplot(data=anova, x="angularDisparity", y=anova["key_resp_3.rt"]*1000,ci=68, color='grey')
ax1.set_ylabel('RT(ms)', fontweight='bold', color='black', size= 15)
ax1.set_xlabel('Angular Disparity', fontweight='bold', color='black', size= 15)
ax1.tick_params(axis='y', labelsize=18)
plt.show()

plt.figure(figsize=(10,10))
ax1=sns.barplot(data=anova, x="angularDisparity", y=anova["key_resp_3.corr"]*100,ci=68, color='grey')
ax1.set_ylabel('Accuracy (%Correct)', fontweight='bold', color='black', size= 15)
ax1.set_xlabel('Angular Disparity', fontweight='bold', color='black', size= 15)
ax1.tick_params(axis='y', labelsize=18)
plt.show()

############################################################################################
order = [1,2,3,4,5,6,7,8]
####################### BAR PLOT WITH MEAN BEHAVO OF PARTICPANTS  #######################
anova = anova.copy()
anova['Accuracy'] = anova['key_resp_3.corr'] * 100

deg_map = {1:40, 2:80, 3:120, 4:160, 5:200, 6:240, 7:280, 8:320}
anova = anova.copy()
anova['deg360'] = anova['angularDisparity'].map(deg_map)

order = [1,2,3,4,5,6,7,8]
pastel_colors = sns.color_palette("Blues", 4)

group_keys = [(8, 1), (7, 2), (6, 3), (5, 4)]
group_palette = {}
for idx, (high, low) in enumerate(group_keys):
    group_palette[high] = pastel_colors[idx]
    group_palette[low]  = pastel_colors[idx]

# 5) Construye la lista de colores en el orden deseado
palette = [group_palette[x] for x in order]

# 6) Dibuja el gráfico
plt.figure(figsize=(8, 6))

# Barras con paleta pastel
sns.barplot(
    data=anova,
    x='angularDisparity',
    y='Accuracy',
    order=order,
    ci=None,
    palette=palette
)

sns.pointplot(
    data=anova,
    x='angularDisparity',
    y='Accuracy',
    order=order,
    ci=68,
    color='black',
    linestyles='dotted',
    markers='o',
    capsize=.09,
    errwidth=1,
    dodge=False, legend=False
)


plt.xlabel('Angular Disparity', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (% Correct)', fontsize=16, fontweight='bold')
plt.ylim(0, None)
plt.tight_layout()
plt.show()

###################################################################################################################
# Reaction Time 
anova = anova.copy()
anova['rt_ms'] = anova['key_resp_3.rt'] * 1000

pastel_colors = sns.color_palette("Blues", 4)

group_keys = [(8, 1), (7, 2), (6, 3), (5, 4)]
group_palette = {}
for idx, (high, low) in enumerate(group_keys):
    group_palette[high] = pastel_colors[idx]
    group_palette[low]  = pastel_colors[idx]

# 5) Construye la lista de colores en el orden deseado
palette = [group_palette[x] for x in order]


plt.figure(figsize=(8, 6))

# Barras con paleta pastel
sns.barplot(
    data=anova,
    x='angularDisparity',
    y='rt_ms',
    order=order,
    ci=None,
    palette=palette
)

sns.pointplot(
    data=anova,
    x='angularDisparity',
    y='rt_ms',
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
plt.ylabel('RT (ms)', fontsize=16, fontweight='bold')
plt.ylim(0, None)
plt.tight_layout()
plt.show()
###################################################################################################################
# GRAPH PLOT
##################################################################################################################
deg_map = {1:40, 2:80, 3:120, 4:160, 5:200, 6:240, 7:280, 8:320}
anova = anova.copy()
anova['deg360'] = anova['angularDisparity'].map(deg_map)

# 2) Media de Accuracy por ángulo
order360 = [40, 80, 120, 160, 200, 240, 280, 320]
means = anova.groupby('deg360')['Accuracy'] \
             .mean().reindex(order360).values

# 3) Generar paleta de 4 colores con Blues
blues_colors = sns.color_palette("Blues", 4)

# 4) Mapear cada ángulo a su índice de grupo
group_map = {
    40: 0, 320: 0,   # par ±40°
    80: 1, 280: 1,   # par ±80°
    120:2, 240:2,    # par ±120°
    160:3, 200:3     # par ±160°
}
# 5) Construir lista de colores en orden de barras
bar_palette = [blues_colors[group_map[ang]] for ang in order360]

# 6) Conversión a radianes y ancho de barra
theta = np.deg2rad(order360)
bar_width = np.deg2rad(45) * 0.8  # 80% de un sector de 45°

# 7) Crear figura polar
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

# 8) Dibujar las barras con la paleta agrupada
ax.bar(theta, means,
       width=bar_width,
       bottom=0,
       color=bar_palette,
       edgecolor='black',
       linewidth=1,
       alpha=0.8,
       align='center')

# 9) Línea punteada de medias
ax.plot(theta, means, 'k--o', lw=1.5, markersize=6)
# 11) Círculo rojo punteado en 50%
circle_theta = np.linspace(0, 2*np.pi, 200)
ax.plot(circle_theta, [25]*200, linestyle=':', color='red', linewidth=1.5)

# 12) Cuadrícula radial sin etiquetas numéricas
ax.set_yticks([30, 60, 90, 100])
ax.set_yticklabels([])
ax.grid(True, linestyle='--', alpha=0.4)

# 13) Orientación y etiquetas angulares
ax.set_theta_zero_location('S')
ax.set_theta_direction(1)
label_angles = ['40°','80°','120°','160°','-160°','-120°','-80°','-40°']
ax.set_xticks(theta)
ax.set_xticklabels(label_angles, fontsize=20)
ax.tick_params(axis='x', which='major', pad=10)

plt.tight_layout()
plt.show()
#########################################################################################################################
# PLOTTING PRESPECTIVE AGULAR DISPARITY 
#########################################################################################################################
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import seaborn as sns

# Offsets from 0° at the bottom (in degrees)
offsets = [0, 40, 80, 120, 160, 200, 240, 280, 320]

# Generate a Blues palette for the non-zero offsets
blues_colors = sns.color_palette("Blues", len(offsets) - 1)

# Parameters to control tick length and thickness
tick_length = 0.2   # proportion of circle radius
tick_width = 6      # line thickness for ticks
radius =    2       # radio del círculo

fig, ax = plt.subplots(figsize=(6, 6))

# 1) Draw the disk with outline
circle = Circle((0, 0), radius, facecolor='lightgray', edgecolor='black', linewidth=2)
ax.add_patch(circle)

# 2) Draw and label ticks
for i, offset in enumerate(offsets):
    theta = np.deg2rad(270 + offset)
    # calculamos inner_r = punto de inicio común
    inner_r = radius * (1 - tick_length)
    x0, y0 = inner_r * np.cos(theta), inner_r * np.sin(theta)
    x1, y1 = radius * np.cos(theta),  radius * np.sin(theta)

    if offset == 0:
        # ahora es un tick corto igual que los demás
        ax.plot([x0, x1], [y0, y1],
                color='red', linewidth=tick_width, solid_capstyle='butt')
        # etiqueta a 0° justo por debajo
        ax.text(0, -radius * 1.15, '0°',
                ha='center', va='top',
                fontsize=20, fontweight='bold', color='red')
    else:
        # ticks coloreados con Blues
        color = blues_colors[i - 1]
        ax.plot([x0, x1], [y0, y1],
                color=color, linewidth=tick_width, solid_capstyle='butt')
        # label
        lbl_r = radius * 1.25
        xl, yl = lbl_r * np.cos(theta), lbl_r * np.sin(theta)
        ax.text(xl, yl, f"{offset}°",
                ha='center', va='center',
                fontsize=20, color='black')

# Final styling
ax.set_aspect('equal')
lim = radius * 1.3
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axis('off')

plt.show()

