# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 09:56:36 2025

@author: aramendi
"""

### Social anchor phase

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.gridspec import GridSpec

output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"

dfAnchoring = pd.read_csv(output_path+"\\processed\\egosocialtask.csv")
###############################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

# Valores mínimo y máximo
vmin = dfAnchoring['Vself'].min()
vmax = dfAnchoring['Vself'].max()


bins = np.arange(vmin, vmax+2) - 0.5
bin_centers = np.arange(vmin, vmax+1)
widths = np.ones_like(bin_centers)


participants = dfAnchoring['PROLIFIC_PID'].unique()
counts = []
for p in participants:
    vals = dfAnchoring.loc[dfAnchoring['PROLIFIC_PID']==p, 'Vself']
    c, _ = np.histogram(vals, bins=bins)
    counts.append(c)
counts = np.vstack(counts)
mean_c = counts.mean(axis=0)
se_c   = counts.std(axis=0, ddof=1) / np.sqrt(len(participants))

plt.figure(figsize=(6,6))
plt.bar(
    bin_centers,
    mean_c,
    width=widths,
    yerr=se_c,
    capsize=3,
    color='lightgrey',    # relleno gris
    edgecolor='black',    # borde blanco
    linewidth=1         # grosor del borde
)
plt.xticks(bin_centers)
plt.xlabel('Self rating ')
plt.ylabel('Count')
plt.title('Self rating distribution')
plt.tight_layout()
plt.show()
###############################################################################################################################################
###############################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

#dfOther1= dfAnchoring.loc[dfAnchoring['Individual']=='P1']

# Valores mínimo y máximo
vmin = dfAnchoring['RD'].min()
vmax = dfAnchoring['RD'].max()


bins = np.arange(vmin, vmax+2) - 0.5
bin_centers = np.arange(vmin, vmax+1)
widths = np.ones_like(bin_centers)


participants = dfAnchoring['PROLIFIC_PID'].unique()
counts = []
for p in participants:
    vals = dfAnchoring.loc[dfAnchoring['PROLIFIC_PID']==p, 'RD']
    c, _ = np.histogram(vals, bins=bins)
    counts.append(c)
counts = np.vstack(counts)
mean_c = counts.mean(axis=0)
se_c   = counts.std(axis=0, ddof=1) / np.sqrt(len(participants))

plt.figure(figsize=(6,6))
plt.bar(
    bin_centers,
    mean_c,
    width=widths,
    yerr=se_c,
    capsize=3,
    color='lightgrey',    # relleno gris
    edgecolor='black',    # borde blanco
    linewidth=1         # grosor del borde
)
plt.xticks(bin_centers)
plt.xlabel('RD')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
###############################################################################################################################################
dfAnchoring_mean= dfAnchoring.groupby(['PROLIFIC_PID','Individual'])['RD'].mean().reset_index()
sns.barplot(x='Individual', y='RD', data= dfAnchoring_mean, color='grey',ci=None)
sns.stripplot(x='Individual', y='RD', data=dfAnchoring_mean,color='black', dodge=True,alpha=0.8, size=3)
sns.lineplot(x='Individual',
             y='RD',
             data=dfAnchoring_mean,
             units='PROLIFIC_PID',
             estimator=None,
             color='black',
             alpha=0.3,
             legend=False)
plt.xlabel('Individual')
plt.ylabel('Rating discrepancy')

############################################# ANCHOR GRAPHS ###############################################################################

###Step 1: transformation RT to log
dfAnchoring['RTlog']= np.log(dfAnchoring['RTothers'])

### ELiminate outlier
out_sd_lo= dfAnchoring['RTlog'].mean() - (2.5*  (dfAnchoring['RTlog'].std()))
out_sd_hi= dfAnchoring['RTlog'].mean()  + (2.5*  (dfAnchoring['RTlog'].std()))
dfAnchoring= dfAnchoring.loc[(dfAnchoring.RTlog> out_sd_lo) & (dfAnchoring.RTlog< out_sd_hi)]
dfAnchoring['RTlog_mean'] = dfAnchoring.groupby('PROLIFIC_PID')['RTlog'].transform('mean')
dfAnchoring['std'] = dfAnchoring.groupby('PROLIFIC_PID')['RTlog'].transform('std')
dfAnchoring['RTlog_cwc']= (dfAnchoring['RTlog'] - dfAnchoring['RTlog_mean'])
dfAnchoring['RTlog_std']= (dfAnchoring['RTlog'] - dfAnchoring['RTlog_mean'])/(dfAnchoring['std'] )


fig = plt.figure(figsize=(5, 5))
gs = GridSpec(2, 2, figure=fig)
sns.set_theme(style="white", palette=None)
ax1 = fig.add_subplot(gs[0, 0])
jg1 = sns.JointGrid(data=dfAnchoring, x="RTlog_cwc", y="RD")
sns.regplot(data=dfAnchoring, x="RTlog_cwc", y="RD", color="red", scatter=False, ax=jg1.ax_joint)
jg1.ax_marg_x.hist(dfAnchoring["RTlog_cwc"], color="grey")
jg1.ax_marg_y.hist(dfAnchoring["RD"], color="grey", orientation="horizontal", bins=8)
# Set axis labels
jg1.ax_joint.set_ylabel("Rating discrepancy", fontsize=16,fontweight='bold')  # Change "Custom X-axis Label" to your desired label
jg1.ax_joint.set_xlabel("RT (log-transformed)", fontsize=16,fontweight='bold')  # Change "Custom Y-axis Label" to your desired label

jg1.ax_joint.set_xlim(-2, +2)
jg1.ax_joint.set_ylim(0, 5)

plt.savefig(r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\plots\SocialAnchoring.png", dpi=300, bbox_inches='tight')


plt.show()

rt = dfAnchoring.groupby(['PROLIFIC_PID'])[['RTothers','Individual']].agg(['mean','sem','std']).reset_index()

###############################################################################################################################################