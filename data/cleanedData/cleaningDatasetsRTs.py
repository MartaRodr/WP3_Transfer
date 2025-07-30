# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:02:25 2025

@author: aramendi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import spearmanr, pearsonr



output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data"

################################################################################################
dfAnchoring = pd.read_csv(output_path+"\\processed\\egosocialtask.csv")
dfegoSpatial= pd.read_csv(output_path+"\\processed\\egospatialtask.csv")
################################################################################################
############################## CLEANING DATABASE ######################################################
### Cleaning database SOCIAL
### Step 1: transformation RT to log
dfAnchoring['RTlog'] = np.log(dfAnchoring['RTothers'])
### Step 2: ELiminate outlier
out_sd_lo= dfAnchoring['RTlog'].mean() - (2.5 *  (dfAnchoring['RTlog'].std()))
out_sd_hi= dfAnchoring['RTlog'].mean()  + (2.5 *  (dfAnchoring['RTlog'].std()))
dfAnchoring= dfAnchoring.loc[(dfAnchoring.RTlog> out_sd_lo) & (dfAnchoring.RTlog< out_sd_hi)]
dfAnchoring['RTlog_mean'] = dfAnchoring.groupby('PROLIFIC_PID')['RTlog'].transform('mean')
dfAnchoring['RTlog_cwc']= dfAnchoring['RTlog'] - dfAnchoring['RTlog_mean']

dfAnchoring.to_csv(output_path +"\\cleanedData\\egosocialCleanedRT.csv",index=False)

### CLENING DATABASE  EGO-SPATIAL
dfegoSpatial['RTlog']= np.log(dfegoSpatial['Response.rt'])
### Step 2: ELiminate outlier
out_sd_lo_pre= dfegoSpatial['RTlog'].mean() - (2.5 *  (dfegoSpatial['RTlog'].std()))
out_sd_hi_pre= dfegoSpatial['RTlog'].mean()  + (2.5 *  (dfegoSpatial['RTlog'].std()))
dfegoSpatial= dfegoSpatial.loc[(dfegoSpatial.RTlog> out_sd_lo_pre) & (dfegoSpatial.RTlog< out_sd_hi_pre)]
dfegoSpatial['RTlog_mean'] = dfegoSpatial.groupby('PROLIFIC_PID')['RTlog'].transform('mean')
dfegoSpatial['RTlog_cwc']= dfegoSpatial['RTlog'] - dfegoSpatial['RTlog_mean']
dfegoSpatial.to_csv(output_path +"\\cleanedData\\egospatialCleanedRT.csv",index=False)
################################################################################################