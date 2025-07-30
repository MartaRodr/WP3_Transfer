# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:14:36 2025

@author: aramendi
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

def correct(row):
    if row['correctKeyboard']== row['Response.keys']:
        return 1
    else:
        return 0
    
from pandas.errors import EmptyDataError
import csv

output_path= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\processed"
folder= r"C:\Users\aramendi\Desktop\EscritorioMARTA\WP_Transfer\WP3Project\data\rawData\dataOrder1"

# Open participants raw data from Pavlovia
csv_files = glob.glob(os.path.join(folder, "*.csv"))
dfs = []
for path in csv_files:
    try:
        df = pd.read_csv(
          path,
          sep=',',
          engine='python',
          quoting=csv.QUOTE_ALL,on_bad_lines='skip')
    except EmptyDataError as e:
        print(e)
        continue
    dfs.append(df)

    
filtered_dfs = [df for df in dfs if df.shape[0] == (194)] # Size outputExperiment N=194

df_all = pd.concat(filtered_dfs, axis=0, ignore_index=True)
df = df_all[df_all['PROLIFIC_PID'].notna()]
print("The number of participants is :" + str(len(df['PROLIFIC_PID'].unique())))

df['exp_index'] = pd.to_numeric(df['exp_index'], errors='coerce')
df['exp_index'].fillna(method='ffill', inplace=True)


############################## ALLOCENTRIC SPATIAL TASK ##########################################################
df4MT= df[['PROLIFIC_PID','gender','exp_index','key_resp_3.corr','key_resp_3.rt','AlloTrials.thisIndex','env_type',
           'choices','angularDisparity']]

# Corrections RT,trial without response. 
df4MT.loc[(df4MT['key_resp_3.corr'] == 0) & (df4MT['key_resp_3.rt'].isna()), 'key_resp_3.rt'] = 20
df4MT= df4MT.dropna()

# Count NCorrects trials
df4MT['Ncorrect'] = df4MT.groupby('PROLIFIC_PID')['key_resp_3.corr'].transform(lambda x: (x == 1).sum())

mean4MT = df4MT.groupby(['PROLIFIC_PID']).mean().reset_index() # MEAN ACCURACY 
##################################################################################################################             
                                    #--------- BAD PARTICIPANTS---------#

# 1) Identify participants to drop (mean Accuracy < 0.30) 4MTTask
bad_participants = (
    df4MT.groupby('PROLIFIC_PID')['key_resp_3.corr']
         .mean()
         .loc[lambda s: s < 0.30]
         .index
    .tolist()
)

print("The number of participant with 4MT performance lower than 30% : "+  str(len(bad_participants)))

#------------------------------------------------------------------------------------------------------------ 
############################## SOCIAL ANCHOR TASK ################################################################

dfAnchorSelf= df[['PROLIFIC_PID','exp_index','gender','key_resp_4.keys', 'key_resp_4.rt','trialsSelf.thisIndex','escenario']]

# Corrections RT,trial without response. 
dfAnchorSelf= dfAnchorSelf.dropna()

dfAnchorOthers= df[['PROLIFIC_PID','gender','Other','key_resp_5.keys', 'key_resp_5.rt','trialsOthers.thisIndex','escenario']]
dfAnchorOthers=dfAnchorOthers.dropna()

dfAnchoring= pd.merge(dfAnchorOthers, dfAnchorSelf, on=['PROLIFIC_PID','gender','escenario'])

def changeIndividuals(row):
    if row['Other'] in ['Harry', 'Elisabeth', 'Alex']:
        return 'P1'
    elif row['Other'] in ['Jack','Charlotte','Sam']:
        return 'P2'

dfAnchoring['Individual'] = dfAnchoring.apply(changeIndividuals, axis=1)
dfAnchoring= dfAnchoring.rename(columns={'key_resp_4.keys':'Vself', 'key_resp_4.rt':'RTself','escenario':'item'})
dfAnchoring= dfAnchoring.rename(columns={'key_resp_5.keys':'Vother', 'key_resp_5.rt':'RTothers','escenario':'item'})
dfAnchoring['RD']= abs(dfAnchoring['Vself']- dfAnchoring['Vother'])

##################################################################################################################
## EGO SPATIAL DATASET
dfegoSpatial=df[['PROLIFIC_PID','gender','exp_index','Response.keys','Response.rt','distCorrSelf','distIncorrSelf','Response.corr',
                 'correctKeyboard', 'Self_proximity','SelfvsLandmark_proximity','AD','Difficulty','Angle','Type_trialEncoded',
                 'encodedTrial.started','routine_2AFC_Ego.stopped','n_objeto']]

# Corrections RTs  
mask = ((dfegoSpatial['Response.corr'] == 0) &(dfegoSpatial['Response.keys'].isna()) &(dfegoSpatial['Response.rt'].isna()))
dfegoSpatial.loc[mask, ['Response.keys', 'Response.rt']] = [0, 6]
dfegoSpatial = dfegoSpatial.dropna()
#New variable
dfegoSpatial['meanDistance']= (dfegoSpatial['distCorrSelf'] + dfegoSpatial['distIncorrSelf'])/2
dfegoSpatial['Accuracy']= dfegoSpatial.apply(correct, axis=1)

##################################################################################################################
##################################################################################################################             
                                    #--------- BAD PARTICIPANTS---------#

bad_participantsEgo = (
    dfegoSpatial.groupby('PROLIFIC_PID')['Accuracy']
         .mean()
         .loc[lambda s: s < 0.60]
         .index
    .tolist()
)
print("The number of participant with EgoSpatialTask performance lower than 60% : "+  str(len(bad_participantsEgo)))


##################################################################################################################
# Elimanate participants with bad performance# 
# Filter dfs and save dfs
dfAnchoring = dfAnchoring[~dfAnchoring['PROLIFIC_PID'].isin(bad_participants)]
dfAnchoring = dfAnchoring[~dfAnchoring['PROLIFIC_PID'].isin(bad_participantsEgo)]
dfAnchoring.to_csv(output_path + "\egosocialtaskOrder1.csv",index=False)

dfegoSpatial = dfegoSpatial[~dfegoSpatial['PROLIFIC_PID'].isin(bad_participants)]
dfegoSpatial = dfegoSpatial[~dfegoSpatial['PROLIFIC_PID'].isin(bad_participantsEgo)]
dfegoSpatial.to_csv(output_path + "\egospatialtaskOrder1.csv", index=False)

df4MT = df4MT[~df4MT['PROLIFIC_PID'].isin(bad_participants)]
df4MT = df4MT[~df4MT['PROLIFIC_PID'].isin(bad_participantsEgo)]
df4MT.to_csv(output_path + "\\allotaskOrder1.csv", index=False)

##################################################################################################################

