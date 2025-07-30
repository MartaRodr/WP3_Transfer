# Transfer bias social and spatial task

## 30/07
---
- Cleaning data and create project structure
- Folder structure

- Script to create datasets for each  orders, eliminating the
participants that did not performed well the task: (4MT Acc<0.3, and SpatialEgoTask<0.6)
   - scripting/preprocessing/DataSets.py  
   - scripting/preprocessing/DataSetsOrders.py

- Outputs:
  - `data/processed/allotask.csv`
  - `data/processed/egospatialtask.csv`
  - `data/processed/egosocialtask.csv`
  - `data/processed/df_summaryAgeGender.csv` 

- Scripting folder:
  - `scripting/preprocessing/`
  - `scripting/modelling/` : Folder for each experiment and the analysis runned to conect all the experiments
    - `scripting/modelling/egosocialtask/`
    In this folder we will have the scripts to run the models for the egosocial task
     - Model:
       ```r
       RD ~ RTlog_cwc * Individual + (1 | PROLIFIC_PID) + (0+RTlog_cwc || PROLIFIC_PID)
       
       mod1 <- mixed(RD ~ RTlog_cwc * Individual + (1 | PROLIFIC_PID), data=dat, method="S")
       ```

    - `scripting/modelling/egospatialtask/`
      
    - `scripting/modelling/allotask/`  
 

