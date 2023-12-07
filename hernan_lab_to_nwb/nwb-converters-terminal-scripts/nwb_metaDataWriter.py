# NWB-Excel package
# Save out different parameters for NWB creation as different NWB sheets
#
# Once the metadata is in place, the user can add what is needed.
# 
# John Stout

# GOALS:
# If the user wants to create a metadata excel file, user opens terminal and runs: run_nwb_metadata
# If the user wants to use an existing metadata file, the user opens terminal and runs: use_nwb_metadata
# then the user enters: write_nwb

# ::::: THIS IS THE WAY :::::
# I really just gotta make like folders with scripts that people can modify if needed
# ---- metadata_to_nwb
# -------- script - easy terminal access
# -------- notebook - easy user manipulation
# ---- neuralynx_to_nwb
# -------- script
# -------- notebook
# ---- pinnacle_to_nwb
# -------- script
# -------- notebook
# ---- uclaMiniscope_to_nwb
# -------- script
# -------- notebook

# likewise, I'll have jupyter notebooks in the folders

import os
import pandas as pd
from dateutil.tz import tzlocal
from datetime import datetime
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from uuid import uuid4
from pynwb.file import Subject
import openpyxl
# xlsxwriter

# TAKE WHAT IS BELOW AND EXTEND FOR ALL POSSIBLE NWB FILE TYPES

# scratch code
dir = '/Users/js0403/local data/2021-03-31_08-59-02 16eB R1 10min rec after sec drive cells on 2 and 3 - Control'

# create excel directory to save out file
filepath = os.path.join(dir,'nwb_metaData.xlsx')

# experiment details
df_exp = pd.DataFrame()
df_exp['experiment_description']=[]
df_exp['experimenter name(s)']=[]
df_exp['institution']=[]
df_exp['lab_name']=[]
df_exp['session_description']=[]
df_exp['session_notes']=[]

# subject info
df_sub = pd.DataFrame()
df_sub['subject_id']=[]
df_sub['subject_age']=[]
df_sub['subject_description']=[]
df_sub['subject_species/genotype']=[]
df_sub['subject_sex']=[]

# recording system
df_dev = pd.DataFrame()
df_dev['recording_device_name']=[]
df_dev['recording_device_description']=[]
df_dev['recording_device_manufacturer']=[]

# write to file
writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
df_exp.to_excel(writer, sheet_name = 'experiment details')
df_sub.to_excel(writer, sheet_name = 'subject details')
df_dev.to_excel(writer, sheet_name = 'device details')
writer.close()