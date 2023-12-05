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

# write to file
writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
df_exp.to_excel(writer, sheet_name = 'experiment details')
df_sub.to_excel(writer, sheet_name = 'subject details')
writer.close()
















class make_nwb():

    # this function is specific for one subject
    def __init__(self, dir: str, data_type: str = None):

        """
        Args:
            >>> dir: directory for data
            >>> data_type: follows conventions set by NWB (ophys,ephys,behavior)
        """

        # 
        df = pd.DataFrame()
        df['experiment_description']=[]
        df['experimenter name(s)']=[]
        df['institution']=[]
        df['lab_name']=[]
        df['session_description']=[]
        df['session_notes']=[]
        self.dir_nwb_init = create_excel_file(dir = dir, save_name = "nwb_init", df = df)
        pass

    def add_subject():
        pass

# helper functions
def create_excel_file(dir: str, save_name: str = 'pandas_excel', df = None):

    # df cannot be None
    if df is None:
        TypeError("df argument cannot be None. You must include a pandas array")

    # save out data
    if '.xlsx' not in save_name:
        save_name = save_name+".xlsx"

    # create excel directory to save out file
    excel_dir = os.path.join(dir,save_name)

    # write data
    df.to_excel(excel_dir, sheet_name = 'experiment details')

    # report
    print(save_name, "written to ",dir)
    return excel_dir

def add_sheet(filename: str, sheet_name: str = None, df = None):
    """
    Write to existing .xlsx file, adding a sheet
    """

    with pd.ExcelWriter(filename) as writer:
        writer.book = openpyxl.load_workbook(filename)
        df.to_excel(writer, sheet_name=sheet_name)


def pandas_to_excel(dir: str, save_name: str = 'pandas_excel', df = None):

    # df cannot be None
    if df is None:
        TypeError("df argument cannot be None. You must include a pandas array")

    # save out data
    if '.xlsx' not in save_name:
        save_name = save_name+".xlsx"

    # create excel directory to save out file
    excel_dir = os.path.join(dir,save_name)

    # write data
    df.to_excel(excel_dir, index=True, na_rep='NaN')

    # report
    print(save_name, "written to ",dir)
    return excel_dir

# TESTERS
dir = '/Users/js0403/local data/2021-03-31_08-59-02 16eB R1 10min rec after sec drive cells on 2 and 3 - Control'
make_nwb(dir = dir)



# tester code for functions above - need to write, then add sheets, to a file
df = pd.DataFrame()
df['experiment_description']=[]
df['experimenter name(s)']=[]
df['institution']=[]
df['lab_name']=[]
df['session_description']=[]
df['session_notes']=[]
filename = create_excel_file(dir = dir, save_name = "nwb_init", df = df)
# now load and write
with pd.ExcelWriter(filename) as writer:
    writer.book = openpyxl.load_workbook(filename)
    df.to_excel(writer, sheet_name='tester')



writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')
df.to_excel(writer, sheet_name = 'x1')
df.to_excel(writer, sheet_name = 'x2')
writer.close()





def nwb_to_excel_template(dir: str, save_name: str = "nwb_template"):
    """
    This function creates an excel template to instantiate the nwb object off of

    Args:
        >>> dir: directory to save out nwb_template
        >>> save_name: file name for your excel sheet

    Returns:
        >>> excel_dir: directory of excel nwb template to initialize the NWB object
    
    """
    df = pd.DataFrame()
    df['experiment_description']=[]
    df['experimenter name(s)']=[]
    df['institution']=[]
    df['lab_name']=[]
    df['session_description']=[]
    df['session_notes']=[]
    df['session_id']=[]

    df['recording_device_name']=[]
    df['recording_device_description']=[]
    df['recording_device_manufacturer']=[]

    # save out data
    excel_dir = pandas_to_excel(dir = dir, save_name = save_name, df = df)
    return excel_dir

def template_to_nwb(template_dir: str):
    """
    Load in the template and create the NWB file
    Args:
        >>> template_dir: directory of excel template

    """
    df = pd.read_excel(template_dir) 
 
    nwbfile = NWBFile(

        # experiment details
        experiment_description=df['experiment_description'].values[0],
        experimenter=[df['experimenter name(s)'].values[0]],
        lab=df['lab_name'].values[0],
        institution=df['institution'].values[0],

        # session details
        session_description=str(df['session_description'].values[0]),
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()), # filling in automatically
        session_id=df['session_id'].values[0],
        notes = df['session_notes'].values[0]
    )

    # enter subject specific information
    nwbfile.subject = Subject(
            subject_id=df['subject_id'].values[0],
            age=df['subject_age'].values[0],
            description=df['subject_description'].values[0],
            species=df['subject_species/genotype'].values[0],
            sex=df['subject_sex'].values[0],
        )

    # add recording device information
    device = nwbfile.create_device(
        name=df['recording_device_name'].values[0], 
        description=df['recording_device_description'].values[0], 
        manufacturer=df['recording_device_manufacturer'].values[0]
        )
    
    return nwbfile, device




def pandas_excel_interactive(dir: str, df = None, save_name: str = "nwb_excel_sheet.xlsx"):
    """
    You pass this function a dataFrame, it saves it as an excel file,
        you then modify that excel file, resave it, then hit any button on 
        your keypad in the terminal or interactive python window. The function
        concludes by returning the updated dataFrame with your added info.

    Unlike the template creation with nwb_to_excel_template, nor the template conversion
        with template_to_nwb, pandas_excel_interactive takes a dataFrame with whatever columns
        you have designed, then allows you to modify it.

    In simpler terms, the functions with term "template" are not flexible in their dataFrame
        column names. This function is agnostic.

    Args:
        >>> dir: directory of where to save data
        >>> df: dataFrame to modify
        >>> save_name: an optional input to change the output save file name

    Returns:
        >>> df_new: new dataFrame with modified tables from excel

    Written by John Stout
    """

    # now we save out to an excel sheet
    excel_dir = os.path.join(dir, save_name)
    df.to_excel(excel_dir)
    print("File saved to ", excel_dir)
    input("Please edit the excel file, resave it, then hit any key to continue...")
    df_new = pd.read_excel(excel_dir) 
    
    return df_new