# nwb utils
import os
import pandas as pd
from dateutil.tz import tzlocal
from datetime import datetime
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from uuid import uuid4
from pynwb.file import Subject

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
    input(('').join(["Please edit the excel file titled: ",save_name,", resave it, then hit any key to continue..."]))
    df_new = pd.read_excel(excel_dir) 
    
    return excel_dir, df_new

def nwb_to_excel_template(dir: str):
    """
    This function creates an excel template to instantiate the nwb object off of

    Args:
        >>> dir: directory to save out nwb_template

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
    df['subject_id']=[]
    df['subject_age']=[]
    df['subject_description']=[]
    df['subject_species/genotype']=[]
    df['subject_sex']=[]
    df['recording_device_name']=[]
    df['recording_device_description']=[]
    df['recording_device_manufacturer']=[]

    # save out data
    excel_dir = os.path.join(dir,"nwb_template.xlsx")
    df.to_excel(excel_dir)

    return excel_dir, df

def template_to_nwb(template_dir: str):
    """
    Load in the template and create the NWB file
    Args:
        >>> template_dir: directory of excel template

    """
    df = pd.read_excel(template_dir) 
 
    nwbfile = NWBFile(
        # experiment details
        experiment_description=str(df['experiment_description'].values[0]),
        experimenter=str([df['experimenter name(s)'].values[0]]),
        lab=str(df['lab_name'].values[0]),
        institution=str(df['institution'].values[0]),

        # session details
        session_description=str(df['session_description'].values[0]),
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()), # filling in automatically
        session_id=str(df['session_id'].values[0]),
        notes = str(df['session_notes'].values[0])
    )

    # enter subject specific information
    nwbfile.subject = Subject(
            subject_id=str(df['subject_id'].values[0]),
            age=str(df['subject_age'].values[0]),
            description=str(df['subject_description'].values[0]),
            species=str(df['subject_species/genotype'].values[0]),
            sex=str(df['subject_sex'].values[0]),
        )

    # add recording device information
    device = nwbfile.create_device(
        name=str(df['recording_device_name'].values[0]), 
        description=str(df['recording_device_description'].values[0]), 
        manufacturer=str(df['recording_device_manufacturer'].values[0])
        )
    
    return nwbfile, device