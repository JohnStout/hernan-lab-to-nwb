# caiman nwb-excel interface for metadata
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