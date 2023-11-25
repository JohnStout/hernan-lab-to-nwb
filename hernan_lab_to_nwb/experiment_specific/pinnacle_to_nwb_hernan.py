# Creating this code to be internally dependent on itself,
# rather than externally dependent on self-created packages

# numpy, pyEDFlib, ipykernel, matplotlib, openpyxl
import os
from pyedflib import highlevel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.ecephys import LFP, ElectricalSeries
from uuid import uuid4

# helpers
def pandas_excel_interactive(dir: str, df = None):
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

    Returns:
        >>> df_new: new dataFrame with modified tables from excel

    Written by John Stout
    """

    # now we save out to an excel sheet
    excel_dir = os.path.join(dir,"nwb_excel_sheet.xlsx")
    df.to_excel(excel_dir)
    print("File saved to ", excel_dir)
    input("Please edit the excel file, resave it, then hit any key to continue...")
    df_new = pd.read_excel(excel_dir) 
    
    return df_new

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

    return excel_dir

def template_to_nwb(template_dir: str):
    """
    Load in the template and create the NWB file
    Args:
        >>> template_dir: directory of excel template

    """
    df = pd.read_excel(template_dir) 
 
    nwbfile = NWBFile(
        session_description=df['session_description'].values[0],
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()), # filling in automatically
        experimenter=df['experimenter name(s)'].values[0],
        lab=df['lab_name'].values[0],
        institution=df['institution'].values[0],
        experiment_description=df['experiment_description'].values[0],
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
    
    return nwbfile

# directory input requested from user
dir = input("Enter the directory to load .edf data: ") #'/Users/js0403/edf recording'
experimenter_name = input("Enter your name: ")
session_id = input("Enter information about this recording: ")

# automation begins...
dir_contents = sorted(os.listdir(dir))
edf_names = [i for i in dir_contents if '.edf' in i]

# we could throw a time marker into the dataset through annotation on the EEG
# and through on the video side, we can use those annotations as time markers to align data
# Arduio would become source of truth. Maybe use a TTL controller rather than an Arduino bc their
# clock is imperfect.

# go back to OG serenia data, pvfs file, grab end time, then integrate into python edf file with start time, sample rate, end time.
# open up the serenia file, find the end time.
# scroll to the end of the data and annotate.

# Need the TTL controller to align start and end time, with a consistent clock
# find events in behavior, use that time to get LFPs

# if we do the TTL controller, don't do annotation every 5ms. Maybe every 5-10s.
# the accumulated error over a minute will be trivial.

# if we grab the end time off of the pinnacle pvfs datafile, then correct the edf time based on the end time of the edf file,
# we know start time, end time, number of samples, adjust the sample rate based on start-time end time, num samples, adjust sample rate
# ----> When you save data from camera, it is prob getting saved and not timestamped, prob gettings aved against PC clock. IS THIS THE MASTER CLOCK FOR EEG? YES
# --------> The end time of EEG recording, that now becomes the end time for both video and EEG, assuming video is being shut down at the same time, you now have all the info needed.


# TODO: HAVE NOT FOUND TIMESTAMPS!!!!
for edfi in edf_names:

    # this will be a loop
    dir_edf = os.path.join(dir,edfi)
    signals, signal_headers, header = highlevel.read_edf(dir_edf)

    # 4 animals, each have 3 channels
    animal_ids = [i['transducer'] for i in signal_headers]
    electr_ids = [i['label'].split(' ')[-1] for i in signal_headers]
    fs = [i['sample_rate']for i in signal_headers]
    unit = [i['dimension']for i in signal_headers]

    # make pandas array
    data_table = pd.DataFrame(animal_ids,columns=['Subject'])
    data_table['Sex'] = [[]] * data_table.shape[0]
    data_table['Genotype'] = [[]] * data_table.shape[0]
    data_table['Genotype'] = [[]] * data_table.shape[0]
    data_table['Age Range'] = [[]] * data_table.shape[0]
    data_table['Electrodes'] = electr_ids
    data_table['Skull Location'] = [[]] * data_table.shape[0]
    data_table['Sampling Rate'] = fs
    data_table['Unit'] = unit

    # excel file gets saved out, edited, then reloaded
    data_table = pandas_excel_interactive(dir = dir, df = data_table)

    # TODO: Add preprocessing module to the NWB file for times when the user does things like filters/rereferences
    datetime_str = header['startdate']

    # loop over each animal and create an NWB file
    animal_ids_uniq = list(np.unique(np.array(animal_ids)))
    for i in animal_ids_uniq:

        # test if an nwbfile exists and remove it
        if 'nwbfile' in locals():
            del nwbfile

        # restrict the pandas table to your subject
        temp_table = data_table.loc[data_table['Subject']==i]

        # create NWB file
        nwbfile = NWBFile(
            session_description="Continuous EEG recordings in TSC mice",
            identifier=str(uuid4()),
            session_start_time = datetime_str,
            experimenter = experimenters,
            lab="Hernan Lab",
            institution="Nemours Children's Hospital"
        )

        # enter subject specific information
        subject_id = 'mouse-'+i.split(':')[-1]
        subject = Subject(
                subject_id=subject_id,
                age=str(list(temp_table['Age Range'])[0]),
                description=str([]),
                species=str(list(temp_table['Genotype'])[0]),
                sex=str(list(temp_table['Sex'])[0]),
            )
        nwbfile.subject = subject

        # add recording device information
        device = nwbfile.create_device(
            name="Pinnacle", 
            description="EEG", 
            manufacturer="Pinnacle"
            )
        
        # loop over groups and create electrode objects
        nwbfile.add_electrode_column(name='label', description="label of electrode")
        brain_grouping = temp_table['Skull Location']
        for braini in temp_table.index: # loop over brain regions

            # create an electrode group for a given tetrode
            electrode_group = nwbfile.create_electrode_group(
                name=temp_table['Electrodes'][braini],
                description='EEG data',
                device=device,
                location=temp_table['Skull Location'][braini]) 

            nwbfile.add_electrode(
                group = electrode_group,
                label = temp_table['Electrodes'][braini],
                location = temp_table['Skull Location'][braini]) 

        all_table_region = nwbfile.create_electrode_table_region(
            region=list(range(temp_table.shape[0])),
            description="all electrodes")

        # get signal data and make into a numpy array (samples,num wires)
        sig_temp = np.array([])
        sig_temp = np.array(signals[list(temp_table.index)]).T
        fs = np.unique(np.array(temp_table['Sampling Rate']))
        if len(fs) > 1:
            ValueError("Different sampling rates within a single subject was detected. Must fix code to handle this... ")
        fs = float(fs)

        # add EEG data
        eeg_electrical_series = ElectricalSeries(
                name="ElectricalSeries",
                data=sig_temp,
                electrodes=all_table_region,
                starting_time=0.0,
                rate=fs)
        nwbfile.add_acquisition(eeg_electrical_series)

        dir_save = os.path.join(dir,subject_id+'.nwb')
        print("Writing .nwb file as ",dir_save)
        with NWBHDF5IO(dir_save, "w") as io:
            io.write(nwbfile)    
            io.close()
