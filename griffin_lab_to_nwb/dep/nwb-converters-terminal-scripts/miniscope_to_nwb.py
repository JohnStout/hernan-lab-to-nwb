# Python 3.9
#
#
# Unlike the ioreaders, I'm thinking of making this a standalone script, where the user downloads and simply runs in the terminal window


"""

Download the latest release for miniscope software: 

        https://github.com/Aharoni-Lab/Miniscope-DAQ-QT-Software/releases


This package works with miniscope v4.4.

This package will create a very simple NWB file, after which you can load and append new information to. Like for example, if you want to add motion corrected or ROI data, you would do that

John Stout


TODO: MUST CREATE A CONFIG JSON FILE FOR MINISCOPE THAT FILLS IN ALL INFORMATION
People can simply take this config, use it, then it is plug and play
-> This script will be run from terminal and create the NWB file on the backend
-> Would be lovely to implement this code into Aharoni code

>>> Age
>>> name
>>> Virus type
>>> Lab
>>> Institution

#TODO: MUST FIX TIMESTAMPS

"""

#%% 
import os
import json
import pandas

# This code will generate an NWB file for ophys data
from datetime import datetime
from dateutil import tz
from dateutil.tz import tzlocal

from uuid import uuid4

import cv2

import numpy as np
import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

#from skimage import io

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.image import ImageSeries
from pynwb.ophys import (
    CorrectedImageStack,
    Fluorescence,
    ImageSegmentation,
    MotionCorrection,
    OnePhotonSeries,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)

from hernan_lab_to_nwb.utils import nwb_utils

#%% This chunk is dedicated to defining directories and loading .json or .csv files

# I really should build in a search method in case these things change with updated versions
#metaData_json = [i for i in dir_contents if 'metaData' in i][0]

# could also run a search for anything json related and store that as a 'metaData' file
#   and anything .csv related

# TODO: dir will be the only input to this code
dir = '/Users/js0403/miniscope/data/134A/AAV2/3-Syn-GCaMP8f/2023_11_14/13_21_49'
dir_contents = sorted(os.listdir(dir))

# Directly accessible information
folder_metaData = json.load(open(os.path.join(dir,'metaData.json')))
folder_notes = pandas.read_csv(os.path.join(dir,'notes.csv'))

# behavior
behavior_id = [i for i in dir_contents if 'behavior' in i][0]
behavior_dir = os.path.join(dir,behavior_id)
behavior_metaData = json.load(open(os.path.join(behavior_dir,'metaData.json')))
behavior_pose = pandas.read_csv(os.path.join(behavior_dir,'pose.csv'))

# cameraDevice
camera_id = [i for i in dir_contents if 'camera' in i][0]
camera_dir = os.path.join(dir,camera_id)
camera_metaData = json.load(open(os.path.join(camera_dir,'metaData.json')))
camera_times = pandas.read_csv(os.path.join(camera_dir,'timeStamps.csv'))

# miniscopeDevice - where the miniscope data is located - use this to identify miniscope file name
miniscope_id = [i for i in dir_contents if 'miniscope' in i][0]
miniscope_dir = os.path.join(dir,miniscope_id)
miniscope_data = [i for i in sorted(os.listdir(miniscope_dir)) if '.avi' in i]
miniscope_metaData = json.load(open(os.path.join(miniscope_dir,'metaData.json')))
miniscope_times = pandas.read_csv(os.path.join(miniscope_dir,'timeStamps.csv'))
miniscope_head_orientation = pandas.read_csv(os.path.join(miniscope_dir,'headOrientation.csv'))

# experiment
print("This version does not support the experiment folder due to no testing data")

# %% Put data into NWB

# interface excel
df = pandas.DataFrame()
# experiment information
df['experiment_description']=[]
df['experimenter name(s)']=[]
df['institution']=[]
df['lab_name']=[]
# session information
df['session_description']=[]
df['session_notes']=[]
df['session_id']=[]
# subject information
df['subject_id']=[]
df['subject_age']=[]
df['subject_description']=[]
df['subject_species/genotype']=[]
df['subject_sex']=[]
# recording device
df['recording_device_name']=miniscope_metaData['deviceName']
df['recording_device_description']=miniscope_metaData['deviceType']
df['recording_device_manufacturer']=[]

# save out data
excel_dir = os.path.join(dir,"nwb_template.xlsx")
df.to_excel(excel_dir)

# year, month, day, hour, minute, second
time_data = folder_metaData['recordingStartTime']
rec_time = datetime(time_data['year'],time_data['month'],time_data['day'],
                    time_data['hour'],time_data['minute'],time_data['second'],
                    time_data['msec'],tzinfo=tzlocal())

# creating the NWBFile
print("This file does not handle multiple custom entries")
nwbfile = NWBFile(
    session_description=input("Enter a description of what you did this session: "),
    identifier=str(uuid4()),
    session_start_time=rec_time,
    experimenter=[folder_metaData['researcherName']],
    lab=input("Enter lab name: "),
    institution=input("Enter institution name: "),
    experiment_description=folder_metaData['experimentName'],
    session_id=folder_metaData['baseDirectory'].split('/')[-1],
    notes = folder_metaData['customEntry0']
    #viral_construct = input("Enter the virus used for imaging: ")
)

# subject information
subject = Subject(
    subject_id=folder_metaData['animalName'],
    age="P90D",
    description="mouse 5",
    species="Mus musculus",
    sex="M",
)

# imaging device
device = nwbfile.create_device(
    name = miniscope_metaData['deviceType'],
    description="UCLA Miniscope v4.4",
    manufacturer="Open Ephys",
)
optical_channel = OpticalChannel(
    name="OpticalChannel",
    description="an optical channel",
    emission_lambda=500.0, # NOT SURE HOW I FIND THIS
)

imaging_plane = nwbfile.create_imaging_plane(
    name="ImagingPlane",
    optical_channel=optical_channel,
    imaging_rate=float(miniscope_metaData['frameRate']),
    description=input("What kinds of cells are you targeting? "),
    device=device,
    excitation_lambda=600.0, # WHERE DO I FIND THIS??
    indicator=input("Enter the viral construct used for imaging (e.g. AAV2/3-Syn-GCaMP8f): "),
    location=input("Enter your brain structure (e.g. PFC/V1/M2/CA1 etc...)"),
)

# save the nwb file
nwbpath = os.path.join(dir,"nwbfile.nwb")
with NWBHDF5IO(nwbpath, mode="w") as io:
    io.write(nwbfile)
del nwbfile # delete the file to remove objects

# reload
#with NWBHDF5IO(nwbpath, mode="r") as io:
    #nwbfile = io.read()

#%% Writing data to NWB file by loading it lazily

# this approach maximizes memory space by only loading what is needed,
# and only saving out what is needed. Then clearing memory of the large arrays.

# It loads the nwbfile that has no data, it adds a calcium imaging movie to it,
# saves the nwbfile, loads it again lazily (not loading the actual data), then adds
# a new object with new data, and so forth.

# This circumvents the issue of having to load all data into memory, then save all data to disk at once.

# movie times - this must be segmented according to the movie
movie_times = miniscope_times['Time Stamp (ms)']

# TODO: ADD an index to label each timestamp to a video
# Can I add a pandas array as a timestamps videO?
        
# open the NWB file in r+ mode
counter = 0; 
for i in miniscope_data:

    # define directory
    temp_dir = os.path.join(miniscope_dir,i)
    print(temp_dir)

    # read movie file
    movie_path = os.path.join(miniscope_dir,i)
    print("Reading movie from: ", movie_path)
    cap = cv2.VideoCapture(movie_path) 
    movie_data = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        else:
            movie_data.append(frame[:,:,0]) # only the first array matters
    movie_mat = np.dstack(movie_data)
    movie_data = np.moveaxis(movie_mat, -1, 0)

    # get times by index
    idx = np.arange(movie_data.shape[0])
    temp_times = movie_times[idx].to_numpy(dtype=int) # THESE ARE THE DATA TO SAVE!!!

    # now get an updated version of the movie_times, with the previous data dropped
    movie_times = movie_times.drop(idx).reset_index(drop=True)
    
    # search for nwbfile as a variab
    # le and remove it
    if 'nwbfile' in locals():
        del nwbfile

    with NWBHDF5IO(nwbpath, "r+") as io:
        print("Reading nwbfile from: ",nwbpath)
        nwbfile = io.read()

        # create OnePhotonSeries Object
        one_p_series = OnePhotonSeries(
            name="recording"+str(counter),
            data=movie_data,
            #timestamps = temp_times,
            imaging_plane=nwbfile.get_imaging_plane(),
            rate=float(miniscope_metaData['frameRate']), # I'm not sure what this refers to
            unit="raw video - rate in terms of frame-rate",
        )
        nwbfile.add_acquisition(one_p_series)

        # write the modified NWB file
        print("Rewriting nwbfile with recording",str(counter))
        io.write(nwbfile)
        io.close()
        counter += 1

    del movie_mat, movie_data, nwbfile, one_p_series

# confirmed !!!
# read to check NWB file
with NWBHDF5IO(nwbpath, "r+") as io:
    print("Reading nwbfile from: ",nwbpath)
    nwbfile = io.read()
    tester = nwbfile.acquisition['recording1'].data[:]
