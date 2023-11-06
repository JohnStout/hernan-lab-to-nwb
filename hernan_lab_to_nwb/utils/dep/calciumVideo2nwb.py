# This code will be a simple module for convering ophys to nwb
# dependencies:
#   pynwb
#   scikit-image
#   matplotlib
#   numpy

# This code will generate an NWB file for ophys data
from datetime import datetime
from uuid import uuid4
import numpy as np
from dateutil import tz
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

#import cv2
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
from dateutil.tz import tzlocal

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

session_start_time = datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz("US/Pacific"))
# initialize the nwbfile
nwbfile = NWBFile(
    session_description=input("Enter description of your recording session: "),  # required
    identifier=str(uuid4()),  # required
    session_start_time=session_start_time,  # required
    session_id=input("Enter unique identifier for session: "),  # optional
    experimenter=[
        input("Enter experimenter name: "),
    ],  # optional
    lab=input("Enter lab name: "),  # optional
    institution=input("Enter institution name: "),  # optional
    experiment_description=input("Enter a description of your experiment"),  # optional
    related_publications=input("Enter any information about publication (if relevant)"),  # optional
)

# enter information about subject
nwbfile.subject = Subject(
    subject_id=input("Enter subject ID: "),
    age=input("Enter subject age as such (PD100):  "),
    description=input("Enter subject identifier: "),
    species=input("Enter species name: "),
    sex=input("Enter sex of subject: "),
)

# directory information
folder_name = input("Enter the folder name for your data: ")
fname_neuron = input("Enter file name with extension: ")
frame_rate = float(input("Enter the frame rate: "))

# read data
data = io.imread(folder_name+'/'+fname_neuron)

# working on getting the real ophys way working. nwbwidgets spits an error

# create device
device = nwbfile.create_device(
    name=input('Microscope name: '),
    description= input('Microscope description (i.e. 1p/2p/etc...)',
    manufacturer="The best microscope manufacturer",
)
optical_channel = OpticalChannel(
    name="OpticalChannel",
    description="an optical channel",
    emission_lambda=525.0,
)

# create imagingplane object
imaging_plane = nwbfile.create_imaging_plane(
    name="ImagingPlane",
    optical_channel=optical_channel,
    imaging_rate=frame_rate,
    description="Activation of cells",
    device=device,
    excitation_lambda=600.0,
    indicator="GFP",
    location="Somewhere",
    grid_spacing=[0.01, 0.01],
    grid_spacing_unit="meters",
    origin_coords=[1.0, 2.0, 3.0],
    origin_coords_unit="meters",
)

# using internal data. this data will be stored inside the NWB file
one_p_series1 = OnePhotonSeries(
    name="CalciumDye",
    data=data,
    imaging_plane=imaging_plane,
    rate=10.0,
    unit="pixels",
)

nwbfile.add_acquisition(one_p_series1)
with NWBHDF5IO(folder_name+"/data_ophys_nwb.nwb", "w") as io:
    io.write(nwbfile)
