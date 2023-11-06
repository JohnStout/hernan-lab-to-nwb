# This code is meant to be used like a config file, where you change certain variables, then hit Run
#
# Scroll down until you see "CONFIGURATION - CHANGE ME" and adjust the input as needed.
# It is recommended that you save some sort of template for this so you don't have to keep changing it.
#
# Currently, this file only handles the creation of the ophys NWB file. Later version will include additions to the file
#       --> Post processing/motion correction
#       --> ROI or mask variables
#
# John Stout - 9/12/23

# ------------------------------------- #

# dependencies:
#   pynwb
#   scikit-image
#   matplotlib
#   numpy

# This code will generate an NWB file for ophys data
from datetime import datetime
from dateutil import tz
from dateutil.tz import tzlocal

from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from skimage import io

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

from decode_lab_code.utils.dep.nwb_utils import write_nwb

# ------------------------------------------------- #

# CONFIGURATION - CHANGE ME

# NWB initialization
session_description = ""
session_id = ""
experimenter_name = "Akanksha Goyal"
lab_name = "DECODE Lab - Hernan Lab"
institution_name = "Nemours"
experiment_description = "Calcium imaging in slice"
related_publications = "NA"

# subject information
subject_id = ""
age = "PD30" # PD30
description = "" # C57BL/6J
species = "mus musculus"
sex = "" # Male/Female

# data/directory information
folder_name = '/Users/js0403/ophysdata/Akanksha/Tiff_series_Process_7' # directory where the data is stored
fname = 'tiff_stack.tif' # name of the data file
frame_rate = 10 # frame rate

# recording device information
microscope_name = "Miniscope"
microscope_description = "1Photon microscopy"
microscope_manufacturer = "UCLA developed, manufactured by OpenEphys"

# optical info
optical_channel_name = "OpticalChannel"
optical_channel_description = "Channel for ophys data"
emission_lambda = 525.0 # float, change

# description of imaging
imaging_plane_name = "imaging_plane"
imaging_plane_description = "imaging data as np array"
excitation_lambda = 645.0 # float
indicator = "GCaMP8f"
brain_location = "Hippocampus"
grid_spacing = [0, 0, 0]
grid_spacing_unit = "meters"
origin_coords = [0, 0, 0] # list of surgery coordinates
origin_coords_unit = "mm"
imaging_unit = "pixels"
imaging_technique = "1Photon" # calciumDye, 2p, etc..



# -------------------------------------------------------- #

# CREATING THE NWB FILE

# initialize the nwbfile
session_start_time = datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz("US/Pacific"))
nwbfile = NWBFile(
    session_description=session_description,  # required
    identifier=str(uuid4()),  # required
    session_start_time=session_start_time,  # required
    session_id=session_id,
    experimenter = experimenter_name,  # optional
    lab = lab_name,  # optional
    institution = institution_name,  # optional
    experiment_description=experiment_description,  # optional
    related_publications=related_publications,  # optional
)

# enter information about subject
nwbfile.subject = Subject(
    subject_id=subject_id,
    age=age,
    description=description,
    species=species,
    sex=sex,
)

# read data
data = io.imread(folder_name+'/'+fname)
data_np = np.array(data)

# create device
device = nwbfile.create_device(
    name = microscope_name,
    description = microscope_description,
    manufacturer = microscope_manufacturer,
)

optical_channel = OpticalChannel(
    name=optical_channel_name,
    description=optical_channel_description,
    emission_lambda=float(emission_lambda),
)

# create imagingplane object
imaging_plane = nwbfile.create_imaging_plane(
    name=imaging_plane_name,
    optical_channel=optical_channel,
    imaging_rate=float(frame_rate),
    description=imaging_plane_description,
    device=device,
    excitation_lambda=float(excitation_lambda),
    indicator=indicator,
    location=brain_location,
    grid_spacing=grid_spacing,
    grid_spacing_unit=grid_spacing_unit,
    origin_coords=origin_coords,
    origin_coords_unit=origin_coords_unit,
)

# using internal data. this data will be stored inside the NWB file
one_p_series1 = OnePhotonSeries(
    name=imaging_technique,
    data=data,
    imaging_plane=imaging_plane,
    rate=float(frame_rate),
    unit=imaging_unit,
)
nwbfile.add_acquisition(one_p_series1)

# save output
write_nwb(folder_name = folder_name, data_name = fname, nwb_file=nwbfile)