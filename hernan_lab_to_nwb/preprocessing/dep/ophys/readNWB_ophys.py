## This code is meant to act as an interface between nwb files and caiman
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# fill out directory and dataset names
dir = '/Users/js0403/ophysdata/DANDI/Plitt & Giocomo (2021)'
data_name = 'sub-F2_ses-20190416T210000_behavior+ophys.nwb'

# load the data sequentially, then memory map the output.
with NWBHDF5IO(dir+'/'+data_name, "r") as io:

    # read file lazily
    read_nwbfile = io.read()

    # first get the spatial shape of the dataset.
    spat_shape = read_nwbfile.acquisition["TwoPhotonSeries"].data[0,:,:].shape
    temp_shape = read_nwbfile.acquisition["TwoPhotonSeries"].data[:,0,0].shape

    # to memory map, first memory map an empty variable
    array_shape = (temp_shape[0],spat_shape[0],spat_shape[1])
    emp_array = np.zeros(shape = array_shape, dtype = float)
    data = read_nwbfile.acquisition["TwoPhotonSeries"].data[:]

    # Then read, write sequentially




    