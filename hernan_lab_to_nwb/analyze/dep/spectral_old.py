# spectral.py
#
# The purpose of this code is to provide an interface between NWB and packages like
# scipy and pynapple. This file is required to lower the energy barrier between
# collecting data and analyzing data. There are some lab-specific features that need to
# be accounted for, like how we collect data as that can impact what sort of information
# inputs to NWB. 
#
# Pynapple has a ton of incredible open source features for unit analysis, but it's missing
# some unit-LFP analyses, particularly LFP analyses
#
# scipy.signal has a ton of signal processing tools
#
# There already exists open source tools for things like spike field coherence
#
# This code draws heavily on open sources tools and therefore has lower risk for code error.
#
# Written by John Stout

from scipy.signal import welch
from decode_lab_code.readers.ioreaders import load_nwb
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from decode_lab_code.core.base import base
from decode_lab_code.readers.ioreaders import read_nlx
import pandas as pd

# run below in interactive window for troubleshooting
"""
    from decode_lab_code.analyze.spectral import spectral
    nwbpath = '/Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS/nwbfile.nwb'
    self = spectral(nwbpath)
"""

print("Cite Pynapple and PYNWB")

class spectral():

    def __init__(self, nwbpath: str):
        #pass

        """
        This init is dependent on using ioreader.write_nwb
        
        Args:
            >>> nwbpath: directory that includes the nwbfile
            >>> fs: sampling rate of the dataset. Can create multiple objects if needed.
        """

        # load the nwbfile and important data, like events
        self.nwbfile = load_nwb(nwbpath=nwbpath)
        self.events = self.nwbfile.epochs.to_dataframe()
        self.csc_data_names = self.nwbfile.electrodes.to_dataframe()['label']
        #self.rec_times = self.csc_data[0].time_support        
        print(self.events)

        # get unit data
        self.unit_data = nap.load_file(nwbpath)['units']

        # get csc data
        csc_data = nap.load_file(nwbpath)['ElectricalSeries']
        csc_times = self.nwbfile.acquisition['ElectricalSeries'].timestamps[:]

        # estimate sampling rate
        start_time = self.nwbfile.epochs.to_dataframe()['start_time'][0]
        end_time = self.nwbfile.epochs.to_dataframe()['stop_time'][0]
        temp_data = csc_data[0].data()
        numsamples = len(temp_data[np.where(csc_times==start_time)[0][0]:np.where(csc_times==end_time)[0][0]])
        self.fs = np.round(numsamples/(end_time-start_time)).astype(float)

        # get signal from separate start/stop times
        start_times = [i for i in self.events.start_time] # get times
        end_times = [i for i in self.events.stop_time] # get times
        start_idx = []; end_idx = []
        for i in start_times:
            start_idx.append(np.where(csc_times==i))
        for i in end_times:
            end_idx.append(np.where(csc_times==i))
        
        # sanity check
        if len(start_times) != len(end_times):
            ValueError("start_times do not match end times - something is wrong with the recording or extraction")

        # organize data into pandas array (TsdFrame) - organizing features are recording, time(neuralynx), time(sec)
        # TODO: Include helper functions to include epochs
        rec_var = np.zeros(csc_data.shape[0]); times = []
        for i in range(len(start_times)):
            rec_var[start_idx[i][0][0]:end_idx[i][0][0]+1]=i+1
            numsamples = np.where(rec_var==float(i+1))[0].shape[0]
            times.append(np.linspace(0,numsamples/self.fs,numsamples))
        times_data = np.hstack(times)
        # can prove above works by plotting rec_var with and without +1 indexing

        """       
        # change information about tsdFrame
        csc_data = csc_data.reset_index() # reset 
        csc_data.rename(columns={'Time (s)': 'Time (neuralynx)'})
        csc_data.insert(0,"Time (sec)", times_data, True)
        csc_data.insert(2,"RecordingIndex",rec_var, True)

        # reformat
        self.csc_data = csc_data
        self.csc_data.index.name = 'Time (neuralynx)'
        csc_data.insert(loc=[0,1],column=["Time (sec)","Recording"],value=[times_data,rec_var])
        self.csc_data.insert(loc=0,column="Time (sec)",value=times_data)
        """

        # set index
        
        # save history
        self.history = []
        self.history.append("nwbfile: the full nwbfile, read lazily")
        self.history.append("unit_data: unit data and corresponding times, as read with pynapple")
        self.history.append("csc_data: lfp data and corresponding times, as read using pynapple")
        self.history.append("csc_data_names: names of lfp channels and their corresponding indices to lfp_data")
        self.history.append("rec_times: recording start and stop times, as retrieved from pynapple")
        self.history.append("csc_times: csc timestamps")
        self.history.append("fs: estimated sampling rate from the first .ncs file")


    def name_to_electrode(self, channel_name: str):
        """
        Given an nwbfile and the name of your CSC channels, return the index of that CSC to pynapple

        Args:
        >>> channel_name: name of LFP channel to get index back to pynapple data

        Returns:
        >>> idx: the index of a specific channel name in the NWB file to pynapple tsdata

        """
        #id_label = self.nwbfile.electrodes.to_dataframe()['label']
        idx = self.csc_data_names.loc[self.csc_data_names==channel_name].index.tolist()[0]
        return idx
    
    def power(self, channel_name: str, start_time: list = [], end_time: list = [], unit_time: str = 'neuralynx'):

        """
        Run welch's power analysis

        Args:
            >>> channel_name: name of the CSC channel you want to use
            >>> start_time: start time for your analysis (accepts lists of time)
            >>> end_time: list of times to index to. MUST MATCH START_TIME SIZE/SHAPE
            >>> unit_time (OPTIONAL): 'neuralynx' or 'sec'
                    >>> if neuralynx, you can use the raw timestamp values provided by the NWB file and data 
                    collection procedure
                    >>> if 'sec', the timestamps are converted to seconds
        """

        # get start/end time

        # TODO: CONSIDER making the start_time and end_time in neuralynx time or in relative time (0-10s).
        print("Running power analysis")
        if start_time is None:
            start_time_samples = self.csc_data.time_support.start.tolist()[0]
            print(" between start")
        #else:
            #start_time_samples = start_time*self.fs

        if end_time is None:
            end_time = self.csc_data.time_support.end.tolist()[0]
            print(" and end times")

        # get lfp signal
        idx = self.name_to_electrode(channel_name)
        ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's')
        data = self.csc_data[idx].restrict(ep).data()

        # power spectrum
        f,Ptemp = welch(lfp,fs,nperseg=fs)

        # restrict data to 1-50hz for plot proofing
        #f[f>1]
        idxspec = np.where((f>1) & (f<100))
        fSpec = f[idxspec]
        PSpec = Ptemp[idxspec]

        # log10 transform
        PSpecLog = np.log10(PSpec)

        pass
    
    
    





