# ioreaders
#
# input-output readers
#
# specific purpose is to convert between raw data and dictionaries
#
# written by John Stout

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4
import itertools

import re
import os
import pandas as pd
import json

# can I import this later??
import cv2

# pynwb
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject
from pynwb import validate
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

# numpy
import numpy as np

# loading neo package
from hernan_lab_to_nwb.utils.neuralynxrawio import NeuralynxRawIO
from hernan_lab_to_nwb.utils.neuralynxio import NeuralynxIO

# pyedflib
from pyedflib import highlevel

# from utils
from hernan_lab_to_nwb.utils import nlxhelper

# inheritance - base __init__ inherited
from hernan_lab_to_nwb.core.base import base

# import utilities for nwb 
from hernan_lab_to_nwb.utils import nwb_utils

# neuralynx data
class read_nlx(base):

    def read_all(self):

        """
        TODO: read all data at once
        Argument that allows the user to read all information from a file using the methods
        ascribed below
        """

        # just run everything below
        self.read_ephys()
        self.read_events()
        self.read_header()
        self.read_vt()

    def read_ephys(self, opts = None):

        """
        A method to read electrophysiology data acquired by Neuralynx Cheetah in DECODE lab

        Args:
            TODO: opts: optional argument for which data to load in
        
        Returns:
            csc_data: data acquired and stored as .ncs

        """

        # TODO: Build some code that checks for timestamps in spikes outside of timestamps in LFP
        
        
        # Use Neo package
        print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

        # read events data
        # TODO: Make events into a dictionary
        self.read_events()

        # opts
        if opts is None:
            opts = ['CSC','TT']

        # group data according to extension, then by naming
        split_contents = [i.split('.') for i in self.dir_contents]

        # extract extension values
        ext = [split_contents[i][1] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # extract pre-extension names, if . was used to split
        pre_ext = [split_contents[i][0] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # group extensions
        unique_ext = np.unique(ext) # can test for unique extension names

        # here is a way to do a letter-to-digit search and return letter combo
        #naming = "".join([i for i in pre_ext[10] if i.isdigit()==False])

        # group data based on extension type
        csc_names = []; tt_names = []
        for ci in self.dir_contents:
            if '.ncs' in ci.lower():
                csc_names.append(ci)
            elif '.ntt' in ci.lower():
                tt_names.append(ci)

        # sort files
        def atoi(text):
            return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)',text) ]

        # sort
        csc_names.sort(key=natural_keys)
        tt_names.sort(key=natural_keys)

        # now lets put these into a dict for working with in NeuralynxIO
        neural_dict = {'CSC': csc_names, 
                        'TT': tt_names}
        
        # Here we create separate dictionaries containing datasets with their corresponding labels
        dict_keys = neural_dict.keys()
        self.csc_data = dict(); self.tt_data = dict(); self.csc_data_fs = dict()
        csc_added = False; tt_added = False
        for groupi in dict_keys: # grouping variable to get TT data
            print("Working with",groupi)
            for datai in neural_dict[groupi]: # now we can get data

                # read data using Neo's NeuralynxIO
                if 'blks' in locals():
                    del blks
                blks = NeuralynxIO(filename=self.folder_path+self.slash+datai, keep_original_times=True).read(lazy=False) # blocks
                #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

                if len(blks) > 1:
                    TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

                # get blocked data
                blk = blks[0]

                # TODO: Handle multisegments (CSC1 from /Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS)
                # You can probably just combine the csc_times and csc_data into one vector

                # TODO: Get sampling rate

                # organize data accordingly 
                # it is VERY important that you only include LFP times between starting/stopping recording
                if 'CSC' in groupi and 'CSC' in opts: # CSC files referenced to a different channel
                    
                    # doesn't matter how many blocks there are, concatenate, then separate by events
                    
                    # do a search for starting/stopping recordings
                    counter=0; start_times = []; end_times = []
                    for i in self.event_strings:
                        if 'starting recording' in i.lower():
                            start_times.append(self.event_times[counter])
                        elif 'stopping recording' in i.lower():
                            end_times.append(self.event_times[counter])
                        #print(counter)
                        counter+=1

                    # restrict CSC data to these times
                    temp_csc = []; temp_times = []; csc_fs = []
                    for segi in range(len(blk.segments)):
                        temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                        temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())

                    if len(temp_times) > 1:
                        Warning("Multiple segments detected. Check code.")

                    # now restrict CSC data and times to be within event_times
                    for i in range(len(start_times)):
                        # convert to numpy
                        temp_times[i]=np.array(temp_times[i])
                        temp_csc[i]=np.array(temp_csc[i])
                        # get index of start/stop using dsearchn
                        idx_start = int(dsearchn(temp_times[i],start_times[i])[0])
                        idx_end = int(dsearchn(temp_times[i],end_times[i])[0])
                        # get data in between - have to add 1 at the end because python sees this as [0:-1] (all but the last datapoint)
                        temp_csc[i] = temp_csc[i][idx_start:idx_end+1]
                        temp_times[i] = temp_times[i][idx_start:idx_end+1]

                    # horizontally stack data
                    self.csc_data[datai] = np.hstack(temp_csc)
                    self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient                        

                    # add sampling rate if available

                    #TODO: add fs for each csc channel and segment!!!
                    temp_fs = float(blk.segments[0].analogsignals[0].sampling_rate.magnitude)
                    self.csc_data_fs[datai] = temp_fs
                    csc_added = True

                # Notice that all values are duplicated. This is because tetrodes record the same spike times.
                # It is the amplitude that varies, which is not captured here, despite calling magnitude.
                elif 'TT' in groupi and 'TT' in opts: # .ntt TT files with spike data

                    if len(blk.segments) > 1:
                        InterruptedError("Detected multiple stop/starts in spike times. No code available to collapse recordings. Please add code")

                    spikedata = blk.segments[0].spiketrains
                    num_tts = len(spikedata)
                    if num_tts > 4:
                        print("Detected clustered data in",datai)
                        num_trains = len(spikedata) # there will be 4x the number of clusters extracted
                        num_clust = int(num_trains/4) # /4 because there are 4 tetrodes
                        temp_dict = dict()
                        for i in range(num_clust):
                            if i > 0: # skip first cluster, it's noise
                                temp_dict['cluster'+str(i)+'spiketimes'] = spikedata[i].magnitude
                        self.tt_data[datai] = temp_dict
                    else:
                        temp_dict = dict()
                        for i in range(num_tts):
                            temp_dict['channel'+str(i)+'spiketimes'] = spikedata[i].magnitude
                        self.tt_data[datai] = temp_dict
                    tt_added = True
                    self.tt_data_fs = int(32000) # hardcoded

        # history tracking
        if 'blk_logger' in locals():
            self.history.append("LOGGER: multiple start/stop recordings detected. CSC data is ")

        # get keys of dictionary
        if csc_added is True:
            self.csc_data_names = csc_names
            self.history.append("csc_data: CSC data as grouped by ext .ncs")
            self.history.append("csc_data_names: names of data in csc_data as organized by .ncs files")
            self.history.append("csc_data_fs: sampling rate for CSC data, defined by .ncs extension")
            self.history.append("csc_times: timestamps for csc data - accounts for multiple start/stop times")

            # add a grouping table for people to dynamically edit
            self.csc_grouping_table = pd.DataFrame(self.csc_data_names)
            self.csc_grouping_table.columns=['Name']
            self.csc_grouping_table['TetrodeGroup'] = [[]] * self.csc_grouping_table.shape[0]
            self.csc_grouping_table['BrainRegion'] = [[]] * self.csc_grouping_table.shape[0]
            self.csc_grouping_table['Inclusion'] = [True] * self.csc_grouping_table.shape[0]

            self.history.append("csc_grouping_table: pandas DataFrame to organize csc. This is good if you want to cluster data as the NWB file will detect your organization. try adding structure columns and tetrode grouping columns!")
            self.history.append("csc_grouping_table.TetrodeGroup: group for tetrode assignment (CSC1-4 might belong to Tetrode 1)")
            self.history.append("csc_grouping_table.BrainRegion: Enter Structure")
            self.history.append("csc_grouping_table.Inclusion: default is True, set to False if you want to exclude grouping in NWB")


        if tt_added is True:
            self.tt_data_names = tt_names
            self.history.append("tt_data: Tetrode data as grouped by ext .ntt")
            self.history.append("tt_data_names: names of data in tt_data as organized by .ntt files")
            self.history.append("tt_data_fs: hard coded to 32kHz after not detected neo extraction of sampling rate")
            
            # add a grouping table for people to dynamically edit
            self.tt_grouping_table = pd.DataFrame(self.tt_data_names)
            self.tt_grouping_table.columns=['Name']
            self.tt_grouping_table['TetrodeGroup'] = [[]] * self.tt_grouping_table.shape[0]
            self.tt_grouping_table['BrainRegion'] = [[]] * self.tt_grouping_table.shape[0]
            self.tt_grouping_table['Inclusion'] = [True] * self.tt_grouping_table.shape[0]

            self.history.append("tt_grouping_table: pandas DataFrame to organize csc. This is good if you want to cluster data as the NWB file will detect your organization. try adding structure columns and tetrode grouping columns!")
            self.history.append("tt_grouping_table.TetrodeGroup: group for tetrode assignment (CSC1-4 might belong to Tetrode 1)")
            self.history.append("tt_grouping_table.BrainRegion: Enter Structure")
            self.history.append("tt_grouping_table.Inclusion: default is True, set to False if you want to exclude grouping in NWB")

    def read_ncs_file(self, wire_names: str):

        '''
        This function reads whichever ncs files the user requires

        Args:
            >>> wire_names: list array containing wire names. Takes partial inputs (e.g. 'CSC1' rather than 'CSC1.ncs')
        
        '''

        # Use Neo package
        print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

        # read events data
        # TODO: Make events into a dictionary
        self.read_events()

        # group data according to extension, then by naming
        split_contents = [i.split('.') for i in self.dir_contents]

        # extract extension values
        ext = [split_contents[i][1] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # extract pre-extension names, if . was used to split
        pre_ext = [split_contents[i][0] for i in range(len(split_contents)) if len(split_contents[i])>1]

        # group extensions
        unique_ext = np.unique(ext) # can test for unique extension names

        # check that wire_names is a list
        if type(wire_names) is str:
            wire_names = [wire_names] # wrap string into list
            print("Single named input wrapped to list")

        # do a global path search for the files input
        csc_names = []; tt_names = []
        for ci in self.dir_contents:
            for wi in wire_names:
                if wi.lower() in ci.lower():
                    csc_names.append(ci)

        if len(csc_names) == 0:
            raise TypeError("No CSC files found at:",self.folder_path)

        # sort
        csc_names.sort(key=natural_keys)
        tt_names.sort(key=natural_keys)

        # now lets put these into a dict for working with in NeuralynxIO
        neural_dict = {'CSC': csc_names}
        
        # Here we create separate dictionaries containing datasets with their corresponding labels
        dict_keys = neural_dict.keys()
        self.csc_data = dict(); self.tt_data = dict(); self.csc_data_fs = dict()
        csc_added = False; tt_added = False
        for groupi in dict_keys: # grouping variable to get TT data
            print("Working with",groupi)
            for datai in neural_dict[groupi]: # now we can get data

                # read data using Neo's NeuralynxIO
                if 'blks' in locals():
                    del blks
                blks = NeuralynxIO(filename=self.folder_path+self.slash+datai, keep_original_times=True).read(lazy=False) # blocks
                #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

                if len(blks) > 1:
                    TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

                # get blocked data
                blk = blks[0]

                # TODO: Handle multisegments (CSC1 from /Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS)
                # You can probably just combine the csc_times and csc_data into one vector

                # TODO: Get sampling rate

                # organize data accordingly 
                # it is VERY important that you only include LFP times between starting/stopping recording
                if 'CSC' in groupi: # CSC files referenced to a different channel
                    
                    # doesn't matter how many blocks there are, concatenate, then separate by events
                    
                    # do a search for starting/stopping recordings
                    counter=0; start_times = []; end_times = []
                    for i in self.event_strings:
                        if 'start' in i.lower():
                            start_times.append(self.event_times[counter])
                        elif 'stop' in i.lower():
                            end_times.append(self.event_times[counter])
                        #print(counter)
                        counter+=1

                    # restrict CSC data to these times
                    temp_csc = []; temp_times = []; csc_fs = []
                    for segi in range(len(blk.segments)):
                        temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                        temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())

                    if len(temp_times) > 1:
                        Warning("Multiple segments detected. Check code.")

                    # now restrict CSC data and times to be within event_times
                    for i in range(len(start_times)):
                        # convert to numpy
                        temp_times[i]=np.array(temp_times[i])
                        temp_csc[i]=np.array(temp_csc[i])
                        # get index of start/stop using dsearchn
                        idx_start = int(dsearchn(temp_times[i],start_times[i])[0])
                        idx_end = int(dsearchn(temp_times[i],end_times[i])[0])
                        # get data in between - have to add 1 at the end because python sees this as [0:-1] (all but the last datapoint)
                        temp_csc[i] = temp_csc[i][idx_start:idx_end+1]
                        temp_times[i] = temp_times[i][idx_start:idx_end+1]

                    # horizontally stack data
                    self.csc_data[datai] = np.hstack(temp_csc)
                    self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient                        

                    # add sampling rate if available

                    #TODO: add fs for each csc channel and segment!!!
                    temp_fs = float(blk.segments[0].analogsignals[0].sampling_rate.magnitude)
                    self.csc_data_fs[datai] = temp_fs
                    csc_added = True        

    def read_vt(self):

        # Get VT data from .NVT files
        vt_name = [i for i in self.dir_contents if '.nvt' in i.lower()][0]

        # get video tracking data if it's present
        filename = os.path.join(self.folder_path,vt_name)
        # Example usage:

        # data = read_nvt("path_to_your_file.nvt")

        vt_data = nlxhelper.read_nvt(filename = filename)
        self.vt_x = vt_data['Xloc']
        self.vt_y = vt_data['Yloc']
        self.vt_t = vt_data['TimeStamp']

        # add history
        self.history.append("vt_x: x-position data obtained from .nvt files")
        self.history.append("vt_y: y-position data obtained from .nvt files")
        self.history.append("vt_t: timestamp data obtained from .nvt files")

    def read_events(self):

        """
        TODO: Read events information and this information will be packaged into nwb.epochs
        
        """
        # Get VT data from .NVT files
        ev_name = [i for i in self.dir_contents if '.nev' in i.lower()][0]

        # get video tracking data if it's present
        filename = os.path.join(self.folder_path,ev_name)

        # read data
        blks = NeuralynxIO(filename=filename, keep_original_times=True).read(lazy=False) # blocks       
        #blks = neo_io(filename=filename, keep_original_times=True).read(lazy=False) # blocks       
        
        if len(blks) > 1:
            TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

        # get blocked data
        blk = blks[0]     

        # loop over blocked data if multiple blocks exist
        event_strings = []; event_times = []
        if len(blk.segments) > 1:
            TypeError("CODE DOES NOT HANDLE MULTIPLE BLOCKS - FIX")
        else:
            events = []; times = []
            for blkev in range(len(blk.segments[0].events)):
                event_dict = blk.segments[0].events[blkev].__dict__
                events.append(event_dict['_labels'])
                times.append(blk.segments[0].events[blkev].times.magnitude)
            #appeend and sort
            event_strings = list(itertools.chain(*events))
            event_times = list(itertools.chain(*times))

        # get the event strings sorted by event times
        self.event_strings, self.event_times = sortXbyY(x = event_strings, y = event_times)

        self.history.append("event_strings: Event variables during recordings (in written format)")
        self.history.append("event_times: Event variables during recordings (in seconds)")

    def read_header(self):

        # attempt to read files until you get a header
        ncs_file = [i for i in self.dir_contents if '.ncs' in i.lower()]
        for i in ncs_file:
            try:
                reader = NeuralynxRawIO(filename = os.path.join(self.folder_path,i))
                reader.parse_header()
                file_header = reader.file_headers
            except:
                pass
            if 'file_header' in locals():
                break

        # time information for NWB
        header_dict = dict(list(file_header.values())[0])
        #datetime_str = header_list['recording_opened']
        self.header = header_dict
        self.history.append("header: example header from the filepath of a .ncs file")

    def write_nwb(self, metadata = None):

        """
        All .ncs files will be taken in
        >>> metadata: pandas array of metadata. Keep set to None unless you know exactly what metadata works

        """

        # TODO: Add preprocessing module to the NWB file for times when the user does things like filters/rereferences

        # make sure you have the header
        self.read_header()
        datetime_str = self.header['recording_opened']

        # TODO: THIS IS A GENERAL FUNCTION THAT will be supported by all write_nwb functions
        # create NWB template interactivately
        if metadata is None:
            template_dir, df_temp = nwb_utils.nwb_to_excel_template(self.folder_path)      
            print("nwb_template.xlsx written to", self.folder_path)
            input("Please fill in the nwb_template.xlsx sheet, then press any key to continue...")
            nwbfile, device = nwb_utils.template_to_nwb(template_dir = template_dir)
        else:
            nwbfile, device = nwb_utils.template_to_nwb(template_data = metadata)

        #%% 

        # get start and stop times, according to the self.event_strings data
        times_start = [self.event_times[i] for i in range(len(self.event_strings)) if "starting recording" in self.event_strings[i].lower()]
        times_stop  = [self.event_times[i] for i in range(len(self.event_strings)) if "stopping recording" in self.event_strings[i].lower()]

        for i in range(len(times_start)):
            nwbfile.add_epoch(times_start[i], times_stop[i], ["rec"+str(i)])
        nwbfile.epochs.to_dataframe()

        #TODO: This will get more complicated for the fear dataset that had some pre-defined tone condition

        #%% before moving forward, remove any rows set to be removed in the pandas array
               
        # remove any rows of the pandas array if the inclusion is set to False
        try:
            rem_data_csc = self.csc_grouping_table['Inclusion'][self.csc_grouping_table['Inclusion']==False].index.tolist()
            if len(rem_data_csc) > 1:
                self.csc_grouping_table=self.csc_grouping_table.drop(index=rem_data_csc)   
                print("Removing:\n",self.csc_grouping_table.iloc[rem_data_csc].Name) 
            self.csc_grouping_table=self.csc_grouping_table.reset_index()       
        except:
            pass
        try:
            rem_data_tt = self.tt_grouping_table['Inclusion'][self.tt_grouping_table['Inclusion']==False].index.tolist()
            if len(rem_data_tt) > 1:
                self.tt_grouping_table=self.tt_grouping_table.drop(index=rem_data_tt)
                print("Removing:\n",self.tt_grouping_table.iloc[rem_data_tt].Name)
            self.tt_grouping_table = self.tt_grouping_table.reset_index()
        except:
            pass


        #%% Add electrode column and prepare for adding actual data

        # The first step to adding ephys data is to create categories for organization
        nwbfile.add_electrode_column(name='label', description="label of electrode")

        # loop over pandas array, first organize by array, then index tetrode and electrode
        brain_regions = self.csc_grouping_table['BrainRegion'].unique().tolist()
        electrode_group = self.csc_grouping_table['TetrodeGroup'].unique().tolist() # grouped into tetrde
        csc_table_names = self.csc_grouping_table['Name'].tolist()
        self.csc_grouping_table.set_index('Name', inplace=True) # set Name as the index
        #self.csc_grouping_table.reset_index(inplace=True)
        
        # now create electrode groups according to brain area and electrode grouping factors
        try:
            for bi in brain_regions: # loop over brain regions

                for ei in electrode_group: # loop over tetrode or probe

                    # create an electrode group for a given tetrode
                    electrode_group = nwbfile.create_electrode_group(
                        name='Tetrode{}'.format(ei),
                        description='Raw tetrode data',
                        device=device,
                        location=bi)     
        except:
            print("Failed to create electrode group. You may only have one channel per group.")
            pass

        # loop over the pandas array for csc data, then assign the data accordingly
        electrode_counter = 0
        for csci in csc_table_names: # loop over electrodes within tetrode

            # get index of csc belonging to brain region bi and electrode ei
            pd_series = self.csc_grouping_table.loc[csci]
            #electrode_group = nwbfile.electrode_groups['Tetrode'+pd_series.TetrodeGroup]
            nwbfile.add_electrode(
                group = nwbfile.electrode_groups['Tetrode'+str(pd_series.TetrodeGroup)], 
                label = csci.split('.')[0],
                location=pd_series.BrainRegion
            )      
            electrode_counter += 1

        nwbfile.electrodes.to_dataframe()

        #%% NOW we work on adding our data. For LFPs, we store in ElectricalSeries object

        #TODO: MAYBE ADD a try statement for LFP 
        
        # create dynamic table
        all_table_region = nwbfile.create_electrode_table_region(
            region=list(range(electrode_counter)),  # reference row indices 0 to N-1
            description="all electrodes",
        )

        # now lets get our raw data into a new format
        print("This make take a few moments if working with a lot of CSC data...")
        csc_all = np.zeros(shape=(len(self.csc_data[self.csc_data_names[0]]),electrode_counter))
        self.csc_grouping_table.reset_index(inplace=True)
        counter = 0
        for csci in self.csc_grouping_table.Name:
            csc_all[:,counter]=self.csc_data[csci]
            counter += 1

        # estimate the sampling rate
        # TODO: consider removing or find a way to iclude this information somewhere in the NWB file
        #ts1 = nwbfile.epochs.to_dataframe()['start_time'][0] # get start time
        #ts2 = nwbfile.epochs.to_dataframe()['stop_time'][0] # get end time
        #temp_array = np.array(self.csc_times) # get csc times
        #array_start = np.where(temp_array==ts1)[0][0] # find index of csc start
        #array_end = np.where(temp_array==ts2)[0][0] # find index of csc end
        #numsamples = len(np.array(temp_array[array_start:array_end])) # get number of csc samples from start-end in one recording
        #fs = np.round(numsamples/(ts2-ts1))

        # add electrical series
        raw_electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=csc_all,
            timestamps = self.csc_times, # need timestamps
            electrodes=all_table_region,
            #starting_time=float(np.array(self.csc_times[0])),  # timestamp of the first sample in seconds relative to the session start time
            #rate=self.csc_data_fs[0],  # in Hz
        )
        nwbfile.add_acquisition(raw_electrical_series)

        #%% Add spiketimes

        # get unit IDs for grouping
        try:
            unit_ids = self.tt_grouping_table.Name.tolist()
            self.tt_grouping_table.set_index('Name', inplace=True) # set Name as the index
            #self.tt_grouping_table.reset_index(inplace=True)

            # add unit column if units are present
            nwbfile.add_unit_column(name="quality", description="sorting quality")

            unit_num = 0
            for i in unit_ids:
                #print(self.tt_data[i])
                tetrode_num = self.tt_grouping_table.loc[i].TetrodeGroup
                brain_reg = self.tt_grouping_table.loc[i].BrainRegion
                for clusti in self.tt_data[i]:
                    #print(i+' '+clusti)
                    #clust_id = i.split('.ntt')[0]+'_'+clusti
                    nwbfile.add_unit(spike_times = self.tt_data[i][clusti],
                                    electrode_group = nwbfile.electrode_groups['Tetrode'+str(tetrode_num)], 
                                    quality = "good",
                                    id = unit_num)
                    unit_num += 1
            nwbfile.units.to_dataframe()
        except:
            print("Units not added to NWB file")
            pass

        # -- Save NWB file -- #
        save_nwb(folder_path=self.folder_path, nwb_file=nwbfile)
        validate_nwb(nwbpath = os.path.join(self.folder_path,'nwbfile.nwb'))
        
        return nwbfile, device

# miniscope data
class read_miniscope(base):

    def write_nwb(self):

        """
        This code converts recorded data from the UCLA miniscope to the NWB format.
        As long as your separate folders with behavior, camera tracking, miniscope, and experiment details
        have the names 'behavior', 'camera', 'experiment', and 'miniscope', this code works.

        John Stout
        """

        # TODO: dir will be the only input to this code
        dir = self.folder_path
        dir_contents = sorted(os.listdir(dir))

        # Directly accessible information
        folder_metaData = json.load(open(os.path.join(dir,'metaData.json')))
        try:
            folder_notes = pd.read_csv(os.path.join(dir,'notes.csv'))
        except:
            pass

        # TODO: Build in a more generalizable way to extract .csv and .json files. 

        # behavior
        behavior_id = [i for i in dir_contents if 'behavior' in i][0]
        behavior_dir = os.path.join(dir,behavior_id)
        behavior_metaData = json.load(open(os.path.join(behavior_dir,'metaData.json')))
        behavior_pose = pd.read_csv(os.path.join(behavior_dir,'pose.csv'))

        # cameraDevice
        camera_id = [i for i in dir_contents if 'camera' in i][0]
        camera_dir = os.path.join(dir,camera_id)
        camera_metaData = json.load(open(os.path.join(camera_dir,'metaData.json')))
        camera_times = pd.read_csv(os.path.join(camera_dir,'timeStamps.csv'))

        # miniscopeDevice - where the miniscope data is located - use this to identify miniscope file name
        miniscope_id = [i for i in dir_contents if 'miniscope' in i][0]
        miniscope_dir = os.path.join(dir,miniscope_id)
        miniscope_data = [i for i in sorted(os.listdir(miniscope_dir)) if '.avi' in i]
        miniscope_metaData = json.load(open(os.path.join(miniscope_dir,'metaData.json')))
        miniscope_times = pd.read_csv(os.path.join(miniscope_dir,'timeStamps.csv'))
        miniscope_head_orientation = pd.read_csv(os.path.join(miniscope_dir,'headOrientation.csv'))

        # experiment
        print("This version does not support the experiment folder due to no testing data")

        # %% Put data into NWB

        # use nwb_utils to extract the dataframe and fill it with the json data
        excel_dir, df = nwb_utils.nwb_to_excel_template(self.folder_path)

        # edit dataframe to resave
        df['experiment_description']=[folder_metaData['experimentName']]
        df['experimenter name(s)']=[str(folder_metaData['researcherName'])]
        df['institution']=['Nemours']
        df['lab_name']=['Hernan']
        df['subject_id']=[folder_metaData['animalName']]
        df['recording_device_name']=[miniscope_metaData['deviceType']]
        df['recording_device_description']=["UCLA Miniscope v4.4"]
        df['recording_device_manufacturer']=["Open Ephys"]
        df['session_id'] = [folder_metaData['baseDirectory'].split(self.slash)[-1]]
        df['virus_injected']=[[]]
        df['virus_brain_targets']=[[]]
        df.to_excel(excel_dir)
        excel_dir, df = nwb_utils.pandas_excel_interactive(self.folder_path,df=df,save_name='nwb_template.xlsx')

        # year, month, day, hour, minute, second
        time_data = folder_metaData['recordingStartTime']
        rec_time = datetime(time_data['year'],time_data['month'],time_data['day'],
                            time_data['hour'],time_data['minute'],time_data['second'],
                            time_data['msec'],tzinfo=tz.tzlocal())

        # creating the NWBFile
        print("This file does not handle multiple custom entries")
        nwbfile, device = nwb_utils.template_to_nwb(template_dir = excel_dir)

        optical_channel = OpticalChannel(
            name="OpticalChannel",
            description="an optical channel",
            emission_lambda=500.0, # NOT SURE HOW I FIND THIS
        )

        imaging_plane = nwbfile.create_imaging_plane(
            name="ImagingPlane",
            optical_channel=optical_channel,
            imaging_rate=float(miniscope_metaData['frameRate']),
            description="Calcium Imaging",
            device=device,
            excitation_lambda=600.0, # WHERE DO I FIND THIS??
            indicator=df['virus_injected'].values[0],
            location=df['virus_brain_targets'].values[0],
        )

        # save the nwb file
        nwbpath = os.path.join(dir,"nwbfile.nwb")
        self.nwbpath = nwbpath
        self.history = 'nwbpath: directory of nwbfile'
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

            # validate file
            validate_nwb(nwbpath=nwbpath)            
            del movie_mat, movie_data, nwbfile, one_p_series


        # confirmed !!!
        # read to check NWB file
        #with NWBHDF5IO(nwbpath, "r+") as io:
        #    print("Reading nwbfile from: ",nwbpath)
        #    nwbfile = io.read()
        #    tester = nwbfile.acquisition['recording1'].data[:]

# pinnacle data
class read_pinnacle(base):

    #print("PINNACLE CODE DOES NOT ALIGN TIMES TO MOVIE")    
    def write_nwb(self, edf_file = None):

        # reassign dir
        dir = self.folder_path

        # nwb_table
        nwb_table = pd.DataFrame()
        nwb_table['experimenter_name(s)']=[]
        nwb_table['experiment_description']=[]
        nwb_table['session_description']=[]
        nwb_table['institution']=[]
        nwb_table['lab_name']=[]
        excel_dir, nwb_table = nwb_utils.pandas_excel_interactive(dir = dir, df = nwb_table, save_name = 'nwb_template.xlsx')

        # test if .edf is in the directory already
        if edf_file is None:
            # automation begins...
            dir_contents = sorted(os.listdir(dir))
            edf_names = [i for i in dir_contents if '.edf' in i]
        else:
            edf_names = [edf_file]

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
            excel_dir, data_table = nwb_utils.pandas_excel_interactive(dir = dir, df = data_table)

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
                sess_id = edfi.split('.')[0]
                nwbfile = NWBFile(
                    # experiment details
                    experimenter = list(nwb_table['experimenter_name(s)']),
                    experiment_description=str(nwb_table['experiment_description'].values[0]),
                    lab=str(nwb_table['lab_name'].values[0]),
                    institution=str(nwb_table['institution'].values[0]),

                    # session details
                    session_id=str(sess_id),
                    session_description=str(nwb_table['session_description'].values[0]),
                    identifier=str(uuid4()),
                    session_start_time = datetime_str,
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

                # Accounting for duplicate NWB files
                nwb_name = os.path.join(subject_id+'_'+sess_id+'.nwb')
                dir_save = os.path.join(dir,nwb_name)
                duplicate_name = [i for i in self.dir_contents if nwb_name in i]
                if nwb_name in duplicate_name:
                    print("You already have an NWB file with this name, adding '_new' extension")                
                    nwb_name = nwb_name.split('.nwb')[0]+'_new'+'.nwb'
                    dir_save = os.path.join(dir,nwb_name)
                    
                print("Writing .nwb file as ",dir_save)
                with NWBHDF5IO(dir_save, "w") as io:
                    io.write(nwbfile)    
                    io.close()

                # validate file
                validate_nwb(nwbpath=dir_save)

#%%  some general helper functions for nwb stuff

def validate_nwb(nwbpath: str):
        val_out = validate(paths=[nwbpath], verbose=True)
        print("NWB validation may be incorrect. Still need an invalid NWB file to check against....10/10/2023")
        if val_out[1]==0:
            print("No errors detected in NWB file")
        else:
            print("Error detected in NWB file")

def load_nwb(nwbpath: str):
    """
        Read NWB files

        Args:
            nwbpath: path directly to the nwb file
    """
    io = NWBHDF5IO(nwbpath, mode="r")
    nwb_file = io.read()

    return nwb_file

def save_nwb(folder_path: str, data_name: str = 'nwbfile.nwb', nwb_file=None):
    """
        Write NWB files. Separated for the purposes of flexible saving

        Args:
            folder_name: location of data
            data_name (OPTIONAL): name of nwb file
            nwb_file: nwb file type
    """

    with NWBHDF5IO(os.path.join(folder_path,data_name), "w") as io:
        io.write(nwb_file)

    print("Save .nwb file to: ",os.path.join(folder_path,data_name))

def read_edf(dir: str):
    '''
    Reads .edf files
        Args:
            >>> dir: directory with the .edf extension

        Returns:
            signals: signal data
            signal_headers: header files
    '''
    # this will be a loop
    dir_edf = os.path.join(dir)
    signals, signal_headers, header = highlevel.read_edf(dir)    

# TODO: This could be its own function
def read_movie(movie_name: str):

    """
    Args:
        >>> movie_name: name of the movie to load with extension

    John Stout
    """

    # read movie file
    movie_path = os.path.join(self.folder_path, movie_name)
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

def dsearchn(x, v):
    z=np.atleast_2d(x)-np.atleast_2d(v).T
    return np.where(np.abs(z).T==np.abs(z).min(axis=1))[0]

def sortXbyY(x,y):
    '''
    Args:
        x: list to sort
        y: list to sort by
    '''
    y_new, x_new = zip(*sorted(zip(y, x)))

    return list(x_new), list(y_new)
