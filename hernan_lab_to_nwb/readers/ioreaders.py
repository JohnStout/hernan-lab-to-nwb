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
#from pyedflib import highlevel

# from utils
from hernan_lab_to_nwb.utils import nlxhelper

# inheritance - base __init__ inherited
from hernan_lab_to_nwb.core.base import base

# import utilities for nwb 
from hernan_lab_to_nwb.utils import nwb_utils

# Add a second import that imports nlx packages??
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
        self.csc_data = dict(); self.tt_data = dict(); self.csc_data_fs = dict();
        csc_added = False; tt_added = False
        for groupi in dict_keys: # grouping variable to get TT data
            print("Working with",groupi)
            for datai in neural_dict[groupi]: # now we can get data

                # read data using Neo's NeuralynxIO
                if 'blks' in locals():
                    del blks
                blks = NeuralynxIO(filename=self.folder_path+'/'+datai, keep_original_times=True).read(lazy=False) # blocks
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
                        counter+=1

                    # restrict CSC data to these times
                    temp_csc = []; temp_times = []; csc_fs = []
                    for segi in range(len(blk.segments)):
                        temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                        temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())

                    # now restrict CSC data and times to be within event_times
                    for i in range(len(start_times)):
                        # convert to numpy
                        temp_times[i]=np.array(temp_times[i])
                        temp_csc[i]=np.array(temp_csc[i])
                        # get indices of start/stop times based on events file
                        idx_start = np.where(temp_times[i]==start_times[i])[0][0]
                        idx_end = np.where(temp_times[i]==end_times[i])[0][0]
                        # get data in between - not sure why I need to add 1 to the end
                        temp_csc[i] = temp_csc[i][idx_start:idx_end+1]
                        temp_times[i] = temp_times[i][idx_start:idx_end+1]

                    # horizontally stack data
                    self.csc_data[datai] = np.hstack(temp_csc)
                    self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient                        

                    # add sampling rate if available

                    #TODO: add fs for each csc channel and segment!!!
                    temp_fs = str(blk.segments[0].analogsignals[0].sampling_rate)
                    self.csc_data_fs[datai] = csc_fs
                    csc_added = True

                # Notice that all values are duplicated. This is because tetrodes record the same spike times.
                # It is the amplitude that varies, which is not captured here, despite calling magnitude.
                elif 'TT' in groupi: # .ntt TT files with spike data

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
        if len(blks) > 1:
            TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

        # get blocked data
        blk = blks[0]     

        # loop over blocked data if multiple blocks exist
        event_strings = []; event_times = []
        if len(blk.segments) > 1:
            TypeError("CODE DOES NOT HANDLE MULTIPLE BLOCKS - FIX")
        else:
            event_dict = blk.segments[0].events[0].__dict__
            event_strings = event_dict['_labels']
            event_times = blk.segments[0].events[0].times.magnitude

        self.event_strings = event_strings
        self.event_times = event_times
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

    def write_nwb(self, template_dir: str):

        """
        All .ncs files will be taken in

        """

        # TODO: Add preprocessing module to the NWB file for times when the user does things like filters/rereferences

        # make sure you have the header
        self.read_header()
        datetime_str = self.header['recording_opened']

        # TODO: THIS IS A GENERAL FUNCTION THAT will be supported by all write_nwb functions
        # create NWB template interactivately
        template_dir = nwb_utils.nwb_to_excel_template(self.folder_path)      
        print("nwb_template.xlsx written to", self.folder_path)
        input("Please fill in the nwb_template.xlsx sheet, then press any key to continue...")
        nwbfile, device = nwb_utils.template_to_nwb(template_dir = template_dir)

        #%% 

        # get start and stop times, according to the self.event_strings data
        times_start = [self.event_times[i] for i in range(len(self.event_strings)) if "start" in self.event_strings[i].lower()]
        times_stop  = [self.event_times[i] for i in range(len(self.event_strings)) if "stop" in self.event_strings[i].lower()]

        for i in range(len(times_start)):
            nwbfile.add_epoch(times_start[i], times_stop[i], ["rec"+str(i)])
        nwbfile.epochs.to_dataframe()

        #TODO: This will get more complicated for the fear dataset that had some pre-defined tone condition

        #%% before moving forward, remove any rows set to be removed in the pandas array
               
        # remove any rows of the pandas array if the inclusion is set to False
        rem_data_csc = self.csc_grouping_table['Inclusion'][self.csc_grouping_table['Inclusion']==False].index.tolist()
        rem_data_tt = self.tt_grouping_table['Inclusion'][self.tt_grouping_table['Inclusion']==False].index.tolist()

        print("Removing:\n",self.tt_grouping_table.iloc[rem_data_tt].Name,self.csc_grouping_table.iloc[rem_data_csc].Name)

        self.csc_grouping_table=self.csc_grouping_table.drop(index=rem_data_csc)
        self.tt_grouping_table=self.tt_grouping_table.drop(index=rem_data_tt)

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
        for bi in brain_regions: # loop over brain regions

            for ei in electrode_group: # loop over tetrode or probe

                # create an electrode group for a given tetrode
                electrode_group = nwbfile.create_electrode_group(
                    name='Tetrode{}'.format(ei),
                    description='Raw tetrode data',
                    device=device,
                    location=bi)         

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
            #rate=fs,  # in Hz
        )
        nwbfile.add_acquisition(raw_electrical_series)

        #%% Add spiketimes
        nwbfile.add_unit_column(name="quality", description="sorting quality")

        # get unit IDs for grouping
        unit_ids = self.tt_grouping_table.Name.tolist()
        self.tt_grouping_table.set_index('Name', inplace=True) # set Name as the index
        #self.tt_grouping_table.reset_index(inplace=True)

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

        # -- Save NWB file -- #
        save_nwb(folder_path=self.folder_path, nwb_file=nwbfile)
        val_out = validate(paths=[os.path.join(self.folder_path,'nwbfile.nwb')], verbose=True)
        
        print("NWB validation may be incorrect. Still need an invalid NWB file to check against....10/10/2023")
        if val_out[1]==0:
            print("No errors detected in NWB file")
        else:
            print("Error detected in NWB file")

# TODO: a class for reading optical physiology data
class read_ophys(base):

    # TODO: This could be its own function
    def read_movie(self, movie_name: str):

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
        
        # cache it
        self.movie_data = movie_data
        self.movie_name = movie_name
        self.history.append("movie_data: video recording from miniscope. Axis 0 = time, Axis 1 = Rows, Axis 2 = Cols, Elements = pixels")
        self.history.append("movie_name: video recording from miniscope. Axis 0 = time, Axis 1 = Rows, Axis 2 = Cols, Elements = pixels")

    def miniscope_to_nwb(self):

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
        folder_notes = pd.read_csv(os.path.join(dir,'notes.csv'))

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
        df['session_id'] = [folder_metaData['baseDirectory'].split('/')[-1]]
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

            del movie_mat, movie_data, nwbfile, one_p_series


        # confirmed !!!
        # read to check NWB file
        #with NWBHDF5IO(nwbpath, "r+") as io:
        #    print("Reading nwbfile from: ",nwbpath)
        #    nwbfile = io.read()
        #    tester = nwbfile.acquisition['recording1'].data[:]

class read_pinnacle(base):
    pass

#%%  some general helper functions for nwb stuff

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

    with NWBHDF5IO(folder_path+'/'+data_name, "w") as io:
        io.write(nwb_file)

    print("Save .nwb file to: ",folder_path+'/'+data_name)
