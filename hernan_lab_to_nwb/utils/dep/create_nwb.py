# this code will create an NWB file for us to work with and we can then populate with ephys or ophys data
# You must have the neuroconv_envir
#
# Unlike the "calciumVideo2nwb" script, this is suppose to be object oriented and allow for the conversion
# various kinds of datasets
#

from datetime import datetime

from uuid import uuid4
import numpy as np
from dateutil import tz

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

# added dependency
#from decode_lab_code.utils.util_funs import find

class nwbfile:

    def __init__(self):

        session_start_time = datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz("US/Pacific"))

        # initialize the nwbfile
        self = NWBFile(
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
        self.subject = Subject(
            subject_id=input("Enter subject ID: "),
            age=input("Enter subject age as such (PD100):  "),
            description=input("Enter subject identifier: "),
            species=input("Enter species name: "),
            sex=input("Enter sex of subject: "),
        )
    
    # if working with ophys data, we need to get caiman functions online
    def add_ophys(self):

        """
        This function relies on the following package:
            - caiman
        """

        envSet = input("Is your environment set to caiman? [Y/N]")
        if envSet == 'y' or envSet == 'Y':
            # lets load in our caiman wrapper
            from decode_lab_code.preprocessing.caiman_wrapper import caiman_preprocess
            
            # directory information
            folder_name = input("Enter the folder name for your data: ")
            fname_neuron = input("Enter file name with extension: ")
            frame_rate = float(input("Enter the frame rate: "))

            # if you get the Error: "Exception: A cluster is already runnning", restart the kernel
            cp = caiman_preprocess(folder_name,fname_neuron,frame_rate,False)

            # lets load in 
            data = cp.get_frames()

            time_series_with_rate = TimeSeries(
                name="ophys",
                data=data,
                unit="pixels",
                starting_time=0.0,
                # I'm not sure if this is numsamples/sec or sec/numsamples
                rate=frame_rate, # sampled every second (make sure this is correct***)
            )
            time_series_with_rate
            self.add_acquisition(time_series_with_rate)

    def add_ephys(self, folder_name: str, brain_areas: list):
        
        from spikeinterface import extractors as ex

        """
        This code supports extraction and grouping of data collected with tetrode arrays
        --
        This function relies on the following package:
            - spikeinterface
            - FUTURE: recording_array: 'tetrode'/'probe'
        --INPUTS--
            folder_name: string object for directory
            brain_areas: list:
                    brain_areas = ['PFC', 'HPC']

        """
        # ----- #

        # First, lets extract the data of interest

        # import all .ncs files
        data = ex.read_neuralynx(folder_path=folder_name)
        # get channel names
        channel_ids = data.get_property('channel_name')
        # create a temp variable to add to nwb
        groupNames = []
        for chi in channel_ids:
            if 'CSC' in chi:
                groupNames.append(chi)
            elif 'TT' in chi:
                groupNames.append("Tetrode"+chi[2])
            else:
                from warnings import warn
                warn("Not a recognized channel")
        # create a new property called "group_name" to group CSC and TTs
        data.set_property(key="group_name",values=np.array(groupNames))

        # ---- #
        
        # next lets interact with the user to identify brain region
        brain_area_string = []
        brain_area_index  = []
        for i in range(len(brain_areas)):
            brain_area_string.append(input("Which wires belong to "+brain_areas[i]))
            try: 
                brain_area_index.append(brain_area_string[i].split(' '))
            except:
                brain_area_index.append(brain_area_string[i].split(','))

        # ---- #

        # next, lets add to the NWB file
        device = self.create_device(
            name=input("What kind of recording device? "), 
            description=input("Describe this array (e.g. 32ch tetrode array): "), 
            manufacturer=input("Enter manufacturer: ")
        )

        # automatically identify the number of tetrodes
        ttNames = []
        for chi in channel_ids:
            if 'TT' in chi:
                ttNames.append(chi[2])
        ttNames = np.unique(np.array(ttNames))

        # csc
        cscNames = []
        for chi in channel_ids:
            if 'CSC' in chi:
                cscNames.append(chi[3])
        cscNames = np.unique(np.array(cscNames))        

        # add an electrode column to format the CSC and TT data
        self.add_electrode_column(name='label', 
                                  description="label of electrode")

        # add wires
        for csci in range(len(cscNames)):

            # identify which csc belong to which brain structure
            for bi in range(len(brain_area_index)):
                if str(csci+1) in brain_area_index[bi]:
                    location = brain_areas[bi]

            electrode_group = self.create_electrode_group(
                name='CSC'+cscNames[csci],
                description='Continuously sampled channel (LFP data)',
                device=device,
                location=location,
            )   

            # add electrodes to the electrode table
            electrode_counter = 0
            nchannels = 1 # 4 wires per tetrode
            self.add_electrode(
                group=electrode_group,
                label="CSC{}channel{}",
                location=location,
            )          

        # add CSC data
        for tti in range(len(ttNames)):

            # identify which csc belong to which brain structure
            for bi in range(len(brain_area_index)):
                if str(tti+1) in brain_area_index[bi]:
                    location = brain_areas[bi]

            # create an electrode group
            electrode_group = self.create_electrode_group(
                name='Tetrode'+ttNames[tti],
                description='Tetrodes for spike extraction',
                device=device,
                location=location,
            )  

            # add electrodes to the electrode table
            electrode_counter = 0
            nchannels = 4 # 4 wires per tetrode
            for ielec in range(nchannels):
                self.add_electrode(
                    group=electrode_group,
                    label="Tetrode{}channel{}".format(tti, ielec),
                    location=location,
                )
                electrode_counter += 1                 

        # ---- #
        # next, lets add our data to the nwb file

        # ---- # 
        # next, lets filter and downsample our LFP data to add

        # ---- # 
        # next, lets add our clustered data
