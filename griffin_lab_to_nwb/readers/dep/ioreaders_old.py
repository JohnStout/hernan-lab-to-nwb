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

# pynwb
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject

# numpy
import numpy as np

# loading neo package
from decode_lab_code.utils.neuralynxrawio import NeuralynxRawIO
from decode_lab_code.utils.neuralynxio import NeuralynxIO

# from utils
from decode_lab_code.utils import nlxhelper

# multiple inheritance - ephys gets its __init__ from base and gives it to read_nlx
from decode_lab_code.core.ephys import ephys_tools # this is a core base function to organize data
from decode_lab_code.core.base import base

print("Cite NWB")
print("Cite CatalystNeuro: NeuroConv toolbox if converting Neuralynx data")

# a specific class for unpacking neuralynx data
class read_nlx(ephys_tools):

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
        
        # Use Neo package
        print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

        # TODO: group data by file type, then separate by common naming conventions so that we never
        # have to worry about changing naming conventions

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
        self.csc_data = dict(); self.tt_data = dict()
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
                if 'CSC' in groupi: # CSC files referenced to a different channel
                    
                    if len(blk.segments) > 1:
                        blk_logger = ("Multiple blocks detected in "+datai+". LFP and LFP times have been collapsed into a single array.")
                        print(blk_logger)
                        temp_csc = []; temp_times = []
                        for segi in range(len(blk.segments)):
                            temp_csc.append(blk.segments[segi].analogsignals[0].magnitude.flatten())
                            temp_times.append(blk.segments[segi].analogsignals[0].times.flatten())
                        self.csc_data[datai] = np.hstack(temp_csc)
                        self.csc_times = np.hstack(temp_times) # only need to save one. TODO: make more efficient
                    else:                   
                        self.csc_data[datai] = blk.segments[0].analogsignals[0].magnitude.flatten()
                        self.csc_times = blk.segments[0].analogsignals[0].times.flatten()

                    # add sampling rate if available
                    if 'csc_fs' not in locals():
                        temp_fs = str(blk.segments[0].analogsignals[0].sampling_rate)
                        csc_fs = int(temp_fs.split('.')[0])
                        self.csc_data_fs = csc_fs
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
            self.history.append("LOGGER: csc_data had multiple blocks. This is likely due to multiple start/stops when recording. LFP and times were concatenated into a single array.")

        # get keys of dictionary
        if csc_added is True:
            self.csc_data_names = csc_names
            self.history.append("csc_data: CSC data as grouped by ext .ncs")
            self.history.append("csc_data_names: names of data in csc_data as organized by .ncs files")
            self.history.append("csc_data_fs: sampling rate for CSC data, defined by .ncs extension")

        if tt_added is True:
            self.tt_data_names = tt_names
            self.history.append("tt_data: Tetrode data as grouped by ext .ntt")
            self.history.append("tt_data_names: names of data in tt_data as organized by .ntt files")
            self.history.append("tt_data_fs: hard coded to 32kHz after not detected neo extraction of sampling rate")

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
        TODO: Read events information
        
        """

        pass

    def read_header(self):

        # attempt to read files until you get a header
        next = 0; looper = 0
        while next == 0:
            ncs_file = [i for i in self.dir_contents if '.ncs' in i.lower()][looper]
            reader = NeuralynxRawIO(filename = os.path.join(self.folder_path,self.dir_contents[looper]))
            reader.parse_header()
            file_header = reader.file_headers
            if bool(file_header) is False:
                looper = looper+1
            else:
                next = 1
            if looper == len(self.dir_contents)-1:
                raise ValueError('Could not extract information from header')

        # time information for NWB
        header_dict = dict(list(file_header.values())[0])
        #datetime_str = header_list['recording_opened']
        self.header = header_dict
        self.history.append("header: example header from the filepath of a .ncs file")

    def write_nwb(self, lfp_names = None, clustered_tt_names = None):

        """
        Args:
            lfp_names: default is none and therefore detected based on .ncs file types
            clustered_tt_names: default is none and therefore detected based on .ntt file endings

            RECOMMENDED to actually define these variables

        """

        # make sure you have the header
        self.read_header()
        datetime_str = self.header['recording_opened']

        # create NWB file
        nwbfile = NWBFile(
            session_description=input("Enter a brief discription of the experiment: "),
            identifier=str(uuid4()),
            session_start_time = datetime_str,
            experimenter = input("Enter the name(s) of the experimenter(s): "),
            lab="Hernan Lab",
            institution="Nemours Children's Hospital",
            session_id=self.session_id
        )

        # enter subject specific information
        subject = Subject(
                subject_id=input("Enter subject ID: "),
                age=input("Enter age of subject (PD): "),
                description=input("Enter notes on this mouse as needed: "),
                species=input("Enter species type (e.g. mus musculus (C57BL, etc...), Rattus rattus, homo sapiens): "),
                sex=input("Enter sex of subject: "),
            )

        nwbfile.subject = subject

        # add recording device information
        device = nwbfile.create_device(
            name="Cheetah", 
            description=input("Type of array? (e.g. tetrode/probe)"), 
            manufacturer="Neuralynx"
            )
        
        #%% Add electrode column and prepare for adding actual data

        # The first step to adding ephys data is to create categories for organization
        nwbfile.add_electrode_column(name='label', description="label of electrode")

        # add csc channels. Sometimes the lab uses CSC1-4 as TT1, sometimes we call it TTa-d.
        group_csc = 'into tetrode' # TODO: Make this as an input
        brain_area = 'PFC' # TODO: add brain_area as an input, but also make code flexible for multipl structures
        
        
        if 'tetrode' in group_csc:
            group_csc_to_tt = int(len(self.csc_data)/4) # grouped tts
            idx = [1,2,3,4] # index for tetrode assignment
            electrode_counter = 0 # used later
            for ei in range(group_csc_to_tt): # for loop over electrodes (ei)
                if ei > 0:
                    idx = [idxi+4 for idxi in idx] # index that changes according to tetrode assignment
                print(idx) # parsing error

                # create an electrode group for a given tetrode
                electrode_group = nwbfile.create_electrode_group(
                    name='Tetrode{}'.format(ei+1),
                    description='Raw tetrode data',
                    device=device,
                    location=brain_area)     

                # add each wire to the electrode group
                csc_names_use = []
                csc_names_use = [csc_names[idx[i]-1] for i in range(len(idx))]
                print(csc_names_use) # changes with loop

                for csci in csc_names_use:
                    nwbfile.add_electrode(
                        group = electrode_group,
                        label = csci.split('.')[0],
                        location=brain_area
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
        csc_all = np.zeros(shape=(len(csc_data[csc_names[0]]),electrode_counter))
        for csci in range(len(csc_data)):
            csc_all[:,csci] = np.reshape(csc_data[csc_names[csci]],len(csc_data[csc_names[csci]]))

        raw_electrical_series = ElectricalSeries(
            name="ElectricalSeries",
            data=csc_all,
            timestamps = csc_times, # need timestamps
            electrodes=all_table_region,
            #starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
            #rate=32000.0,  # in Hz
        )
        nwbfile.add_acquisition(raw_electrical_series)

        #%% Add spiketimes
        nwbfile.add_unit_column(name="quality", description="sorting quality")

        # get unit IDs for grouping
        unit_ids = tt_clust_data.keys()

        unit_num = 0
        for i in unit_ids:
            for clusti in tt_clust_data[i]:
                print(clusti)
                #clust_id = i.split('.ntt')[0]+'_'+clusti
                nwbfile.add_unit(spike_times = tt_clust_data[i][clusti],
                                quality = "good",
                                id = unit_num)
                unit_num += 1
        nwbfile.units.to_dataframe()

        #%% Save NWB file
        