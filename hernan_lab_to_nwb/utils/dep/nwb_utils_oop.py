# nwb_utils.py
# 
# This module is meant to act as an interface to read, write, and convert data into NWB ready formats
#
# Non-native dependencies:
#   neuroconv
#       pynwb (wrapped in neuroconv)
#
# To work if using decode_lab_code: 
#
#   1) Open terminal
#   2) conda create -n neuroconv python=3.11.0
#   3) pip install neuroconv
#   4) cd decode_lab_code
#   5) pip install -e .
#
#
# UPDATE: Use conversions
#   1) conda create -n conversion python=3.11.0
#   2) conda activate conversions
#   3) pip install neo
#   4) pip install pynwb
#   5) cd decode_lab_code
#   6) pip install -e .
# 
# Written by John Stout

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from uuid import uuid4

import re

import os

# work on deprecating neuroconv. No need
from neuroconv.datainterfaces import NeuralynxRecordingInterface # not sure why this variable is whited out

from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject

import numpy as np

from neo.io.neuralynxio import NeuralynxIO
from neo.io.neuralynxio import NeuralynxRawIO

from decode_lab_code.utils import read_vt_data

print("Cite NWB")
print("Cite CatalystNeuro: NeuroConv toolbox if converting Neuralynx data")

def read_nwb(folder_name: str, data_name: str):
    """
        Read NWB files

        Args:
            folder_name: location of data
            data_name: name of nwb file
    """
    
    io = NWBHDF5IO(folder_name+'/'+data_name, mode="r")
    nwb_file = io.read()

    return nwb_file

def write_nwb(folder_name: str, data_name = None, nwb_file=None):
    """
        Write NWB files

        Args:
            folder_name: location of data
            data_name (OPTIONAL): name of nwb file
            nwb_file: nwb file type
    """
    if data_name is None:
        data_name = "nwb_file"

    with NWBHDF5IO(folder_name+'/'+data_name, "w") as io:
        io.write(nwb_file)

    print("Wrote nwb_file to: ",folder_name+'/'+data_name)

def generate_nwb_sheet():
    """
    This function is meant to generate and save out a .txt or .doc file for the user to fill out

    It can then be loading into the nwb creation functions

    """

def load_decode_nlx(folder_path: str):
    
    # Use Neo package
    print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

    # assign session ID according to your folder path
    #folder_path = '/Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS'
    session_id = folder_path.split('/')[len(folder_path.split('/'))-1]

    # here's the hack. 
    # Loop over files of interest, 
    # sort according to clustered or non-clustered, 
    # load as such using Neo
    # organize into a dictionary
    # save outputs as components of the NWB file
    dir_contents = sorted(os.listdir(folder_path))

    # TODO: Add a line that ignores folders. Right now, if a folder was named "CSC", it'll be
    # grouped into this code and an error will spit out

    # group our data together
    csc_names = []; idx_rem = []; tt_names = []
    for ci in dir_contents:
        if 'csc' in ci.lower():
            csc_names.append(ci)
        elif 'tt' in ci.lower():
            tt_names.append(ci)
    
    # Separate the TT files into .ncs and .ntt types. They will be handled differently
    tt_ncs = []; tt_ntt = []
    for ti in tt_names:
        if 'ncs' in ti.lower():
            tt_ncs.append(ti)
        elif 'ntt' in ti.lower():
            tt_ntt.append(ti)
        
    # now we need to handle the processing aspect of the dataset. This is important for NWB generation
    tt_ntt_raw = []; tt_ntt_filt = []; tt_ntt_clust = []
    for ti in tt_ntt:
        if 'clust' in ti.lower():
            tt_ntt_clust.append(ti)
        elif 'filt' in ti.lower() and not 'clust' in ti.lower():
            tt_ntt_filt.append(ti)
        elif 'filt' != ti.lower() and 'clust' != ti.lower():
            tt_ntt_raw.append(ti)

    # sort files
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)',text) ]

    # sort
    csc_names.sort(key=natural_keys)
    tt_ncs.sort(key=natural_keys)
    tt_ntt_raw.sort(key=natural_keys)
    tt_ntt_filt.sort(key=natural_keys)
    tt_ntt_clust.sort(key=natural_keys)

    # now lets put these into a dict for working with in NeuralynxIO
    neural_dict = {'CSC': csc_names, 
                    'TT_CSC': tt_ncs, 
                    'TT_Raw': tt_ntt_raw,
                    'TT_Filt': tt_ntt_filt,
                    'TT_Clust': tt_ntt_clust}
    
    # Here we create separate dictionaries containing datasets with their corresponding labels
    dict_keys = neural_dict.keys()
    csc_data = dict(); tt_csc_data = dict(); tt_raw_data = dict()
    tt_filt_data = dict(); tt_clust_data = dict()
    for groupi in dict_keys: # grouping variable to get TT data
        print("Working with ",groupi)
        for datai in neural_dict[groupi]: # now we can get data

            # read data using Neo's NeuralynxIO
            if 'blks' in locals():
                del blks
            blks = NeuralynxIO(filename=folder_path+'/'+datai, keep_original_times=True).read(lazy=False) # blocks
            #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

            if len(blks) > 1:
                TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

            # get blocked data
            blk = blks[0]

            # organize data accordingly
            if 'CSC' in groupi and 'TT' not in groupi: # CSC files referenced to a different channel
                csc_data[datai] = blk.segments[0].analogsignals[0].magnitude
                csc_times = blk.segments[0].analogsignals[0].times

            elif 'TT_CSC' in groupi: # CSC files from Tetrodes referenced locally
                tt_csc_data[datai] = blk.segments[0].analogsignals[0].magnitude

            # Notice that all values are duplicated. This is because tetrodes record the same spike times.
            # It is the amplitude that varies, which is not captured here, despite calling magnitude.
            elif 'TT_Raw' in groupi: # .ntt TT files with spike data
                spikedata = blk.segments[0].spiketrains
                num_tts = len(spikedata)
                temp_dict = dict()
                for i in range(num_tts):
                    temp_dict['channel'+str(i)+'spiketimes'] = spikedata[i].magnitude
                tt_raw_data[datai] = temp_dict

            elif 'TT_Filt' in groupi: # filtered spike signals
                spikedata = blk.segments[0].spiketrains
                num_tts = len(spikedata)
                temp_dict = dict()
                for i in range(num_tts):
                    temp_dict['channel'+str(i)+'spiketimes'] = spikedata[i].magnitude
                tt_filt_data[datai] = temp_dict

            # This represents spike times of each cluster, per tetrode. So it looks duplicated.
            # Cross referenced these with clustered data. Ignore first cluster - it's noise.
            elif 'TT_Clust' in groupi: # clustered data (N spiketrains = N units)
                spikedata = blk.segments[0].spiketrains
                num_trains = len(spikedata) # there will be 4x the number of clusters extracted
                num_clust = int(num_trains/4) # /4 because there are 4 tetrodes
                temp_dict = dict()
                for i in range(num_clust):
                    if i > 0: # skip first cluster, it's noise
                        temp_dict['cluster'+str(i)+'spiketimes'] = spikedata[i].magnitude
                tt_clust_data[datai] = temp_dict

    return csc_data, csc_times, tt_csc_data

def nlx2nwb(folder_path: str, save_name = 'data_nwb', nwb_sheet_directory = None):
    """
    Write an NWB file using ephys data collected with neuralynx acquisition system

    ---------------------------------------
    Args:
        folder_path: string input that defines the directory to extract data

    Optional Args:
        save_name: defaults to 'data_nwb', but you could redefine this
        nwb_sheet_directory: location to load NWB sheet. THis is used to automatically populated the NWB file information
        
    Returns:
        Saves a .nwb file with the name following "save_name" argument. Default is 'data_nwb.nwb'

    ---------------------------------------
    STORING DATA:
        You should store your raw nlx files in one folder, and your other files in a "processed" folder.
        Inside that folder, you can specificy what was done to the files and store them hierarchically (e.g. filtered, clustered)

    -----------------------------------------
    # IMPORTANT:
        If you are running into weird errors that tell you to contact the developers, it just means the header
        isn't being read correctly. This is because the xplorefinder.m creates a strange headerfile that is missing elements
        Try pasting the following code in replacing line 246:end of __init__ in the following directory:
                neuroconv/lib/python3.11/site-packages/neo/rawio/neuralyxrawio/nlxheader.py


        # opening time - for the xplorefinder.m code, this is recognized as None, because it provides no info about date/time
        sr = re.search(hpd['datetime1_regex'], txt_header)
        if not sr:
            if av == Version('5.6.3'):
                print("Filling in missing datetime for Cheetah version: 5.6.3")
                #current_date = datetime.datetime.now()
                self['recording_opened'] = datetime.datetime.now().replace(microsecond=0)
            else:
                raise IOError("No matching header open date/time for application {} " +
                            "version {}. Please contact developers.".format(an, av))
        else:
            dt1 = sr.groupdict()
            self['recording_opened'] = datetime.datetime.strptime(
                dt1['date'] + ' ' + dt1['time'], hpd['datetimeformat'])
        print(self['recording_opened'])

        # close time, if available
        if 'datetime2_regex' in hpd:
            sr = re.search(hpd['datetime2_regex'], txt_header)
            if not sr:
                if av == Version('5.6.3'):
                    print("Filling in missing datetime for Cheetah version: 5.6.3")
                    self['recording_closed'] = datetime.datetime.now().replace(microsecond=0)
                else:
                    raise IOError("No matching header close date/time for application {} " +
                                "version {}. Please contact developers.".format(an, av))
            else:
                dt2 = sr.groupdict()
                self['recording_closed'] = datetime.datetime.strptime(
                    dt2['date'] + ' ' + dt2['time'], hpd['datetimeformat'])
    """

    # Use Neo package
    print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

    # assign session ID according to your folder path
    #folder_path = '/Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS'
    session_id = folder_path.split('/')[len(folder_path.split('/'))-1]

    # here's the hack. 
    # Loop over files of interest, 
    # sort according to clustered or non-clustered, 
    # load as such using Neo
    # organize into a dictionary
    # save outputs as components of the NWB file
    dir_contents = sorted(os.listdir(folder_path))

    # TODO: Add a line that ignores folders. Right now, if a folder was named "CSC", it'll be
    # grouped into this code and an error will spit out

    # group our data together
    csc_names = []; idx_rem = []; tt_names = []
    for ci in dir_contents:
        if 'csc' in ci.lower():
            csc_names.append(ci)
        elif 'tt' in ci.lower():
            tt_names.append(ci)
    
    # Separate the TT files into .ncs and .ntt types. They will be handled differently
    tt_ncs = []; tt_ntt = []
    for ti in tt_names:
        if 'ncs' in ti.lower():
            tt_ncs.append(ti)
        elif 'ntt' in ti.lower():
            tt_ntt.append(ti)
        
    # now we need to handle the processing aspect of the dataset. This is important for NWB generation
    tt_ntt_raw = []; tt_ntt_filt = []; tt_ntt_clust = []
    for ti in tt_ntt:
        if 'clust' in ti.lower():
            tt_ntt_clust.append(ti)
        elif 'filt' in ti.lower() and not 'clust' in ti.lower():
            tt_ntt_filt.append(ti)
        elif 'filt' != ti.lower() and 'clust' != ti.lower():
            tt_ntt_raw.append(ti)

    # sort files
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)',text) ]

    # sort
    csc_names.sort(key=natural_keys)
    tt_ncs.sort(key=natural_keys)
    tt_ntt_raw.sort(key=natural_keys)
    tt_ntt_filt.sort(key=natural_keys)
    tt_ntt_clust.sort(key=natural_keys)

    # now lets put these into a dict for working with in NeuralynxIO
    neural_dict = {'CSC': csc_names, 
                    'TT_CSC': tt_ncs, 
                    'TT_Raw': tt_ntt_raw,
                    'TT_Filt': tt_ntt_filt,
                    'TT_Clust': tt_ntt_clust}
    
    # Here we create separate dictionaries containing datasets with their corresponding labels
    dict_keys = neural_dict.keys()
    csc_data = dict(); tt_csc_data = dict(); tt_raw_data = dict()
    tt_filt_data = dict(); tt_clust_data = dict()
    for groupi in dict_keys: # grouping variable to get TT data
        print("Working with ",groupi)
        for datai in neural_dict[groupi]: # now we can get data

            # read data using Neo's NeuralynxIO
            if 'blks' in locals():
                del blks
            blks = NeuralynxIO(filename=folder_path+'/'+datai, keep_original_times=True).read(lazy=False) # blocks
            #blks = NeuralynxRawIO(filename =folder_path+'/'+datai).parse_header()

            if len(blks) > 1:
                TypeError("Blocks exceeding size 1. This indicates that multiple sessions detected. The following code will be terminated.")

            # get blocked data
            blk = blks[0]

            # organize data accordingly
            if 'CSC' in groupi and 'TT' not in groupi: # CSC files referenced to a different channel
                csc_data[datai] = blk.segments[0].analogsignals[0].magnitude
                csc_times = blk.segments[0].analogsignals[0].times

            elif 'TT_CSC' in groupi: # CSC files from Tetrodes referenced locally
                tt_csc_data[datai] = blk.segments[0].analogsignals[0].magnitude

            # Notice that all values are duplicated. This is because tetrodes record the same spike times.
            # It is the amplitude that varies, which is not captured here, despite calling magnitude.
            elif 'TT_Raw' in groupi: # .ntt TT files with spike data
                spikedata = blk.segments[0].spiketrains
                num_tts = len(spikedata)
                temp_dict = dict()
                for i in range(num_tts):
                    temp_dict['channel'+str(i)+'spiketimes'] = spikedata[i].magnitude
                tt_raw_data[datai] = temp_dict

            elif 'TT_Filt' in groupi: # filtered spike signals
                spikedata = blk.segments[0].spiketrains
                num_tts = len(spikedata)
                temp_dict = dict()
                for i in range(num_tts):
                    temp_dict['channel'+str(i)+'spiketimes'] = spikedata[i].magnitude
                tt_filt_data[datai] = temp_dict

            # This represents spike times of each cluster, per tetrode. So it looks duplicated.
            # Cross referenced these with clustered data. Ignore first cluster - it's noise.
            elif 'TT_Clust' in groupi: # clustered data (N spiketrains = N units)
                spikedata = blk.segments[0].spiketrains
                num_trains = len(spikedata) # there will be 4x the number of clusters extracted
                num_clust = int(num_trains/4) # /4 because there are 4 tetrodes
                temp_dict = dict()
                for i in range(num_clust):
                    if i > 0: # skip first cluster, it's noise
                        temp_dict['cluster'+str(i)+'spiketimes'] = spikedata[i].magnitude
                tt_clust_data[datai] = temp_dict

    #%% 

    # VT DATA

    # Get VT data from .NVT files
    vt_name = [i for i in dir_contents if '.nvt' in i.lower()][0]

    # get video tracking data if it's present
    filename = os.path.join(folder_path,vt_name)
    vt_data = read_vt_data.read_nvt(filename = filename)
    vt_x = vt_data['Xloc']
    vt_y = vt_data['Yloc']
    vt_t = vt_data['TimeStamp']

    #%%

    # Extract header information to document the NWB file

    # attempt to read files until you get a header
    next = 0; looper = 0
    while next == 0:
        ncs_file = [i for i in dir_contents if '.ncs' in i.lower()][looper]
        reader = NeuralynxRawIO(filename = os.path.join(folder_path,dir_contents[looper]))
        reader.parse_header()
        file_header = reader.file_headers
        if bool(file_header) is False:
            looper = looper+1
        else:
            next = 1
        if looper == len(dir_contents)-1:
            raise ValueError('Could not extract information from header')

    # time information for NWB
    header_list = list(file_header.values())[0]
    datetime_str = header_list['recording_opened']
    #date_obj = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')

    #%% 

    # interface with the user
    #folder_path = input("Enter directory of recording session: ")
    recording_notes = input("Enter information about your recording wires (e.g. TT1 wasn't working great today...)")

    # create NWB file
    nwbfile = NWBFile(
        session_description=input("Enter a brief discription of the experiment: "),
        identifier=str(uuid4()),
        session_start_time = datetime_str,
        experimenter = input("Enter the name(s) of the experimenter(s): "),
        lab="Hernan Lab",
        institution="Nemours Children's Hospital",
        session_id=session_id,
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
        description="Tetrode array", 
        manufacturer="Neuralynx recording system and self fabricated arrays"
        )
    
    #%% Add electrode column and prepare for adding actual data

    # The first step to adding ephys data is to create categories for organization
    nwbfile.add_electrode_column(name='label', description="label of electrode")

    # add csc channels. Sometimes the lab uses CSC1-4 as TT1, sometimes we call it TTa-d.
    group_csc = 'into tetrode' # TODO: Make this as an input
    brain_area = 'PFC' # TODO: add brain_area as an input, but also make code flexible for multipl structures
    
    if 'tetrode' in group_csc:
        group_csc_to_tt = int(len(csc_data)/4) # grouped tts
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

            # add unit column per tetrode
                   

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

    #%% Add behavioraly relevant times

