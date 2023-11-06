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

import os

# work on deprecating neuroconv. No need
from neuroconv.datainterfaces import NeuralynxRecordingInterface # not sure why this variable is whited out

from pynwb import NWBHDF5IO, NWBFile
import numpy as np

from neo.io.neuralynxio import NeuralynxIO
#from neo.io.neuralynxio import NeuralynxRawIO

from decode_lab_code.utils.dep import read_vt_data

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

    # interface with the user
    #folder_path = input("Enter directory of recording session: ")
    experimenters = input("Enter the name(s) of the experimenter(s): ")
    experiment_description = input("Enter a brief discription of the experiment: ")
    session_notes = input("Enter notes pertaining to this session: ")  
    session_id = input("Enter an ID for the session: ")
    lfp_notes = input('Enter information about your LFP recordings: ')
    tt_notes = input('Enter information about your tetrode recordings: ')

    # Use Neo package
    print("Cite Neo https://github.com/NeuralEnsemble/python-neo/blob/master/CITATION.txt")

    # folder path
    folder_path = '/Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS'

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

    # TODO: Now lets get information about the data and put into NWB.


    # TODO: make fig with car and non car for fun
    import matplotlib.pyplot as plt
    import numpy as np
    raw_csc = [i for i in dir_contents if '.ncs' in i.lower() and 'car' not in i.lower()]

    # plot some examples
    fig, axes = plt.subplots(nrows=int(len(raw_csc)),ncols=1)

    fs = 32000
    for i in range(int(len(raw_csc))):
        temp_data = csc_data['CSC'+str(i+1)+'.ncs'][0:int(fs/2)]
        temp_car = csc_data['csc_car'+str(i+1)+'.ncs'][0:int(fs/2)]
        x_data = np.linspace(0,len(temp_data)/fs,len(temp_data))
        axes[i].plot(x_data,temp_data,'k',linewidth=1)
        axes[i].plot(x_data,temp_car,'r',linewidth=1)

    # check if save directory exists, if not, add one
    save_dir = os.path.join(folder_path,'figs')
    if os.path.exists(save_dir) is False:
        os.mkdir(os.path.join(folder_path,'figs'))
    save_fig = os.path.join(save_dir,'ex1')
    fig.savefig(save_fig, format='eps')

    # plot a fig on small timescale
    idx = [0,4000]
    plt.plot(x_data[idx[0]:idx[1]],temp_data[idx[0]:idx[1]],'k',linewidth=1)
    plt.plot(x_data[idx[0]:idx[1]],temp_car[idx[0]:idx[1]],'r',linewidth=1)
    save_fig = os.path.join(save_dir,'csc_end_zoom')
    plt.savefig(save_fig, format='eps')



    # TODO: make class for signal processing
    from decode_lab_code.preprocessing.signal_utils import process_signal

    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    sig1 = process_signal(data = temp_data)
    sig1.butterworth_filter(lowcut=500, highcut=9000, fs = fs)
    sig2 = process_signal(data = temp_car)
    sig2.butterworth_filter(lowcut=500, highcut=9000, fs=fs)
    x_data = np.linspace(0,len(temp_data)/fs,len(temp_data))

    axes[0].plot(x_data,sig1.data,'k',linewidth=1)
    axes[0].plot(x_data,sig2.data,'r',linewidth=1)
    axes[1].plot(x_data,sig1.signal_filtered,'k',linewidth=1)
    axes[1].plot(x_data,sig2.signal_filtered,'r',linewidth=1)
    save_fig = os.path.join(save_dir,'comp_filt')
    plt.savefig(save_fig, format='eps')






    # Get VT data from .NVT files
    vt_name = [i for i in dir_contents if '.nvt' in i.lower()][0]

    # get video tracking data if it's present
    filename = os.path.join(folder_path,vt_name)
    vt_data = read_vt_data.read_nvt(filename = filename)
    vt_x = vt_data['Xloc']
    vt_y = vt_data['Yloc']
    vt_t = vt_data['TimeStamp']

    # sampling rate
    sampling_rate = spikedata[0].get_signal_sampling_rate()
    
    # spike data
    reader.get_spike_raw_waveforms(spike_channel_index=0) # define the index

    # get information to document from the header files
    file_header = reader.file_headers
    file_header_keys = list(file_header.keys())
    datetime_str = file_header[file_header_keys[0]]["TimeCreated"]
    date_obj = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')

    nwbfile = NWBFile(
        session_description="my first synthetic recording",
        identifier=str(uuid4()),
        session_start_time=date_obj,
        experimenter = experimenters,
        lab="Hernan Lab",
        institution="Nemours Children's Hospital",
        experiment_description=experiment_description,
        session_id=session_id,
    )

    device = nwbfile.create_device(
        name="array", 
        description="the best array", 
        manufacturer="Probe Company 9000"
        )
    
    nwbfile.add_electrode_column(name='label', 
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








    global_metadata = {
        "session_start_time": date_obj,
        "identifier": data[0].file_origin,
        "session_id": file_header[file_header_keys[0]]['SessionUUID'],
        "institution": "Nemours",
        "lab": "Hernan",
        "related_publications": "NA"
    }

    signal_metadata = {
    "nwb_group": "acquisition",
    "nwb_neurodata_type": ("pynwb.icephys", "PatchClampSeries"),
    "nwb_electrode": {
        "name": "patch clamp electrode",
        "description": "The patch-clamp pipettes were pulled from borosilicate glass capillaries "
                        "(Hilgenberg, Malsfeld, Germany) and filled with intracellular solution "
                        "(K-gluconate based solution)",
        "device": {
           "name": "patch clamp electrode"
        }
    },
    "nwb:gain": 1.0
    }

for segment in data[0].segments:
    signal = segment.analogsignals[0]
    signal.annotate(**signal_metadata)

    for blk in data:
        for seg in blk.segments:
            print(seg)
            for asig in seg.analogsignals: # LFP
                print(asig)
            for st in seg.spiketrains: # spikes
                print(st)


    # Change the folder_path to the appropriate location in your system
    interface = NeuralynxRecordingInterface(folder_path=folder_path, verbose=False)

    # in the metadata below, change to "TT" and in the for loop, add a number for the tetrode!!

    # Extract what metadata we can from the source files
    metadata = interface.get_metadata() # here we should change them
    metadata['NWBFile']['session_description'] = session_description
    metadata['NWBFile']['notes'] = session_notes
    metadata['Ecephys']['ElectrodeGroup'][0]['name']='CSC'
    metadata['Ecephys']['ElectrodeGroup'][0]['description']="Continuous Sample Channel (CSC) = LFP |"+lfp_notes
    metadata['Ecephys']['ElectrodeGroup'].append(dict(name='Tetrode',description=tt_notes))
    
    # link the metadata electrode grouping with the actual variables, key = thing we're overrighted
    #channel_ids = interface.recording_extractor.get_channel_ids()
    channel_ids = interface.recording_extractor.get_property(key="channel_name")
    groupNames = []
    for chi in channel_ids:
        if 'CSC' in chi:
            groupNames.append(chi)
        elif 'TT' in chi:
            groupNames.append("Tetrode"+chi[2])
        else:
            from warnings import warn
            warn("Not a recognized channel")
    interface.recording_extractor.set_property(key="group_name",values=np.array(groupNames))

    # now I need to build in a processing nwb devices variable. This will be for TT_filtered


    # the last piece that I need is behavior. 
    # I should be able to create a module that reads the VT1 header


    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = folder_path+'/'+save_name  # This should be something like: "./saved_file.nwb"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
    print("NWB file created and saved to:",nwbfile_path) 






