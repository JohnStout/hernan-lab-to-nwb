# convertNLX2NWB
#
# This code is meant to take a full sessions recording and convert it to a single nwb file
#
# -- Dependencies/how to -- #
#
# This code depends on: https://neuroconv.readthedocs.io/en/main/index.html
# If you have never run this code, make sure that:
#   1) Open terminal
#   2) Download anaconda and git (for terminal), open terminal
#   3) git clone https://github.com/catalystneuro/neuroconv
#   4) cd neuroconv
#   5) conda env create -f make_environment.yml
#   6) conda activate neuroconv_environment
#
# If you are a returning user OR are new and have done the steps above:
#   1) right-click the play button and "run in interactive"
#   2) select the neuroconv_environment interpreter in the interactive window
#
# This code was adapted by catalyst neuro
#
# - written by JS 07/24/23 by adapting code from catalystNeuro

print("Cite NWB and CatalystNeuro")

# get some packages
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import NeuralynxRecordingInterface # not sure why this variable is whited out
from pynwb import NWBHDF5IO, NWBFile
import numpy as np


# function for saving nwb data - this is not so useful right now
def writeNWB(folder_path: str, save_name = 'data_nwb'):
    
    """
    Write an NWB file using ephys data collected with neuralynx acquisition system

    Args:
        folder_path: string input that defines the directory to extract data

    Optional Args:
        save_name: defaults to 'data_nwb', but you could redefine this
    
    """

    # interface with the user
    #folder_path = input("Enter directory of recording session: ")
    session_description = input("Enter a brief discription of the experiment: ")
    session_notes = input("Enter notes pertaining to this session: ")  
    lfp_notes = input('Enter information about your LFP recordings: ')
    tt_notes = input('Enter information about your tetrode recordings: ')

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
    
    # the last piece that I need is behavior

    # Choose a path for saving the nwb file and run the conversion
    nwbfile_path = folder_path+'/'+save_name  # This should be something like: "./saved_file.nwb"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
    print("NWB file created and saved to:",nwbfile_path) 

# function for reading nwb data
def readNWB(folder_path: str ,file_name: str):

    # Open the file in read mode "r", and specify the driver as "ros3" for S3 files
    #filePath = '/Users/js0403/Sample-data/data_nwb'
    file_path = folder_path+'/'+file_name
    io = NWBHDF5IO(file_path, mode="r")
    nwbfile = io.read()
    return nwbfile 

# add a component to test the NWB file for critical failures


#from neo import NeuralynxIO

#io=NeuralynxIO(dirname='/Users/js0403/Sample-data')
#io
# what methods th
#io.get_analogsignal_chunk()

#nvtOpen = open(folder_path+'/VT1.nvt','rb')
#nvtRead = nvtOpen.read()
#nvtRead[:]
#type(nvtOpen)
