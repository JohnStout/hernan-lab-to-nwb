## Script that uses spikeinterface to extract Neuralynx data for DECODE lab
#
# Do the following:
#   conda create -n neuroconv python=3.11.0
#   conda activate neuroconv
#   pip install neuroconv
#   cd decode_lab_code
#   pip install -e .
#
# Relies on the nlx2nwb module
#
# The user will enter a list of directories or a master directory and
# this code will convert all data

#
# CURRENT ISSUE:
#   THe spikeinterface code does not love the xplorefinder.m headers. need to ignore these files.

import spikeinterface as si
import matplotlib.pyplot as plt
import spikeinterface.extractors as ext

"""
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
"""

# read neuralynx data
#datafolder = '/Users/js0403/Sample-data'
datafolder = '/Users/js0403/local data/2021-02-04_11-19-07_14eB_control_extinction_animal_1'
#datafolder = '/Users/js0403/local data/tester/noreads'
neura_data = ext.read_neuralynx(folder_path = datafolder,
neura_data.get_channel_ids()
neura_data.get_neo_io_reader

streams = ext.read_neuralynx.get_streams(folder_path=datafolder)
ids = ext.read_neuralynx.get(folder_path=datafolder)

neo_kwargs=ext.read_neuralynx.map_to_neo_kwargs(folder_path=datafolder)






