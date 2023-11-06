## Script that uses spikeinterface to extract Neuralynx data for DECODE lab
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
datafolder = '/Volumes/decode/M1DataforPatrick/fearExt/Fear Clustering Review/2021-02-04_15-39-53 9ebk ACTH extinction animal 6'
neura_data = ext.read_neuralynx(folder_path = datafolder)




