# spectral.py
# 
# Group of spectral based functions for LFP analysis
#
# Written by John Stout

from scipy.signal import welch
from decode_lab_code.readers.ioreaders import load_nwb
import matplotlib.pyplot as plt
import numpy as np
from decode_lab_code.core.base import base
from decode_lab_code.readers.ioreaders import read_nlx
import pandas as pd

# run below in interactive window for troubleshooting
"""
    All inputs will be:
        >>> data
        >>> fs
        >>> frequency_range
"""

print("Cite Pynapple and PYNWB")

def filt():
    pass

def power(data: list, fs: float, frequency_range: list = [1,100]):

    """
    Run welch's power analysis

    Requires pynapple based loading

    Args:
        >>> channel_name: name of the CSC channel you want to use
        >>> start_time: start time for your analysis (accepts lists of time)
        >>> end_time: list of times to index to. MUST MATCH START_TIME SIZE/SHAPE
    """

    PSpec = [] PSpecLog = [] 
    for i in range(len(data)):

        # power spectrum
        f,Ptemp = welch(data[i],fs,nperseg=fs)

        # restrict data to 1-50hz for plot proofing
        #f[f>1]
        idxspec = np.where((f>frequency_range[0]) & (f<frequency_range[1]))
        fSpec = f[idxspec]
        PSpec = Ptemp[idxspec]

        # log10 transform
        PSpecLog = np.log10(PSpec)

    return PSpec, PSpecLog

# TODO
def coherence():
    pass

def spikefield_coherence():
    pass

def spikephase_entrainment():
    pass

def spikephase_precession():
    pass




    
    





