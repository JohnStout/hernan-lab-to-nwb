# nlx helpers
#
# Set of helper functions to support the readers object

from typing import Dict, Union, List, Tuple
import numpy as np
import os

def read_nvt(filename: str) -> Union[Dict[str, np.ndarray], None]:
    """
    Reads a NeuroLynx NVT file and returns its data.

    Parameters
    ----------
    filename : str
        Path to the NVT file.

    Returns
    -------
    Union[Dict[str, np.ndarray], None]
        Dictionary containing the parsed data if file exists, None otherwise.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.


    Ben Dichtor wrote code
    """
    
    # Constants for header size and record format
    HEADER_SIZE = 16 * 1024
    RECORD_FORMAT = [
        ("swstx", "uint16"),
        ("swid", "uint16"),
        ("swdata_size", "uint16"),
        ("TimeStamp", "uint64"),
        ("dwPoints", "uint32", 400),
        ("sncrc", "int16"),
        ("Xloc", "int32"),
        ("Yloc", "int32"),
        ("Angle", "int32"),
        ("dntargets", "int32", 50),
    ]
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    
    # Reading and parsing data
    with open(filename, 'rb') as file:
        file.seek(HEADER_SIZE)
        dtype = np.dtype(RECORD_FORMAT)
        records = np.fromfile(file, dtype=dtype)
        return {name: records[name].squeeze() for name, *_ in RECORD_FORMAT}

def handle_missing_data(filename: str, missing_data = None):

    """
    Reads neuralynx NVT files and handles missing data

    TODO: add interpolation of missing data. Might be good as a method of a class

    Args:
        filename: directory of data with .nvt extension
        missing_data: str, option to handle missing data. Default = None.
                        Accepts: 
                            'NaN': makes 0 values nan

    Ben Dichtor wrote code. John Stout wrapped into function

    """

    # read data
    data = read_nvt(filename)

    # make 0s nan
    if missing_data == 'NaN':
        x = data["Xloc"].astype(float)
        x[data["Xloc"] <= 0] = np.nan

        y = data["Yloc"].astype(float)
        y[data["Yloc"] <= 0] = np.nan

        t = data["Tloc"].astype(float)

    return x,y,t
