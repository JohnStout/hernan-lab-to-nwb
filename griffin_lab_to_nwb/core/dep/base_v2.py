## Base functions
#
# This will serve as a group of functions to organize data storage, preprocessing,
# wrangling and analysis
#
# written by John Stout

import os
import numpy as np

# this class is for putting directories into the pipeline
class base_converter():

    """
    A class of methods that will serve as integral for objects in the wrangle folder
    
    """


    def __init__(self, folder_path = None, data = None):

        """
        Args:
            You must either specify folder_path OR data

            folder_path: Directory of dataset
            data: dataset of interest
            
        """

        # a tracker variable
        self.history = []
        
        # if folder_path is provided
        if type(folder_path) is str and folder_path is not None and data is None:
            
            # define a folder path
            self.folder_path = folder_path
            self.history.append("folder_path: directory of data - added")

            # assign session ID according to your folder path
            #folder_path = '/Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS'
            self.session_id = self.folder_path.split('/')[len(self.folder_path.split('/'))-1]
            self.history.append("session_id: session identification variables - added")

            # get list of contents in the directory
            self.dir_contents = sorted(os.listdir(self.folder_path))
            self.history.append("dir_contents: the contents in the current directory - added")

        # if data is provided        
        elif type(data) is not None and folder_path is None:
            
            # tracking signal processing steps
            self.history = [] # list
            self.data = dict()

            # make sure we don't work with wrong kind of data
            if type(data) is dict:
                csc_ids = data.keys()
                for csci in csc_ids:
                    if len(data[csci].shape) > 1:
                        self.data[csci] = np.reshape(data[csci],len(data[csci]))
                self.history.append("Reshaped data arrays to 1D")

            elif type(data) is np.array:

                # TODO: convert to dict
                data = dict(enumerate(array.flatten(), 1))
                
                # check the shape of the data and fix as needed
                if len(data.shape) > 1:

                    # you have to reshape if you want any filtering to actually work
                    self.data = np.reshape(data,len(data))

                    # store a history of what you've done
                    self.history.append("Reshaped data array to 1D array")

                else:
                    self.data = data

            # store sampling rate
            self.fs = fs






# this class is for when putting data into the pipeline
class base_preprocess():

    """
    A general group of methods to be inherited for preprocessing purposes.

    This code will be inherited by preprocessing. 

    TODO: Add an exception for NWB data types, where it converts that nwb file to dict

    # TODO: Should this inherit the .history approach like above?. Might not be able to bc
    folder_path is not available

    
    """

    def __init__(self, data, fs: int):

        """

        Accepts inputs of numpy arrays or dictionaries.

        Preferred input is a data that is a dictionary array

        Args:
            data: numpy array vector. Will be reshaped as needed.
            fs: sampling rate as an integer

            TODO: implement a way to detect if input is a list of signals
        """

        # tracking signal processing steps
        self.history = [] # list
        self.data = dict()

        # make sure we don't work with wrong kind of data
        if type(data) is dict:
            csc_ids = data.keys()
            for csci in csc_ids:
                if len(data[csci].shape) > 1:
                    self.data[csci] = np.reshape(data[csci],len(data[csci]))
            self.history.append("Reshaped data arrays to 1D")

        # if just working with a single array, convert to dict
        if type(data) is np.array:
            # TODO: Make this compatible with dictionary type processing
            


            # check the shape of the data and fix as needed
            if len(data.shape) > 1:

                # you have to reshape if you want any filtering to actually work
                self.data = np.reshape(data,len(data))

                # store a history of what you've done
                self.history.append("Reshaped data array to 1D array")

            else:
                self.data = data

        # store sampling rate
        self.fs = fs