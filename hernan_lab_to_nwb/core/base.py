## Base functions
#
# This will serve as a group of functions to organize data storage, preprocessing,
# wrangling and analysis
#
# written by John Stout

import os
import platform

# this class is for putting directories into the pipeline
class base():

    """
    A class of methods that will serve as integral for objects in the converters folder
    
    """

    def __init__(self, folder_path:str):

        """
        Args:
            You must either specify folder_path OR data

            folder_path: Directory of dataset
            data: dataset of interest
            fs: required if data is specified

        """

        # a tracker variable
        self.history = []

        # check operating system        
        if 'Darwin' in platform.system():
            self.slash = '/'
            print("OS Mac detected")
        elif 'Windows' in platform.system():
            self.slash = '\\'
            print("OS windows detected")

        # define a folder path
        self.folder_path = folder_path
        self.history.append("folder_path: directory of data - added")

        # assign session ID according to your folder path
        #folder_path = '/Users/js0403/local data/2020-06-26_16-56-10 9&10eb male ACTH ELS'
        self.session_id = self.folder_path.split(self.slash)[len(self.folder_path.split(self.slash))-1]
        self.history.append("session_id: session identification variables - added")

        # get list of contents in the directory
        self.dir_contents = sorted(os.listdir(self.folder_path))
        self.history.append("dir_contents: the contents in the current directory - added")


    