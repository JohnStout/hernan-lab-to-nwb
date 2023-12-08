#% terminal script to convert data
#
# John Stout

# load packages
from hernan_lab_to_nwb.readers import ioreaders
from hernan_lab_to_nwb.utils import nwb_utils

class convert():

    def __init__(self, dir: str):
        """
        Args:
            >>> dir: directory containing all of your data.
                >>> neuralynx: this would be the file with all of the .ncs and .ntt files
                >>> miniscope: this would be the main folder ".../12-1-2070/"
        """

        # save directory
        self.dir = dir

    def neuralynx(self):

        # Read data
        ephys_object = ioreaders.read_nlx(folder_path = self.dir) # we first instantiate the object
        ephys_object.read_ephys() # read data

        # Edit pandas array dynamically
        ephys_object.csc_grouping_table = nwb_utils.pandas_excel_interactive(dir = ephys_object.folder_path, df = ephys_object.csc_grouping_table)
        ephys_object.tt_grouping_table = nwb_utils.pandas_excel_interactive(dir = ephys_object.folder_path, df = ephys_object.tt_grouping_table)

        # write file
        ephys_object.write_nwb()

    def miniscope(self):

        # instatiate object with directory
        ophys_object = ioreaders.read_miniscope(folder_path = self.dir)

        # write to NWB - this is recommended rather than separately visualizing the datasets
        ophys_object.write_nwb()

    def pinnacle(self):

        # instatiate object with directory
        eeg_object = ioreaders.read_pinnacle(folder_path = self.folder_path)

        # write to NWB - this is recommended rather than separately visualizing the datasets
        eeg_object.write_nwb()