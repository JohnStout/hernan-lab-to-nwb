# This code spatially downsamples the slice ophys recordings
#
# written by John Stout

# importing packages/modules
from decode_lab_code.preprocessing.caiman_wrapper import caiman_preprocess
import glob

# assign directory to downsample and convert data
folder_name = '/Users/js0403/ophysdata/Akanksha_data'
extension = '.avi' # define the extension

# frame_rate (define me)
frame_rate = 30

pathnames = glob.glob(folder_name+'/*'+extension)
pathnames.sort()

# need to check that this works, i changed downsample code recently
for pathi in pathnames:
    # split into directory and filename
    pathsplit = pathi.split('/') # split by / delimeter
    file_name = pathsplit[len(pathsplit)-1] # get file_name
    pathsplit.pop(len(pathsplit)-1) # remove the file_name
    folder_name = '/'.join(pathsplit) # folder_name directory

    # load in dataset
    cp = caiman_preprocess(folder_name,file_name,frame_rate,activate_cluster=False)
    cp.spatial_downsample(downsample_factor=2) # downsample
    cp.temporal_downsample(downsample_factor=2) # cut frame_rate in half
    cp.save_output() # save
    del cp # delete