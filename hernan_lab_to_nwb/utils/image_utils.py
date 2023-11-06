# required packages
import numpy as np
import tifffile as tf
import glob
import tarfile
#import cv2

def stacktiff(dir: str, dir_save = None, downsample_factor = None):
    """
    This function takes a folder with a bunch of .tif images and stacks them

    Args:
        dir: directory containing image data to stack
        dir_save: OPTIONAL but recommended. Directory to save stacked data.
        downsample_factor: OPTIONAL.
            downsample_factor = 2 spatially reduces your dataset by a factor of 2
    """

    if dir_save is None:
        dir_save = dir

    # change me
    # dir = directory of data (change me)
    #dir = '/Volumes/decode/Akanksha/Slice_calcium_imaging_videos_images/Pilots/Static_recordings/08-31-2023/Tiff_series_Process_7'
    extension = '.tif' # no need to change
    mid_ext = '/*' # don't change

    # do stuff
    pathnames = glob.glob(dir+mid_ext+extension)
    pathnames.sort()

    # read in one image to get shape
    im = tf.imread(pathnames[0])
    image_shape = im.shape

    images = []
    counter = 0
    for iname in pathnames:
        im = tf.imread(iname)
        if downsample_factor is not None:
            im = im[0::downsample_factor,0::downsample_factor]
        images.append(im)
        counter = counter+1
        print("Completed with",(counter/len(pathnames)*100),'%')
        del im
    images = np.asarray(images) # convert to numpy array
    print("saving to ",dir_save)
    tf.imwrite(dir_save+'/tiff_stack.tif',images) # save as tiff

def extract_tar(path_name,data_name):
    """
    Function used to extract .tar file

    Args
        path_name: directory of dataset
    """
    # path_name = '/Users/js0403/Documents/DECODE-lab-code/calcium_imaging_analyses/datasets/'
    # data_name = 'FN_dataSharing.tgz-aa'
    data_dir = path_name+data_name
    file = tarfile.open(data_dir)
    file.extractall(path_name)
    file.close()
