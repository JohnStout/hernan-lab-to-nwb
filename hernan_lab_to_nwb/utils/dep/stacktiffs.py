import numpy as np
import tifffile as tf
import glob

# change me
# dir = directory of data (change me)
dir = '/Volumes/decode/Akanksha/Slice_calcium_imaging_videos_images/Pilots/Static_recordings/08-31-2023/Tiff_series_Process_7'
extension = '.tif' # no need to change
mid_ext = '/*' # don't change

# define a downsample factor
downsample_factor = 2 # set to None if you don't want to use

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



