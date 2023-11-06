# split2channelVideo
# This script is meant to split calcium image videos by channel.
# For example, 1 channel might image calcium transients, while the other
#   images a static color
#
# -- Dependencies -- 
# This code requires that you have downloaded the caiman pipeline successfully
# In VScode, you should activate your environment and CD to your downloaded caiman folder
#
# - JS - 07/21/23

import caiman as cm
import os
import tifffile as tiff
import numpy as np

# interface with user
loadFolder = input("Please enter the directory to data: ")
fileName   = input("Please enter the file name, include the file type: ")
frate      = float(input("Please enter the frame rate: "))

# Split 2-dye calcium imaging videos and save separately
#loadFolder = '/Users/js0403/caiman_data/example_movies'
#fileName   = '750K2_200ms_4XSum_video_RedGreen_depolar002.tif'
path_movie = [os.path.join(loadFolder, fileName)]
print("loading movie...")
m_orig = cm.load_movie_chain(path_movie, fr=frate, is3D=True)

# extract movies
m_ch1 = m_orig[:,0,:,:]
m_ch2 = m_orig[:,1,:,:]

# pop the extension
fileSplit = fileName.split('.')
fileSplit.pop(1)
fileName = fileSplit[0]

# play the videos
downsample_ratio = .2  # motion can be perceived better when downsampling in time
m_ch1.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=frate, magnification=0.5)   # play movie (press q to exit)
# enter which channel is which
channel1 = input("Enter name for channel1: ")
path_ch1 = os.path.join(loadFolder,fileName+'_'+channel1+'.tif')
print("Saving as", path_ch1)
tiff.imsave(path_ch1,m_ch1)

m_ch2.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=frate, magnification=0.5)   # play movie (press q to exit)
channel2 = input("Enter name for channel2: ")
path_ch2 = os.path.join(loadFolder,fileName+'_'+channel2+'.tif')
tiff.imsave(path_ch2,m_ch2)
print("Saving as", path_ch2)

print("Split complete")