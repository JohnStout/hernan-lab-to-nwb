#%% Script for messing with caiman parameters
#
# This in-vitro calcium imaging dataset is unique from 1p because the quality
# is so good, but unique from 2p and 1p because the dynamics are relatively poor.
#
# This in-vitro calcium imaging experiment is somewhere between 1p and 2p and 
# therefore, the parameters must treat the data as such
#
# To estimates F0, 10th percentile of detrended signal
#
#
# FIRST, run script_downsample_data.py if working with Akanksha data
#
#
# -JS

# %%

# importing packages/modules
from decode_lab_code.preprocessing.caiman_wrapper import caiman_preprocess

import matplotlib.pyplot as plt

import numpy as np

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr, display_animation
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass
import bokeh.plotting as bpl
import holoviews as hv
bpl.output_notebook()

try:
    hv.notebook_extension('bokeh')
except:
    print("fail")

import copy

from matplotlib import animation

#%%

# assign a folder name for analysis
folder_name = '/Users/js0403/ophysdata/Trevor_data/Neuron'
file_name = '750K2_200ms_4XSum_video_RedGreen_depolar002_neuron'
extension = '.tif'

#folder_name = '/Users/js0403/ophysdata/Akanksha_data'
#file_name = 'Movie_21'
#extension = '.tif'
frame_rate = 10 # OG sampling rate was 30. The data has been temporally downsampled in script_downsample_data

fname = [download_demo(file_name+extension,folder_name)]
movieFrames = cm.load_movie_chain(fname,fr=frame_rate) # frame rate = 30f/s
movieFrames.play()

# initialize the caiman_preprocess object for fast tools
cp = caiman_preprocess(folder_name,file_name+extension,frame_rate,activate_cluster=False)

#%%
# based on the visualization, 13 pixels is consistent with caiman
# neuron_size = 13 # pixels - this can be adjusted as needed after visualizing results
neuron_size = 15

#%% lets identify a good patch size
# patches were 192 for trevors
patch_size = 200; patch_overlap = patch_size/2
cp.test_patch_size(patch_size,patch_overlap)

# %%

# dataset dependent parameters
fname = fname  # directory of data
fr = frame_rate   # imaging rate in frames per second
decay_time = 0.4  # length of a typical transient in seconds

# motion correction parameters - we don't worry about mc
motion_correct = True      # flag for motion correcting - we don't need to here
strides = (patch_size, patch_size)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (patch_overlap, patch_overlap)   # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = False            # flag for performing non-rigid motion correction
border_nan = 'copy'         # replicate values along the border
if border_nan == 'copy': bord_px = 0
gSig_filt = (3,3) # change

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.85            # merging threshold, max correlation allowed
#rf = int(patch_size/2)     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
rf = int(neuron_size*4)
stride_cnmf = int(patch_size/4) # amount of overlap between the patches in pixels
#K = 5                          # number of components per patch
K = 5
gSiz = (neuron_size,neuron_size) # estimate size of neuron
gSig = [int(round(neuron_size-1)/2), int(round(neuron_size-1)/2)] # expected half size of neurons in pixels
method_init = 'corr_pnr'    # greedy_roi, initialization method (if analyzing dendritic data using 'sparse_nmf'), if 1p, use 'corr_pnr'
ssub = 2                    # spatial subsampling during initialization (2)
tsub = 1                    # temporal subsampling during intialization (1)
ssub_B = 2                  # additional downsampling factor in space for background (2)
low_rank_background = None  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
#gnb = 0                     # number of background components, gnb= 0: Return background as b and W, gnb=-1: Return full rank background B, gnb<-1: Don't return background
nb_patch = 2               # number of background components (rank) per patch if gnb>0, else it is set automatically
ring_size_factor = 1.4      # radius of ring is gSiz*ring_size_factor

# These values need to change based on the correlation image
min_corr = .8               # min peak value from correlation image
min_pnr = 10                # min peak to noise ration from PNR image

# parameters for component evaluation
min_SNR = 2.0    # signal to noise ratio for accepting a component
rval_thr = 0.85  # space correlation threshold for accepting a component
cnn_thr = 0.99   # threshold for CNN based classifier, was 0.99
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

# %%

# create a parameterization object
opts_dict = {
            # parameters for opts.data
            'fnames': fname,
            'fr': fr,
            'decay_time': decay_time,

            # parameters for opts.motion  
            'strides': strides,
            'pw_rigid': pw_rigid,
            'border_nan': border_nan,
            'gSig_filt': gSig_filt,
            'max_deviation_rigid': max_deviation_rigid,   
            'overlaps': overlaps,
            'max_shifts': max_shifts,    

            # parameters for preprocessing
            #'n_pixels_per_process': None, # how is this estimated?

            # parameters for opts.init
            'K': K, 
            'gSig': gSig,
            'gSiz': gSiz,  
            'nb': gnb, # also belongs to params.temporal 
            'normalize_init': False,   
            'rolling_sum': True,    
            'ssub': ssub,
            'tsub': tsub,
            'ssub_B': ssub_B,    
            'center_psf': True,
            'min_corr': min_corr,
            'min_pnr': min_pnr,            

            # parameters for opts.patch
            'border_pix': bord_px,  
            'del_duplicates': True,
            'rf': rf,  
            'stride': stride_cnmf,
            'low_rank_background': low_rank_background,                     
            'only_init': True,

            # parameters for opts.spatial
            'update_background_components': True,

            # parameters for opts.temporal
            'method_deconvolution': 'oasis',
            'p': p,

            # parameters for opts.quality
            'min_SNR': min_SNR,
            'cnn_lowest': cnn_lowest,
            'rval_thr': rval_thr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,

            # not sure
            'method_init': method_init,
            'merge_thr': merge_thr, 
            'ring_size_factor': ring_size_factor}

opts = params.CNMFParams(params_dict=opts_dict)

#%% 
# the interactive window hates this

# start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %% 

# motion correct

# Motion correction - don't need to worry about this much for Akanksha's dataset
print("This might take a minute...")
if motion_correct:
    # do motion correction rigid
    mc = MotionCorrect(fname=fname,dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
        plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')

    bord_px = 0 if border_nan == 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name=file_name+'_memmap_', order='C',
                               border_to_0=bord_px)
else:  # if no motion correction just memory map the file
    fname_new = cm.save_memmap(cp.fname, base_name=file_name+'_memmap_',
                               order='C', border_to_0=0, dview=dview)

#%% MEMORY MAPPING

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
#images = Yr.T.reshape((T,) + dims, order='F')
    #load frames in python format (T x X x Y)

#%% 

# restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='ipyparallel', n_processes=None, single_thread=False)

# %%

# run summary image
cn_filter, pnr = cm.summary_images.correlation_pnr(
    images, gSig=gSig[0],center_psf=True,swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile

# change params for min_corr and min_pnr
min_corr = round(np.min(cn_filter),1)
min_pnr  = round(np.min(pnr),1)
opts.change_params(params_dict={
    'min_corr': min_corr,
    'min_pnr': min_pnr})

# plot images
plt.subplot(1,3,1).imshow(np.mean(images,axis=0))
plt.title("Raw data")
plt.subplot(1,3,2).imshow(cn_filter)
plt.title("CN_filter")
plt.subplot(1,3,3).imshow(pnr)
plt.title("PNR")

# %%

# fit data with cnmf
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
cnm.fit(images)

#%% 

# refit the model using only the selected components
#cnm2 = cnm.refit(images, dview=dview)

#%%

# save output
data2save = folder_name+'/'+file_name+'_cnm.hdf5'
cnm.save(filename=data2save)
print("cnmf object saved")

# %%

# remove background and extract output

include_background = False

# Watch the video without the background in order to confirm results
mov = cnm.estimates.play_movie(images, q_max=99.9, magnification=1,
                                 include_bck=include_background, gain_res=2, bpx=bord_px)

#%% 

# extract the reconstructed video 
mov_nobck = mov[:,:,0:512]
mov_recon = mov[:,:,513:512*2+1]

plt.subplot(1,3,1)
plt.imshow(images[1,:,:],cmap='gray')
plt.subplot(1,3,2)
plt.imshow(mov_recon[25,:,:],cmap='binary')

"""
# initialize a nested list
mov_lst = [[] for i in range(3)]
counter = 0
for i in range(len(mov_lst)):
    mov_lst[i].append(mov[:,:,counter:counter+511])
    counter = counter+511

plt.subplot(1,3,1)
plt.imshow(mov_lst[0][0][0])
plt.subplot(1,3,2)
plt.imshow(mov_lst[0][0][1])
plt.subplot(1,3,3)
plt.imshow(mov_lst[0][0][2])
"""
print("Based on watching the film, I'm seeing ~27 ROIs")

#%%

# return background, then subtract from movie
# B = cnm.estimates.compute_background(Yr=images)


#%%
min_SNR = 1            # adaptive way to set threshold on the transient size
r_values_min = 0.9     # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})

# component evaluation
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

# %%

# plot some results
cnm.estimates.plot_contours_nb(img=pnr, idx=cnm.estimates.idx_components)

#%% 

"""
# I want to be able to watch the movie with components and select those components for removal
mov = cnm.estimates.make_color_movie(imgs=images, q_max=99.9, magnification=1,
                               include_bck=True, gain_res=2, bpx=0)
"""

# %%

# This may only work in a notebook!

# we need to identify our components for manual rejection
#cnm.estimates.view_components(img=cn_filter,idx=cnm.estimates.idx_components)
cnm.estimates.view_components(img=pnr,idx=cnm.estimates.idx_components)

#%%
#cnm.estimates.make_color_movie(imgs=images,include_bck=False)

cnm.estimates.make_color_movie(images, q_max=99.9, magnification=1,
                               include_bck=include_background, gain_res=2, bpx=bord_px)


#%% 

"""
You're using select_components correctly. 
Either execute evaluate_components 
followed by select_components(use_object=True) 
for automatic selection or select_components(idx_components=new_inds) 
for manual selection. 
I updated the code in the dev branch (commit 06b4eab), 
so that it is compatible with iterated component selection. 
Newly discarded components are now appended to discarded_components 
instead of overwriting them.
"""

"""
Important objects:
    cnm.estimates.C = temporal traces for each neuron
"""

# unit ID
unitID = 0

# extract components
rois = cnm.estimates.coordinates
roi = rois[unitID].get('coordinates')
CoM = rois[unitID].get('CoM')

# no idea how to plot these one by one for rejection
plt.imshow(pnr)

# plot the transposed numpy array
#plt.plot(*roi.T, c='w') # the star makes the 2D array and treats than as two input variables
plt.plot(roi.T[0,:],roi.T[1,:],c='w')
ax = plt.gca()
ax.text(CoM[1], CoM[0], str(unitID + 1), color='w',fontsize=20)

# mov_recon is a nested list, each element is a timepoint
mov_recon.play()

#%% 

# another way to get the contours
components = plot_contours(A=cnm.estimates.A,Cn=mov_recon[0,:,:],idx=cnm.estimates.idx_components,colors='k')
#plt.savefig('/Users/js0403/ophysdata/Trevor_data/Figs')

# %%

# get df/f
#cnm.estimates.detrend_df_f(quantileMin=8, frames_window=500, flag_auto=True)
cnm.estimates.detrend_df_f(flag_auto=False, detrend_only=True)

# lets get the dF/F
dfF_all = cnm.estimates.F_dff

# get only good components
dfF_good = dfF_all[cnm.estimates.idx_components]

# get total time
totalTime = dfF_good.shape[1]/frame_rate

# make x-axis
xAxis = np.linspace(0,totalTime,dfF_good.shape[1])

"""
fig = plt.figure(dpi=2000)
ax = fig.add_subplot(dfF_good.shape[0],projection='3d')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('norm. dF/F')
ax.axis('off')
for i in range(dfF_good.shape[0]):
    line = ax.plot()
"""

fig = plt.figure(dpi=2000)
gs  = fig.add_gridspec(dfF_good.shape[0], hspace=0)
axs = gs.subplots(sharex=True, sharey=False)

range(dfF_good.shape[0])
for i in range(dfF_good.shape[0]):
    axs[i].plot(xAxis,dfF_good[i,:],'k',linewidth=0.5)
    axs[i].set_yticks([])
    #axs[i].xticks(fontsize=8)       
    if i != dfF_good.shape[0]-1:   
        axs[i].axis('off')
    else:
        axs[i].set_frame_on(False)
# %%
