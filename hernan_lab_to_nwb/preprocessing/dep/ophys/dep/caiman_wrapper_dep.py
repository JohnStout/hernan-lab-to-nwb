# caiman_wrapper
# This wrapper module is meant to make certain preprocessing steps of caiman into one-liners.
# For example, motion correction shouldn't be a multiline process, but instead a one-liner.
# Extracting data and watching the video playback should be easy. Viewing the results should
# be easy
#
# this code is specifically for miniscope. Future additions will include 2p
#
# written by John Stout using the caiman demos

# prep work
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Rectangle as box
import tifffile as tiff
from datetime import datetime

# parent class
class caiman_preprocess:

    def __init__(self, folder_name: str, file_name: str, frate: int, activate_cluster: bool):
        self.fname = [download_demo(file_name,folder_name)]
        self.frate = frate
        self.root_folder = folder_name
        self.root_file = file_name
        print("Loading movie for",self.fname)
        try:
            self.movieFrames = cm.load_movie_chain(self.fname,fr=self.frate) # frame rate = 30f/s
        except:
            self.movieFrames = cm.load_movie_chain(self.fname,fr=self.frate,is3D=True)
            
        # this actually doesn't function without activating the cluster
        if activate_cluster:            
            # start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
            if 'dview' in locals():
                cm.stop_server(dview=dview)
            # n_processes reflect cluster num
            print("cluster set-up")
            self.c, self.dview, self.n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)
            
        # autoparameterize

    def get_init_vars(self):
        init_dict = {
            'fname': self.fname,
            'frame_rate': self.frate,
            'frame_data': self.movieFrames,
            'cluster_processes (n_processes)': self.n_processes,
            'ipyparallel.Client_object': self.c,
            'ipyparallel_dview_object': self.dview
        }
        return init_dict
    
    def update_fname(self, fname_update: str):
        """
            This function updates your caiman_preprocess object with a new movie. This might be useful in times where
            you split your video into two channels, save one video out, then upload a new video.
        """
        fname_update = [fname_update] # put inside list
        self.movieFrames = cm.load_movie_chain(fname_update,fr=self.frate)
        print("Updated fname is: ",fname_update[0])

    def load_second_video(self, fname2: str):
        """
            The caiman_preprocess object works with self.movieFrames as the functional video. But sometimes you might want to load
            in a second video. You could do this by instantiating a second caiman_preprocess object or simply calling this function
        """
        fname_2 = [fname_2]
        self.movieFrames_2 = cm.load_movie_chain(fname_2,fr=self.frate)
        
    def watch_movie(self):
        # playback
        downsample_ratio = .2  # motion can be perceived better when downsampling in time
        self.movieFrames.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=self.frate, magnification=0.5)   # play movie (press q to exit)   

    def get_frames(self): 
        # get frames
        movieFrames = self.movieFrames
        return movieFrames
    
    def test_patch_size(self, patch_size = None, patch_overlap = None, img = None):
        
        """
        Description:
            This function allows the user to test various patch_sizes for analysis. 

        Args:
            OPTIONAL VARIABLES:
            patch_size: size of patches in pixels
            patch_overlap: overlap of patches in pixels
            img: video image to use. Can be a video or image.

        """

        if patch_size is None:
            patch_size = int(len(self.movieFrames[0,:,:])/2)
        if patch_overlap is None:
            patch_overlap = int(patch_size/2)
        if img is None:
            img = self.movieFrames

        # This function will interact with the user to test the size of patches to play with
        exData = np.mean(img,axis=0)

        fig, ax = plt.subplots()
        plt.imshow(exData)        
        ax.add_patch(box(xy=(0,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))
        ax.add_patch(box(xy=(0+patch_overlap,0), width=patch_size, height=patch_size, edgecolor = 'yellow',fill=False))
        plt.title("Patches and their overlap")
        print("Patch size = ",patch_size,". Patch overlap = ",patch_overlap)
        #return fig
        return patch_size, patch_overlap
    
    def spatial_downsample(self, downsample_factor: int):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        downsample_factor: factor to spatially downsample your dataset

        """

        # when slicing, start:stop:step
        frameShape = self.movieFrames.shape # frameshape

        # downsample
        self.movieFrames = self.movieFrames[:,0:frameShape[1]:downsample_factor,0:frameShape[2]:downsample_factor]

        return self.movieFrames
    
    def temporal_downsample(self, downsample_factor: int):
        """
        Spatially downsample your data (downsample pixels) by a chosen factor
        
        --- INPUTS ---
        downsample_factor: scale for downsampling
        """

        # when slicing, start:stop:step
        frameShape = self.movieFrames.shape # frameshape

        # downsample
        self.movieFrames = self.movieFrames[0:frameShape[0]:downsample_factor,:,:]

        self.frate = self.frate/downsample_factor

        return self.movieFrames  
    
    def motion_correct(self, file_name = None, dview = None, opts = None, pw_rigid=False, fname = None):
            
            if fname is None:
                print("Defaulting fname to: ",self.fname)
                fname = self.fname
            if file_name is None:
                file_name = self.root_file

            # do motion correction rigid
            mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
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

            border_nan = opts.motion.get('border_nan')
            bord_px = 0 if border_nan == 'copy' else bord_px
            self.fname_memap = cm.save_memmap(fname_mc, base_name=file_name+'_memmap_', order='C',
                                    border_to_0=bord_px)   
    
    def save_memap(self, file_name = None, dview = None, fname = None):
        """
            This function saves your data as a memory-mapped file 

            --INPUTS--
            file_name: file directory
            dview: pool output from cluster_helper.start_cluster or cluster_helper.refresh_cluster

        """
        if fname is None:
            print("Defaulting fname to: ",self.fname)
            fname = self.fname
        if file_name is None:
            print("Defaulting file_name to root_file: ",self.root_file)
            file_name = self.root_file
            if '.tif' in file_name or '.avi' in file_name:
                file_name = file_name.split('.')[0]
                print("Removing extension. Updated file_name: ", file_name)
        if type(fname) is str:
            fname = [fname]

        self.fname_memap = cm.save_memmap(fname, base_name=file_name+'_memmap_',
                        order='C', border_to_0=0, dview=dview)
        print("Memory mapped file (fname_memap):", self.fname_memap)

    def save_output(self):
        """
        Saving the output. This is useful if you downsampled your dataset and wish to reload the results
        """
        self.fname
        self.file_root = self.fname[0].split('.')[0]
        self.fname_save = self.file_root+'_newSave.tif'
        print("Saving output as",self.fname_save)
        tiff.imsave(self.fname_save,self.movieFrames)

    def split_4D_movie(self, structural_index: int, functional_index: int):
        """
        This function will take a 4D movie, split it into its components based on your index, 
        then save the output based on the save name index.

        This function was specifically designed when one records a structural channel with a functional channel. 
        For example, you might record astrocytes with an RFP, but neurons or all cells with a GCaMP.

        Args:
            self: 
            structural_index:
            functional_index:

        output:
            self.fname_struct: structural fname
            self.fname_funct: functional fname  
        
        """
        self.fname
        self.file_root = self.fname[0].split('.')[0]
        self.fname_funct = self.file_root+'_functional.tif'
        self.fname_struct = self.file_root+'_structural.tif'
        print("Saving functional output as",self.fname_funct)
        print("Saving structural output as",self.fname_struct)

        # split data
        self.structMovie = self.movieFrames[:,structural_index,:,:]
        self.functMovie = self.movieFrames[:,functional_index,:,:]        
        tiff.imsave(self.fname_struct,self.structMovie)  
        tiff.imsave(self.fname_funct,self.functMovie)            

    # inheritance of parent init
    def miniscope_params(self, neuron_size: int = 15, K = None, decay_time: float = 0.4, nb: int = 0, p: int = 1, patch_size = None, patch_overlap = None, ssub: int = 1, tsub: int = 2, ssub_B: int = 2, merge_thr: float = 0.7, fname = None, fr = None, img = None):

        """
            Default parameters for miniscope recordings

            --INPUTS--

            *** ARE YOU HAVING ISSUES WITH COMPONENT ESTIMATION? COMPONENTS TOO SMALL OR TOO LARGE? CHANGE THESE: ***

                neuron_size: the size of the neuron in pixels

                K: Number of neurons estimated per patch

                decay_time: Length of a typical transient in seconds. decay_time is an approximation of the time scale over which to 
                        expect a significant shift in the calcium signal during a transient. It defaults to `0.4`, which `is 
                        appropriate for fast indicators (GCaMP6f)`, `slow indicators might use 1 or even more`. 
                        decay_time does not have to precisely fit the data, approximations are enough

                nb: # of global background components. Default is 2 for relatively homogenous background.
                        IF nb is too high, components will get sent to the background noise

                p: order of autoregression model. Default = 1.
                        p = 0: turn deconvolution off
                        p = 1: for low sampling rate and/or slow calcium transients
                        p = 2: for transients with visible rise-time

            ** IS PROCESSING TAKING TOO LONG? CHANGE THESE: **

                patch_size: automatically estimated based on the size of the movie/3

                patch_overlap: automatically estimated based on the patch_size/2

                ssub: Spatial subsampling. Default = 2.
                
                tsub: temporal subsampling. Default = 2.

                ssub_B: subsampling for background. Default = 2

            **ARE YOU DEALING WITH MULTIPLE COMPONENTS THAT SHOULDVE BEEN MERGED?? TRY THIS: **

                merge_thr: correlation threshold for merging components. Default is R = 0.7

            --OUTPUTS--
            opts: params.CNMF object for CNMF and motion correction procedures
        """

        if patch_size == None:

            if img not in locals():
                # use the size of movie to estimate patch_size
                patch_size = int(len(self.movieFrames[0,:,:])/2)
            else:
                patch_size = int(len(img[0,:,:])/2)

            patch_overlap = int(patch_size/2)

        else: 
            if type(patch_size)!=int or type(patch_overlap)!=int:
                print("converting patch_size and patch_overlap to int")
                patch_size = int(patch_size)
                if patch_overlap is None:
                    patch_overlap = int(patch_size/2)
                else:
                    patch_overlap = int(patch_overlap)

        # add those attributes to self
        self.neuron_size = neuron_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # dataset dependent parameters
        if fname is None:
            print("Defaulting fname to: ", self.fname)
            fname = self.fname  # directory of data
        if fr is None:
            print("Defaulting frame rate to: ", self.frate)
            fr = self.frate   # imaging rate in frames per second
        #decay_time = .4  # length of a typical transient in seconds

        # motion correction parameters - we don't worry about mc
        motion_correct = True      # flag for motion correcting - we don't need to here
        max_shifts = (5,5)          # maximum allowed rigid shifts (in pixels)
        max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
        pw_rigid = False            # flag for performing non-rigid motion correction
        border_nan = 'copy'         # replicate values along the border
        bord_px = 0
        #gSig_filt = (int(round(neuron_size-1)/2),int(round(neuron_size-1)/2)) # change
        gSig_filt = [int(round(neuron_size-1)/4), int(round(neuron_size-1)/4)]

        # parameters for source extraction and deconvolution
        gnb = nb                     # number of global background components

        rf = int(patch_size/2)       # half-size of patches in pixels
        stride_cnmf = int(rf/2)      # amount of overlap between the patches in pixels

        gSiz = (neuron_size,neuron_size) # estimate size of neuron
        gSig = [int(round(neuron_size-1)/4), int(round(neuron_size-1)/4)] # expected half size of neurons in pixels

        method_init = 'corr_pnr'    # greedy_roi, initialization method (if analyzing dendritic data using 'sparse_nmf'), if 1p, use 'corr_pnr'
        low_rank_background = None  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
        nb_patch = nb               # number of background components (rank) per patch if gnb>0, else it is set automatically
        ring_size_factor = 1.4      # radius of ring is gSiz*ring_size_factor

        if stride_cnmf <= 4*gSig[0]:
            Warning('Considering increasing the size of patche_size')        

        # These values need to change based on the correlation image
        min_corr = .8               # min peak value from correlation image
        min_pnr = 10                # min peak to noise ration from PNR image

        # create a parameterization object
        opts_dict = {
                    # parameters for opts.data
                    'fnames': fname,
                    'fr': fr,
                    'decay_time': decay_time,

                    # parameters for opts.motion  
                    #'strides': strides,
                    'pw_rigid': pw_rigid,
                    'border_nan': border_nan,
                    'gSig_filt': gSig_filt,
                    'max_deviation_rigid': max_deviation_rigid,   
                    #'overlaps': overlaps,
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

                    # not sure
                    'method_init': method_init,
                    'merge_thr': merge_thr, 
                    'ring_size_factor': ring_size_factor}

        opts = params.CNMFParams(params_dict=opts_dict) 
        return opts    
    
    def microscope_params(self, neuron_size: int = 15, K = 10, decay_time: float = 2, nb: int = 2, p: int = 0, patch_size = None, patch_overlap = None, ssub: int = 2, tsub: int = 2, ssub_B: int = 2, merge_thr: float = 0.7, fname = None, fr = None, img = None):

        """
            Default parameters for miniscope recordings

            --INPUTS--

            *** ARE YOU HAVING ISSUES WITH COMPONENT ESTIMATION? COMPONENTS TOO SMALL OR TOO LARGE? CHANGE THESE: ***

                neuron_size: the size of the neuron in pixels

                K: Number of neurons estimated per patch

                decay_time: Length of a typical transient in seconds. decay_time is an approximation of the time scale over which to 
                        expect a significant shift in the calcium signal during a transient. It defaults to `0.4`, which `is 
                        appropriate for fast indicators (GCaMP6f)`, `slow indicators might use 1 or even more`. 
                        decay_time does not have to precisely fit the data, approximations are enough

                nb: # of global background components. Default is 2 for relatively homogenous background.
                        IF nb is too high, components will get sent to the background noise

                p: order of autoregression model. Default = 1.
                        p = 0: turn deconvolution off
                        p = 1: for low sampling rate and/or slow calcium transients
                        p = 2: for transients with visible rise-time

            ** IS PROCESSING TAKING TOO LONG? CHANGE THESE: **

                patch_size: automatically estimated based on the size of the movie/3

                patch_overlap: automatically estimated based on the patch_size/2

                ssub: Spatial subsampling. Default = 2.
                
                tsub: temporal subsampling. Default = 2.

                ssub_B: subsampling for background. Default = 2

            **ARE YOU DEALING WITH MULTIPLE COMPONENTS THAT SHOULDVE BEEN MERGED?? TRY THIS: **

                merge_thr: correlation threshold for merging components. Default is R = 0.7

            --OUTPUTS--
            opts: params.CNMF object for CNMF and motion correction procedures
        """

        if patch_size == None:
            
            if img not in locals():
                # use the size of movie to estimate patch_size
                patch_size = int(len(self.movieFrames[0,:,:])/2)
            else:
                patch_size = int(len(img[0,:,:])/2)

            patch_overlap = int(patch_size/2)

        else: 
            if type(patch_size)!=int or type(patch_overlap)!=int:
                print("converting patch_size and patch_overlap to int")
                patch_size = int(patch_size)
                if patch_overlap is None:
                    patch_overlap = int(patch_size/2)
                else:
                    patch_overlap = int(patch_overlap)

        # add those attributes to self
        self.neuron_size = neuron_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # dataset dependent parameters
        if fname is None:
            print("Defaulting fname to: ", self.fname)
            fname = self.fname  # directory of data
        if fr is None:
            print("Defaulting frame rate to: ", self.frate)
            fr = self.frate   # imaging rate in frames per second
        #decay_time = .4  # length of a typical transient in seconds

        # motion correction parameters - we don't worry about mc
        motion_correct = True      # flag for motion correcting - we don't need to here
        max_shifts = (5,5)          # maximum allowed rigid shifts (in pixels)
        max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
        pw_rigid = False            # flag for performing non-rigid motion correction
        border_nan = 'copy'         # replicate values along the border
        bord_px = 0
        #gSig_filt = (int(round(neuron_size-1)/2),int(round(neuron_size-1)/2)) # change
        gSig_filt = [int(round(neuron_size-1)/4), int(round(neuron_size-1)/4)]

        # parameters for source extraction and deconvolution
        gnb = nb                     # number of global background components

        rf = int(patch_size/2)       # half-size of patches in pixels
        stride_cnmf = int(rf/2)      # amount of overlap between the patches in pixels

        gSiz = (neuron_size,neuron_size) # estimate size of neuron
        gSig = [int(round(neuron_size-1)/4), int(round(neuron_size-1)/4)] # expected half size of neurons in pixels

        method_init = 'corr_pnr'    # greedy_roi, initialization method (if analyzing dendritic data using 'sparse_nmf'), if 1p, use 'corr_pnr'
        low_rank_background = True  # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
        nb_patch = nb               # number of background components (rank) per patch if gnb>0, else it is set automatically
        ring_size_factor = 1.4      # radius of ring is gSiz*ring_size_factor

        if stride_cnmf <= 4*gSig[0]:
            Warning('Considering increasing the size of patche_size')        

        # These values need to change based on the correlation image
        min_corr = .8               # min peak value from correlation image
        min_pnr = 10                # min peak to noise ration from PNR image
        
        # parameters for component evaluation
        min_SNR = 2.0    # signal to noise ratio for accepting a component
        rval_thr = 0.85  # space correlation threshold for accepting a component
        cnn_thr = 0.99   # threshold for CNN based classifier, was 0.99
        cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

        # create a parameterization object
        opts_dict = {
                    # parameters for opts.data
                    'fnames': fname,
                    'fr': fr,
                    'decay_time': decay_time,

                    # parameters for opts.motion  
                    #'strides': strides,
                    'pw_rigid': pw_rigid,
                    'border_nan': border_nan,
                    'gSig_filt': gSig_filt,
                    'max_deviation_rigid': max_deviation_rigid,   
                    #'overlaps': overlaps,
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
        return opts    



    def fit_cnm(self, images, n_processes, dview, Ain=None, params=None):

        """
            This function fits the video with the CNMF algorithm

            --INPUTS--
                images: the video file, as memmap file
                n_processes: number of cpu processes for parallel computing
                dview: pool status for parallel processsing
                Ain: None
                params: opts input
        """

        if params is None:
            print("Defaulting to miniscope parameters as no params object was detected")
            params = self.miniscope_params()

        # fit data with cnmf
        print("Fitting CNMF...")
        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=params)
        cnm.fit(images) # fit images input

        # save output
        print("saving...")
        time_id = saving_identifier() # get unique identifier of the time
        data2save = self.root_folder+'/'+self.root_file+'_cnm_'+time_id+'.hdf5' # file storage name
        cnm.save(filename=data2save) # save
        cm.stop_server(dview=dview) # stop parallel processing

        # return output
        self.cnm = cnm

        return cnm
                

class caiman_cnm_curation:

    def __init__(self):
        """
        """

    def component_eval(images, cnm, dview, min_SNR=2, r_values_min=0.9):

        """ 
        component_eval: function meant to evaluate components. Must run this before cleaning up dataset.

        -- INPUTS -- 
            cnm: cnm object
            dview: multiprocessing toolbox state
            min_SNR: signal-noise-ratio, a threshold for transient size
            r_values_min: threshold for spatial consistency (lower to increase component yield)
        
        -- OUTPUTS --
            cnm: cnm object with components
        """
        
        #min_SNR = 2            # adaptive way to set threshold on the transient size
        #r_values_min = 0.9     # threshold on space consistency (if you lower more components
        #                        will be accepted, potentially with worst quality)
        cnm.params.set('quality', {'min_SNR': min_SNR,
                                'rval_thr': r_values_min,
                                'use_cnn': False})

        # component evaluation
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

        print(' ***** ')
        print('Number of total components: ', len(cnm.estimates.C))
        print('Number of accepted components: ', len(cnm.estimates.idx_components))

        return cnm
    

    # a plotter function
    def plot_components(img, cnm, colors: list, colorMap='viridis', clim = []):

        """
        Args:
            img: image to plot results over
            cnm: cnm object 
            colors: list of color indicators
            colorMap: color for imshow map
            clim: colorbar range of heat map
        """

        # extract components
        rois = cnm.estimates.coordinates # get rois
        idx = cnm.estimates.idx_components # get accepted components
        good_rois = [rois[i] for i in range(len(rois)) if i in idx]


        # plot components
        plt.subplot(1,2,1)
        plt.imshow(img,colorMap)
        if len(clim)!=0:
            plt.clim(clim)
        for i in range(len(good_rois)):
            roi = good_rois[i].get('coordinates')
            CoM = good_rois[i].get('CoM')
            plt.plot(roi.T[0,:],roi.T[1,:],c=colors[i],linewidth=2)
            plt.text(CoM[1], CoM[0], str(i + 1), color='w', fontsize=10)

        # plot traces
        cnm.estimates.detrend_df_f(flag_auto=True, frames_window=100, detrend_only=True) # get normalized df/f
        fr = cnm.params.data['fr'] # get frame rate
        dfF_all = cnm.estimates.F_dff # filter
        dfF_good = dfF_all[cnm.estimates.idx_components] # filter
        totalTime = dfF_good.shape[1]/fr # estimate elapsed time
        xAxis = np.linspace(0,totalTime,dfF_good.shape[1]) # make x-axis

        #plt.subplot(100,2,2)
        #plt.plot(xAxis,dfF_good[0,:],c=colors[0],linewidth=1)
        #plt.subplot(100,2,4)
        #plt.plot(xAxis,dfF_good[1,:],c=colors[1],linewidth=1)

        for i in range(dfF_good.shape[0]):
            if i == 0:
                counter = 2
            ax = plt.subplot(dfF_good.shape[0],2,counter)
            plt.plot(xAxis,dfF_good[i,:],c=colors[i],linewidth=1)
            plt.title('ROI #'+str(i+1),fontsize=8,color=colors[i])
            ax.set_axis_off()
            #ax.set_ylabel('ROI #'+str(i),color=colors[i])
            #plt.Axes(frameon=False)
            counter = counter+2

    # some functions to help us merge and reject accepted components
    def merge_components(cnm,idxMergeGood):
        """
        merge_components: helper function to merge
        idxMergeGood: list of accepted components to merge
        """
        for i in range(len(idxMergeGood)):
            for ii in range(len(idxMergeGood[i])):
                idxMergeGood[i][ii]=idxMergeGood[i][ii]-1

        # subtract 1 because the visual plots below have +1 due to 0-indexing
        cnm.estimates.manual_merge(cnm.estimates.idx_components[idxMergeGood],cnm.params)

        return cnm

    def good_to_bad_components(cnm,idxGood2Bad):
        """
        This function will place good components to the rejected index
        """
        # remove components
        data2rem = cnm.estimates.idx_components[idxGood2Bad]
        cnm.estimates.idx_components = np.delete(cnm.estimates.idx_components,np.array(idxGood2Bad)-1)

        # add to rejected array
        cnm.estimates.idx_components_bad = np.sort(np.append(cnm.estimates.idx_components_bad,idxGood2Bad))

        return cnm
    
    def inspect_corr_pnr(correlation_image_pnr, pnr_image, cbar_limits: list = []):
        import pylab as pl

        """
        inspect correlation and pnr images to infer the min_corr, min_pnr

        Args:
            correlation_image_pnr: ndarray
                correlation image created with caiman.summary_images.correlation_pnr
        
            pnr_image: ndarray
                peak-to-noise image created with caiman.summary_images.correlation_pnr

            cbar_limits: nested list containing colorbar scale

        Output:
            min_corr: Minimum correlation from the correlation_image_pnr
            min_pnr: minimum peak to noise ratio returned from the pnr_image

            * these outputs will return min values of the raw inputs, OR the cbar_limits you provide
        """

        fig = pl.figure(figsize=(10, 4))
        pl.axes([0.05, 0.2, 0.4, 0.7])
        im_cn = pl.imshow(correlation_image_pnr, cmap='jet')
        pl.title('correlation image')
        pl.colorbar()
        if len(cbar_limits)!=0:
            pl.clim(cbar_limits[0])
        else:
            pl.clim()
        
        pl.axes([0.5, 0.2, 0.4, 0.7])
        im_pnr = pl.imshow(pnr_image, cmap='jet')
        pl.title('PNR')
        pl.colorbar()
        if len(cbar_limits)!=0:
            pl.clim(cbar_limits[1])
        else:
            pl.clim()

        # assign min_corr and min_pnr based on the image you create
        if len(cbar_limits)==0:
            min_corr = round(np.min(correlation_image_pnr),1)
            min_pnr  = round(np.min(pnr_image),1)
        else:
            min_corr = cbar_limits[0][0]
            min_pnr = cbar_limits[1][0]

        print("minimum correlation: ",min_corr)
        print("minimum peak-to-noise ratio: ",min_pnr)

        return min_corr, min_pnr
    
class cluster_helper:

    def __init__(self):
        """
        """

    def start_cluster():
        print("starting cluster")
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        return c, dview, n_processes
        
    def refresh_cluster(dview):
        print("refreshing cluster")
        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)        
        return c, dview, n_processes

# function to establish a unique identifier for saving
def saving_identifier():
    """
        This function provides the output "time_id" as a unique identifier to save your data
    """

    now = str(datetime.now())
    now = now.split(' ')
    now = now[0]+'-'+now[1]
    now = now.split(':')
    now = now[0]+now[1]+now[2]
    now = now.split('.')
    time_id = now[0]
    return time_id

