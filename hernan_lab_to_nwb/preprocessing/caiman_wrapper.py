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
import os

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
        self.movieFrames = self.movieFrames[:,::downsample_factor,::downsample_factor]
        self.downsampled_spatial = True
        self.downsampled_spatial_factor = downsample_factor

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
        self.movieFrames = self.movieFrames[::downsample_factor,:,:]

        self.frate = self.frate/downsample_factor
        self.downsampled_temporal = True
        self.downsampled_temporal_factor = downsample_factor

        return self.movieFrames  
    
    def motion_correct(self, file_name = None, dview = None, opts = None, pw_rigid=False, fname = None, save_dir = None):
            
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
    
    def save_memap(self, file_name = None, dview = None, fname = None, save_dir = None):
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

        if save_dir is None:
            save_out = fname
        else:
            save_out = [save_dir]
        self.fname_memap = cm.save_memmap(fname, base_name=file_name+'_memmap_',
                        order='C', border_to_0=0, dview=dview)
        print("Memory mapped file (fname_memap):", self.fname_memap)

    def save_output(self, save_dir = None):
        """
        Saving the output. This is useful if you downsampled your dataset and wish to reload the results

        There are two save options:
            1) keep save_dir empty/set to None (don't provide the input):
                This saves a bunch of files to whatever the working directory is (self.fname)
            2) Define save_dir
                This creates a new directory wherever you want that directory to be located. This is preferred
        
        Args: Arguments are optional
            save_dir: string containing directory to save data

        """

        file_root = self.fname[0].split('.')[0]
        if save_dir is None:
            # save files according to the current directory
            self.fname_save = file_root+'_newSave.tif'
        else:
            # save file based on input directory
            self.fname_save = save_dir+'/'+self.root_file.split('.')[0]

            # update for saving
            new_dir = save_dir+'/'+self.root_file.split('.')[0]

            # make directory to save the dataset
            os.mkdir(new_dir)

            # get information about downsampling
            if self.downsampled_spatial == True:
                save_ds_spatial = '_spatialDSx'+str(self.downsampled_spatial_factor)
            else:
                save_ds_spatial = ''

            if self.downsampled_temporal == True:
                save_ds_temporal = '_temporalDSx'+str(self.downsampled_spatial_factor)
            else:
                save_ds_temporal = ''

            # create save directory
            save_dir_new = new_dir+'/'+self.root_file.split('.')[0]+save_ds_temporal+save_ds_spatial+'_'+str(self.frate)+'fps'+'.tif'
            self.fname_save = save_dir_new

        print("Saving output as",self.fname_save)
        tiff.imsave(self.fname_save,self.movieFrames)

    def update_fname(self, fname_update: str):
        """
            This function updates your caiman_preprocess object with a new movie. This might be useful in times where
            you split your video into two channels, save one video out, then upload a new video.

            Args:
                fname_update: update fname. Raw format should be a str inside a list, but just give a str 
                    and let the code handle the rest
        """
        if fname_update is str:
            fname_update = [fname_update] # put inside list
        self.movieFrames = cm.load_movie_chain(fname_update,fr=self.frate)
        print("Updated fname is: ",fname_update[0])        

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

    def fit_cnm(self, images, n_processes, dview, Ain=None, params=None, save_dir=None):

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
        if save_dir is None:
            data2save = self.root_folder+'/'+self.root_file+'_cnm_'+time_id+'.hdf5' # file storage name
        else:
            data2save = save_dir+'/'+self.root_file+'_cnm_'+time_id+'.hdf5' # file storage name
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
        #for i in range(len(idxMergeGood)):
            #for ii in range(len(idxMergeGood[i])):
                #idxMergeGood[i][ii]=idxMergeGood[i][ii]-1
        
        # need to make sure that we're indexing from the raw components, not the good ones

        # subtract 1 because the visual plots below have +1 due to 0-indexing
        #cnm.estimates.manual_merge(cnm.estimates.idx_components[idxMergeGood],cnm.params)
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

