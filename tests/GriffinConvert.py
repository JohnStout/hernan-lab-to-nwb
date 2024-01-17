# test conversion to NWB using Griffin lab data

# load packages
import numpy as np
import pandas as pd
from hernan_lab_to_nwb.readers import ioreaders
from hernan_lab_to_nwb.utils import nwb_utils
import os
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
import scipy.io

from pynwb.behavior import (
    BehavioralEpochs,
    BehavioralEvents,
    BehavioralTimeSeries,
    CompassDirection,
    EyeTracking,
    Position,
    PupilTracking,
    SpatialSeries,
)
from pynwb.epoch import TimeIntervals
from pynwb.misc import IntervalSeries

# this chunk is largely not contributing to anything
ratname = '21-37'

# rats list
rats_list = [['21-12', '21-13', '21-14', '21-33', '21-15', '21-16', '21-21', '21-37'] , 
             ['21-49', '21-55', '21-48'],
             ['21-42', '21-43', '21-45'],
             ['BabyGroot', 'Meusli','Groot']]

# experiment details
experiment_details = [['Delayed Alternation BMI Experiment'],
                      ['Conditional Discrimination BMI Experiment'],
                      ['Optogenetic activation of Ventral Midline thalamus'],
                      ['Simultaneous PFC tetrode recordings with thalamic and hippocampal LFPs']]


# directory  information
directory_root = [[r'X:\01.Experiments\R21'],
                     [r'X:\01.Experiments\R21'],
                     [r'X:\01.Experiments\R21'],
                     [r'X:\01.Experiments\John n Andrew\Dual optogenetics w mPFC recordings\All Subjects - DNMP\Good performance\Medial Prefrontal Cortex']]

# end of directory to fill
directory_fill = [[r'Sessions\DA testing'],
                  [r'CD\Sessions'],
                  [r'Opto\data\Sessions'],
                  ['dir_contents']]

# fullpath for each rat
rat_fullpath = [

    # DA testing data
    [r'X:\01.Experiments\R21\21-12\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-13\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-14\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-15\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-16\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-21\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-33\Sessions\DA testing',
     r'X:\01.Experiments\R21\21-37\Sessions\DA testing'],

    # CD testing data
    [r'X:\01.Experiments\R21\21-49\CD\Sessions',
     r'X:\01.Experiments\R21\21-48\CD\Sessions',
     r'X:\01.Experiments\R21\21-55\CD\Sessions'],
     
     # Opto testing data
     [r'X:\01.Experiments\R21\21-42\Opto\data\Sessions',
      r'X:\01.Experiments\R21\21-43\data\Sessions',
      r'X:\01.Experiments\R21\21-45\Opto\data\Sessions'],
      
      # Stout Unit with LFPs
      [r'X:\01.Experiments\John n Andrew\Dual optogenetics w mPFC recordings\All Subjects - DNMP\Good performance\Medial Prefrontal Cortex\BabyGroot',
       r'X:\01.Experiments\John n Andrew\Dual optogenetics w mPFC recordings\All Subjects - DNMP\Good performance\Medial Prefrontal Cortex\Groot',
       r'X:\01.Experiments\John n Andrew\Dual optogenetics w mPFC recordings\All Subjects - DNMP\Good performance\Medial Prefrontal Cortex\Meusli']
       ]

# Manually selected which list to run

# loop over rats_list
for ratname in rats_list[1]:

    # subject info
    if '21-55' in ratname:
        subject_details = '64ch silicon probe in PFC, AAV-ChR2 injected into VMT'
    elif 'BabyGroot' in ratname or 'Meusli' in ratname or 'Groot' in ratname:
        subject_details = 'Tetrode array in PFC, stainless steel wires in VMT and HPC. Signal quality varies session-by-session. For LFP, I sifted through the dataset at 1.25s intervals to select epochs that had low incidence of artifactual events. This was specifically focused on the delay phase'
    else:
        subject_details = 'Stainless steel electrodes placed in PFC and HPC'

    # automatic assignment of rats sex
    if '21-12' in ratname or '21-13' in ratname or '21-14' in ratname or '21-33' in ratname or '21-49' in ratname or '21-55' in ratname:
        rats_sex = 'Female'
    else:
        rats_sex = 'Male'

    # get directory containing all session data
    experiment = [experiment_details[i] for i in range(len(rat_fullpath)) for ii in rat_fullpath[i] if ratname in ii][0]
    folder_paths = [ii for i in rat_fullpath for ii in i if ratname in ii][0]
    session_paths = sorted(os.listdir(folder_paths))

    # loop over sessions
    for sessi in session_paths:

        # define folder name
        fname = os.path.join(folder_paths,sessi)
        print("Working on:",fname)

        # automated detection of metadata
        ratcheck = fname.split('R21')[-1].split('\\')[1]
        if ratcheck != ratname:
            assert ValueError("Rat name did not checkout with input")
        print("Rat:",ratname," Session:",sessi)

        # metadata assignment
        df = pd.DataFrame()
        df['experimenter name(s)']=[['John Stout', 'Allison George', 'Suhyeong Kim', 'Henry Hallock', 'Amy Griffin']]
        df['institution']=['University of Delaware']
        df['lab_name']=['Griffin Lab']
        df['subject_age']=['>PD90']
        df['subject_description']=['Rat']
        df['subject_species/genotype']=['Long Evans Rat']
        df['recording_device_name']=['Digitalynx SX']
        df['recording_device_description']=['Cheetah Software']
        df['recording_device_manufacturer']=['Neuralynx']

        # TODO: CHANGE ME
        df['experiment_description']=[experiment]
        df['session_description']   =['See experiment_description']
        df['session_notes']         =['Post Criterion Recording']
        df['session_id']            =[sessi]
        df['subject_id']            =[ratname]
        df['subject_sex']           =[rats_sex]

        try:
            # Read data
            if 'ephys_object' in locals():
                del ephys_object
            ephys_object = ioreaders.read_nlx(fname) # we first instantiate the object
            ephys_object.read_ephys(opts = 'CSC') # read 

            # tetrode group assignment
            for i in range(len(ephys_object.csc_grouping_table['TetrodeGroup'])):
                # fill out tetrode group
                ephys_object.csc_grouping_table['TetrodeGroup'][i] = i+1

                # fill out brainregion
                if 'HPC' in ephys_object.csc_grouping_table['Name'][i]:
                    ephys_object.csc_grouping_table['BrainRegion'][i] = 'HPC'
                elif 'PFC' in ephys_object.csc_grouping_table['Name'][i]:
                    ephys_object.csc_grouping_table['BrainRegion'][i] = 'PFC'
                elif 'Str' in ephys_object.csc_grouping_table['Name'][i]:
                    ephys_object.csc_grouping_table['BrainRegion'][i] = 'dStriatum'        
                elif 'Cer' in ephys_object.csc_grouping_table['Name'][i] or 'ref' in ephys_object.csc_grouping_table['Name'][i]:
                    ephys_object.csc_grouping_table['BrainRegion'][i] = 'Cerebellum_reference'

            # write NWB file
            ephys_object.write_nwb(metadata=df)

            # lets read vt data
            ephys_object.read_vt()
            pos_data = np.array([ephys_object.vt_x, ephys_object.vt_y]).transpose()

            # now lets work with that file lazily
            nwbpath = os.path.join(fname,'nwbfile.nwb')

            # read nwb file
            with NWBHDF5IO(nwbpath, "r+") as io:
                print("Reading nwbfile from: ",nwbpath)
                nwbfile = io.read()

                #io = NWBHDF5IO(nwbpath, mode="r")
                #nwbfile = io.read()

                # add position data
                position_spatial_series = SpatialSeries(
                    name="SpatialSeries",
                    description="Position data (x and y) from LED tracking on rats head",
                    data=pos_data,
                    timestamps=ephys_object.vt_t,
                    reference_frame="Neuralynx Time",
                )
                position = Position(spatial_series=position_spatial_series)
                behavior_module = nwbfile.create_processing_module(
                    name="behavior", 
                    description="Processed behavioral data")
                behavior_module.add(position)

                # add trial-specific information
                nwbfile.add_trial_column(name="trials", description="TTL Events")
                for evi in range(len(ephys_object.event_strings)-1):
                    nwbfile.add_trial(
                        start_time=ephys_object.event_times[evi],
                        stop_time=ephys_object.event_times[evi+1],
                        trials=ephys_object.event_strings[evi],
                        tags = ["start time refers to onset of event, stop time is simply the event following it. Ignore stop time"],
                        timeseries=[position_spatial_series],
                    )
                nwbfile.trials.to_dataframe()
                io.write(nwbfile)
                io.close()
        except:
            print("Skipped:", fname)
        if 'nwbfile' in locals():
            del nwbfile




# 
