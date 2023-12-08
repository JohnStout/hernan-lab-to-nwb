# hernan-lab-to-nwb

Code that supports the conversion of raw data to NWB and extraction of NWB for analysis.

The work-flow of this code requires to run code and interact with excel files.

Current supported formats: 
1) Neuralynx
    Must have data from one session in a unique folder
2) UCLA Miniscope
    Must have data from one session in a unique folder. In the subfolder with name "miniscope", you must have all recorded movies in that folder from that session! So 0.avi, 1.avi, 2.avi, 3.avi. This code works by loading each file separately, then adding that file to the NWB File. To avoid loading the fullfile to memory, the nwbfile is loaded lazily, then the next movie is added.
3) Pinnacle
    Can have as many animals as you want in a unique folder. Handles multiple simultanoeous recordings across animals


To download:
1) Download anaconda3 if you haven't already
2) conda create -n decode_lab_env python=3.9
3) conda activate decode_lab_env
4) git clone https://github.com/JohnStout/hernan-lab-to-nwb 
5) cd hernan-lab-to-nwb
6) pip install - e.

If you want to convert data in the terminal:
1) Open terminal (if on mac, normal terminal. If on PC, open conda terminal)
2) Enter the following things into the terminal:
        python
        from hernan_lab_to_nwb.converters import convert
4) Determine what you want to convert. Here we will convert miniscope data. Enter the following into terminal:
        IF MINISCOPE:
            dir = ".../miniscope/data/134A/AAV2/3-Syn-GCaMP8f/2023_11_14/13_21_49"
            convert(dir=dir).miniscope()
        IF NEURALYNX
            dir = ".../2020-06-26_16-56-10 9&10eb male ACTH ELS"
            convert(dir=dir).neuralynx()
        IF PINNACLE
            dir = ".../data" or whatever your folder is
            convert(dir=dir).pinnacle()





