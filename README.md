# hernan-lab-to-nwb

Code that supports the conversion of raw data to NWB and extraction of NWB for analysis.

The work-flow of this code requires to run code and interact with excel files.

Current supported formats: 
1) Neuralynx
2) UCLA Miniscope
3) Pinnacle

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
        from hernan_lab_to_nwb import converters
3) Determine what you want to convert. Here we will convert miniscope data. Enter the following into terminal:



