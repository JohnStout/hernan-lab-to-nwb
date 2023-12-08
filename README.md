# hernan-lab-to-nwb

Code that supports the conversion of raw data to NWB and extraction of NWB for analysis.

Still in development* and validated on MAC apple silicon.

The work-flow of this code requires to run code and interact with excel files.

Current supported formats: 
1) Neuralynx:
    Must have data from one session in a unique folder
2) UCLA Miniscope:
    Must have data from one session in a unique folder. In the subfolder with name "miniscope", you must have all recorded movies in that folder from that session! So 0.avi, 1.avi, 2.avi, 3.avi. This code works by loading each file separately, then adding that file to the NWB File. To avoid loading the fullfile to memory, the nwbfile is loaded lazily, then the next movie is added.
3) Pinnacle:
    Can have as many animals as you want in a unique folder. Handles multiple simultanoeous recordings across animals


To download:
1) Download anaconda3 and git.
2) Run the lines below in your terminal (mac) or conda terminal (pc)

```
conda create -n my_env python=3.9
conda activate my_env
git clone https://github.com/JohnStout/hernan-lab-to-nwb 
cd hernan-lab-to-nwb
pip install -e .
``` 

If you want to convert data in the terminal:
1) Open terminal (if on mac, normal terminal. If on PC, open conda terminal)
2) Enter the greyed out lines below in your terminal:

Start python
```
python
```

Import package
```
from hernan_lab_to_nwb.converters import convert
```

Enter directory

```
dir = "YOUR DATA DIRECTORY GOES IN REPLACEMENT OF THESE WORDS"
```

IF MINISCOPE:

```
convert(dir).miniscope()
```
           
IF NEURALYNX:

```
convert(dir).neuralynx()
```

IF PINNACLE:

```
convert(dir).pinnacle()
```





