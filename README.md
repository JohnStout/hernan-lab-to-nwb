# hernan-lab-to-nwb

This package converts neurophysiology data to NWB and it prompts the user to interact with excel files to fill out metadata information!

**Development information:**
* Validated on MAC silicon M2 - only use on mac devices!
* Early testing and development on PC
* Early development and requires additional testing
* Supports Neuralynx, UCLA miniscope, Pinnacle
    * Neuralynx handles .ncs, .ntt, .nvt, .nev files
    * UCLA Miniscope handles .avi and .json files
    * Pinnacle handles multiple, simultaneously recorded animals if relevant. No inputs required.

**Workflow:**
* Define your directory in terminal python shell
* Interact with excel sheets

**Relevant notebook:**
* .../tests/fulltest.ipynb

**Current supported formats:** 
1) Neuralynx:
    * Must have data from one session in a unique folder
2) UCLA Miniscope:
    * Must have data from one session in a unique folder. In the subfolder with name "miniscope", you must have all recorded movies in that folder from that session! So 0.avi, 1.avi, 2.avi, 3.avi.
    * This code works by loading each file separately, then adding that file to the NWB File. To avoid loading the fullfile to memory, the nwbfile is loaded lazily, then the next movie is added.
3) Pinnacle:
    * Can have as many animals as you want in a unique folder. Handles multiple simultanoeous recordings across animals


## Download
1) Download anaconda3 and git.
2) Run the lines below in your terminal (mac) or conda terminal (pc)

```
conda create -n my_env python=3.9
conda activate my_env
git clone https://github.com/JohnStout/hernan-lab-to-nwb 
cd hernan-lab-to-nwb
pip install -e .
```

**Close terminal and re-open upon first installation**

Converting data in terminal:
1) Open terminal (if on mac, normal terminal. If on PC, open conda terminal).
2) Enter the greyed out lines below in your terminal:

Activate your environment
```
conda activate my_env
```

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

**Interface with Terminal**

https://github.com/JohnStout/hernan-lab-to-nwb/assets/46092030/945eccf8-ea7f-4180-99fa-cf133228047c

**Follow Terminal Instructions**

https://github.com/JohnStout/hernan-lab-to-nwb/assets/46092030/18223fd3-a9c0-4d88-8143-3afef965e073

**Interact with Excel to fill out NWB metadata**

https://github.com/JohnStout/hernan-lab-to-nwb/assets/46092030/981d649c-72a7-43f9-b31c-cf4bde871a3b

**Use Excel to define metadata about recording devices**

https://github.com/JohnStout/hernan-lab-to-nwb/assets/46092030/fc78bfa9-50f8-45a6-8d78-fa60972091f3

**Watch Terminal window to monitor progress**

https://github.com/JohnStout/hernan-lab-to-nwb/assets/46092030/84309878-348a-4157-9867-d1133f7c406d

**Important Details Regarding Credit** 

These packages/sources were used directly or modified. Neuralynx .nvt file reader from Ben Dichters CatalystNeuro team.
* neo: https://github.com/NeuralEnsemble/python-neo
* CatalystNeuro: https://github.com/catalystneuro




