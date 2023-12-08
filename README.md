# hernan-lab-to-nwb

This package converts neurophysiology data to NWB and it prompts the user to interact with excel files to fill out metadata information!

**Development information:**
* Validated on MAC silicon M2
* Not tested on PC
* Early development
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

**Important Details Regarding Credit** 


These packages/sources were used directly or modified. Neuralynx .nvt file reader from Ben Dichtors CatalystNeuro team.
* neo: https://github.com/NeuralEnsemble/python-neo
* CatalystNeuro: https://github.com/catalystneuro




