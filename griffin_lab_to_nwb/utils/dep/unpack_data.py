#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This code is meant to support opening of .tgz files
-JS
"""

import tarfile
def extract_tar(pathName,dataName):
    # pathName = '/Users/js0403/Documents/DECODE-lab-code/calcium_imaging_analyses/datasets/'
    # dataName = 'FN_dataSharing.tgz-aa'
    dataDir = pathName+dataName
    file = tarfile.open(dataDir)
    file.extractall(pathName)
    file.close()
