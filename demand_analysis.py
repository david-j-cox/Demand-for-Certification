#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    cox.david.j@gmail.com
    https://www.researchgate.net/profile/David_Cox26
    twitter: @davidjcox_
    LinkedIn: https://www.linkedin.com/in/coxdavidj/
    Website: https://davidjcox.xyz
"""
#Set current working directory to the folder that contains your data.
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# Set path to local scripts
sys.path.append('/Users/davidjcox/Dropbox/Coding/Local Python Modules/')

# Set path to data
os.chdir('/Users/davidjcox/Dropbox/Projects/CurrentProjectManuscripts/Empirical/PersonalFun/Demand for Certification/')

# Change settings to view all columns of data
pd.set_option('display.max_columns', None)

#%% Read in the data
raw_data =pd.read_csv('Raw_Data.csv')
data = raw_data.copy()
data.head()

#%% Fill NaN vals from choices with 1
cols_check = [ '0','25', '50','100','200','300','400','500','600','700','800','900','1000','1100','1200','1300',\
               '1400','1500','1600','1700','1800','1900','2000','2100','2200','2300','2400','2500','2600','2700',\
               '2800','2900','3000',]

for i in cols_check:
    data[i] = data[i].fillna(1)

data.head()

#%% Fit the demand model to the median values

