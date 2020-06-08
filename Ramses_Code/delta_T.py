#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 23:21:25 2019

@author: irenebonati
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib.mlab import griddata
from itertools import chain

s=0 
n=0 
          
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
os.chdir("out/")

df = pd.read_csv('model_0.out', header=None)
df = pd.DataFrame(df[0].str.split(' ').tolist())

# Latitude from text file
latitude = df.iloc[56:72,4]
latitude = latitude.append(df.iloc[72:74,5])
latitude = latitude.append(df.iloc[74:76,6])
latitude = latitude.append(df.iloc[76:92,5])
latitude0 = np.array(latitude, dtype = np.float32)


# Average temperature from text file
temp_av = df.iloc[56:72,14]
temp_av = temp_av.append(df.iloc[72:74,15])
temp_av = temp_av.append(df.iloc[74:76,16])
temp_av = temp_av.append(df.iloc[76:92,15])
temp_av0 = np.array(temp_av, dtype = np.float32)

# Minimum temperature
temp_av = df.iloc[56:72,20]
temp_av = temp_av.append(df.iloc[72:74,21])
temp_av = temp_av.append(df.iloc[74:76,22])
temp_av = temp_av.append(df.iloc[76:92,21])
temp_min0 = np.array(temp_av, dtype = np.float32)

df = pd.read_csv('model_23.5.out', header=None)
df = pd.DataFrame(df[0].str.split(' ').tolist())

# Latitude from text file
latitude = df.iloc[56:72,4]
latitude = latitude.append(df.iloc[72:74,5])
latitude = latitude.append(df.iloc[74:76,6])
latitude = latitude.append(df.iloc[76:92,5])
latitude235 = np.array(latitude, dtype = np.float32)


# Average temperature from text file
temp_av = df.iloc[56:72,14]
temp_av = temp_av.append(df.iloc[72:74,15])
temp_av = temp_av.append(df.iloc[74:76,16])
temp_av = temp_av.append(df.iloc[76:92,15])
temp_av235 = np.array(temp_av, dtype = np.float32)

# Minimum temperature
temp_av = df.iloc[56:72,20]
temp_av = temp_av.append(df.iloc[72:74,21])
temp_av = temp_av.append(df.iloc[74:76,22])
temp_av = temp_av.append(df.iloc[76:92,21])
temp_min235 = np.array(temp_av, dtype = np.float32)

plt.figure(1)
plt.plot(latitude0,temp_av0,label='0$^{\circ}$')
plt.plot(latitude235,temp_av235,label='23.5$^{\circ}$')
plt.xlabel('Latitude (degrees)')
plt.ylabel('Average temperature (K)')
plt.legend()
plt.savefig('DT.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(1)
plt.plot(latitude0,temp_min0,label='0$^{\circ}$')
plt.plot(latitude235,temp_min235,label='23.5$^{\circ}$')
plt.xlabel('Latitude (degrees)')
plt.ylabel('Minimum temperature (K)')
plt.legend()
plt.savefig('DTmin.pdf', bbox_inches='tight',format='pdf')
plt.show()

