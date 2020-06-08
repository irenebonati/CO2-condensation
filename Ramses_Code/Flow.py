#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:50:12 2019

@author: irenebonati
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv('out/modelearth.out', header=None)
#df = pd.DataFrame(df[0].str.split(' ').tolist())
#
## Average temperature from text file
#tave = df.iloc[56:72,14]
#tave = tave.append(df.iloc[72:74,15])
#tave = tave.append(df.iloc[74:76,16])
#tave = tave.append(df.iloc[76:92,15])
#tave = np.array(tave, dtype = np.float32)
#
#belts = 36
#D = 0.58
#R = 6.378e6  # Earth radius (m)
#lat = np.arange(-87.5,92.5,5,dtype=None)
#latrad = lat*np.pi/180
#x= np.sin(latrad)
#
#tgrad = np.zeros((belts-1,1),dtype=np.float32)
#Flamearth = np.zeros((belts-1,1),dtype=np.float32)
#Dcalc = np.zeros((belts-1,1),dtype=np.float32)
#
#for i in range(len(tgrad)):
#    tgrad[i]= (tave[i+1] - tave[i])/(x[i+1] - x[i])
#    Flamearth[i] = 2*np.pi*R**2*D*(1 - x[i]**2)*tgrad[i]
#    Flamearth[i] = Flamearth[i]/1e15
#
#lat_plot = np.zeros((belts-1,1),dtype=np.float32)

df = pd.read_csv('out/model.out', header=None)
df = pd.DataFrame(df[0].str.split(' ').tolist())

# Average temperature from text file
tave = df.iloc[56:72,14]
tave = tave.append(df.iloc[72:74,15])
tave = tave.append(df.iloc[74:76,16])
tave = tave.append(df.iloc[76:92,15])
tave = np.array(tave, dtype = np.float32)

belts = 36
D = 0.58
R = 6.378e6  # Earth radius (m)
lat = np.arange(-87.5,92.5,5,dtype=None)
latrad = lat*np.pi/180
x= np.sin(latrad)

tgrad = np.zeros((belts-1,1),dtype=np.float32)
Flam = np.zeros((belts-1,1),dtype=np.float32)
Dcalc = np.zeros((belts-1,1),dtype=np.float32)

for i in range(len(tgrad)):
    tgrad[i]= (tave[i+1] - tave[i])/(x[i+1] - x[i])
    Flam[i] = 2*np.pi*R**2*D*(1 - x[i]**2)*tgrad[i]
    Flam[i] = Flam[i]/1e15

plt.figure(1)

lat_plot = np.zeros((belts-1,1),dtype=np.float32)

for i in range(len(lat)-1):
    lat_plot[i,0] = lat[i]
#plt.plot(lat_plot,Flamearth)
plt.plot(lat_plot,Flam)
plt.xlabel('Latitude (degrees)')
plt.ylabel('Northward Meridional Heat Flux (1e15 W)')
plt.savefig('Flux_23.5_old.pdf', bbox_inches='tight',format='pdf')
plt.show()







