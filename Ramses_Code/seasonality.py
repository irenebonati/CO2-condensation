#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:28:02 2019

@author: irenebonati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('fort_0_0.5.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_05 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_05 = np.array(time, dtype = np.float32)

df = pd.read_csv('fort_0_1.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_1 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_1 = np.array(time, dtype = np.float32)

df = pd.read_csv('fort_0_3.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_3 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_3 = np.array(time, dtype = np.float32)

plt.figure(1)
plt.plot(time_05,gl_co2ice_0_05,label='0.1 bar',c='crimson')
plt.plot(time_1,gl_co2ice_0_1,label='1 bar',c='dodgerblue')
plt.plot(time_3,gl_co2ice_0_3,label='3 bar',c='grey')
plt.xlabel('Time (years)')
plt.ylabel('Global $CO_{2}$ ice fraction')
plt.title('0$^{\circ}$ obliquity')
#plt.legend(title="Initial $CO_{2}$ pressure", fancybox=True)
plt.semilogy()
plt.xlim([0,8.5])
plt.ylim([1e-6,1.5])
plt.savefig('seas_0.pdf', bbox_inches='tight',format='pdf')
plt.show()



df = pd.read_csv('fort.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_05 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_05 = np.array(time, dtype = np.float32)

df = pd.read_csv('fort_23.5_1.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_1 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_1 = np.array(time, dtype = np.float32)

df = pd.read_csv('fort_23.5_3.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_3 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_3 = np.array(time, dtype = np.float32)

plt.figure(2)
ax1 = plt.gca()
plt.plot(time_05,gl_co2ice_0_05,label='0.1',c='crimson')
plt.plot(time_1,gl_co2ice_0_1,label='1',c='dodgerblue')
plt.plot(time_3,gl_co2ice_0_3,label='3',c='grey')
plt.xlabel('Time (years)')
#plt.ylabel('Global $CO_{2}$ ice fraction')
plt.setp(ax1.get_yticklabels(), visible=False)
plt.title('23.5$^{\circ}$ obliquity')
plt.semilogy()
plt.xlim([0,8.5])
plt.ylim([1e-6,1.5])
plt.savefig('seas_23.5.pdf', bbox_inches='tight',format='pdf')
plt.show()




df = pd.read_csv('fort.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_05 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_05 = np.array(time, dtype = np.float32)

df = pd.read_csv('fort_45_1.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_1 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_1 = np.array(time, dtype = np.float32)

df = pd.read_csv('fort_45_3.50', header=None)
df = pd.DataFrame(df[0].str.split('\s+', expand = True))


gl_co2ice = df.iloc[:,1]
gl_co2ice_0_3 = np.array(gl_co2ice, dtype = np.float32)

time = df.iloc[:,2]
time_3 = np.array(time, dtype = np.float32)

plt.figure(3)
ax1 = plt.gca()
plt.plot(time_05,gl_co2ice_0_05,label='0.1',c='crimson')
plt.plot(time_1,gl_co2ice_0_1,label='1',c='dodgerblue')
plt.plot(time_3,gl_co2ice_0_3,label='3',c='grey')
plt.xlabel('Time (years)')
#plt.ylabel('Global $CO_{2}$ ice fraction')
plt.setp(ax1.get_yticklabels(), visible=False)
plt.title('45$^{\circ}$ obliquity')
plt.semilogy()
plt.xlim([0,8.5])
plt.ylim([1e-6,1.5])
plt.savefig('seas_45.pdf', bbox_inches='tight',format='pdf')
plt.show()