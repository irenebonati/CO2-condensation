#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:28:02 2019

@author: irenebonati
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import subprocess

# We only have to change these 3 parameters
start = ["Cold"] # "Cold" or "Warm"
STAR = ["Sun"]
obl = ["23.5"]

ocean = ["0.7"]
igeog = ["1"]
pco2i = ["1."]

if start==["Cold"]:
    icoldflag = ["1"]
else:
    icoldflag = ["0"]

if STAR == ["Sun"]: 
    smass = ["1.0"]
    a0 = ["1.25"]
else:
    smass = ["1.5"]
    a0 = ["2.5"]  


with open("input_ebm.dat", "r") as f:
    main = f.read()
    
splitfile = main.split('\n')

var = ["icoldflag","obl","a0","pco2i","ocean","igeog","STAR","smass"]

n=0

for i in itertools.product(icoldflag,obl,a0,pco2i,ocean,igeog,STAR,smass):
    a,b,c,d,e,f,g,h = i    
    print (i)    
    with open("input_ebm.dat", "r") as f:
        main = f.read()
        for j, line in enumerate(splitfile):
            if var[0] in line:
                splitfile[j] = 'icoldflag:       %s' % i[0]
            if var[1] in line:
                splitfile[j] = 'obl:             %s' % i[1]
            if var[2] in line:
                splitfile[j] = 'a0:              %s' % i[2]            
            if var[3] in line:
                splitfile[j] = 'pco2i:           %s' % i[3] 
            if var[4] in line:
                splitfile[j] = 'ocean:           %s' % i[4]            
            if var[5] in line:
                splitfile[j] = 'igeog:           %s' % i[5]   
            if var[6] in line:
                splitfile[j] = 'STAR:      %s' % i[6]
            if var[7] in line:
                splitfile[j] = 'smass:           %s' % i[7] 
                
        with open("input_ebm.dat", "w") as f:
            f.write('\n'.join(splitfile))
            
        subprocess.call(['./driver'],shell=True)
        
        df = pd.read_csv('fort.50', header=None, sep='\s+')
        
        gl_co2ice = df.iloc[:,0]        
        gl_co2ice = np.array(gl_co2ice)
        time = df.iloc[:,2]
        time = np.array(time)
        
        colors = ['crimson','dodgerblue','grey','green']
        label='%.1f bar'%(np.float(pco2i[n]))
                        
        plt.plot(time,gl_co2ice,c=colors[n],label=label)
        plt.semilogy()
        plt.xlabel('Time (years)')
        plt.ylabel('$CO_{2}$ ice fraction (%)')
        plt.xlim([0,8.5])
        plt.ylim([1e-6,1.5])
        plt.title('%.1f obliquity' %(np.float(obl[0])))
        plt.legend(title="Initial $CO_{2}$ pressure")
        plt.savefig('seas.pdf', bbox_inches='tight',format='pdf')
    n=n+1
plt.show() 

            
            
        
        





































'''
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
'''