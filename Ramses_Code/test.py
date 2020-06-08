#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:40:38 2019

@author: irenebonati
"""

import subprocess
import pandas as pd
import numpy as np
import itertools
from itertools import chain
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Warm start, 23.5 obliquity, igeog = 4 (100% water), igeog = 1 (Earth-like) 
icoldflag = ["0"]
ocean = ["0.7"]
igeog = ["1"]
obl = ["23.5"]
a0 = ["1.0","1.05","1.10","1.15","1.2","1.25","1.3","1.35","1.4","1.45","1.50","1.55","1.60"]
pco2i = ["3.0","2.5","2.0","1.5","1.0","0.5","0.01"]

sem_ax = 13
pco2 = 7

sem_ax_plot = np.linspace(1.,1.60,sem_ax)
sem_ax_plot = np.reshape(sem_ax_plot, (sem_ax))
sem_ax_plot = np.array([sem_ax_plot,]*pco2)

pco2_plot = np.linspace(3,0.01,pco2)
pco2_plot = np.array([pco2_plot,]*sem_ax).transpose()

with open("input_ebm.dat", "r") as f:
    main = f.read()
    
splitfile = main.split('\n')

gl = np.zeros((len(pco2i),len(a0)),dtype=np.float32)
CO2_ice = np.zeros((len(pco2i),len(a0)),dtype=np.float32)
Delta_T = np.zeros((len(pco2i),len(a0)),dtype=np.float32)
T_min = np.zeros((len(pco2i),len(a0)),dtype=np.float32)

s=0 
n=0
var = ["icoldflag","obl","a0","pco2i","ocean","igeog"]

for i in itertools.product(icoldflag,obl,a0,pco2i,ocean,igeog):
    a,b,c,d,e,f = i        
    print i
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
        
        with open("input_ebm.dat", "w") as f:
            f.write('\n'.join(splitfile))
        
        if j==0:
            subprocess.call(['make clean'], shell=True)
            subprocess.call(['make'], shell=True)
            
        subprocess.call(['./driver'],shell=True)

    
        df = pd.read_csv('out/model.out', header=None)
        df = pd.DataFrame(df[0].str.split(' ').tolist())
        
        # CO2 surface ice
        for m in range(len(df)):
            if df.iloc[m,2] == "atmospheric" and df.iloc[m,10]!=None:
                CO2_ice[n,s] = df.iloc[m,10]
                break
            elif df.iloc[m,2] == "atmospheric" and df.iloc[m,10]==None:
                CO2_ice[n,s] = df.iloc[m,9]
                break
        print CO2_ice[n,s]
        
        #Average temperature from text file
        temp_av = df.iloc[56:72,14]
        temp_av = temp_av.append(df.iloc[72:74,15])
        temp_av = temp_av.append(df.iloc[74:76,16])
        temp_av = temp_av.append(df.iloc[76:92,15])
        temp_av = np.array(temp_av, dtype = np.float32)
        
        #Minimum temperature from text file
        temp_min = df.iloc[56:72,20]
        temp_min = temp_min.append(df.iloc[72:74,21])
        temp_min = temp_min.append(df.iloc[74:76,22])
        temp_min = temp_min.append(df.iloc[76:92,21])
        temp_min = np.array(temp_min, dtype = np.float32)
        
        
        # 1: Snowball planet, 3: Ice-free planets, 2: All the rest
        if all(i < 263 for i in temp_av) and CO2_ice[n,s]>0:
            gl[n, s] = 1 # blue
        elif all(k > 263 for k in temp_av):
            gl[n,s] = 3 # red
        else:
            gl[n,s] = 2 # grey
            
        Delta_T[n,s] = np.max(temp_av)-np.min(temp_av)
        T_min[n,s] = np.min(temp_min)

        # Similar to Turbet    
        # 1: Snowball planet, 2: Ice-free planets, 3: All the rest
#        if all(i < 263 for i in temp_av) and CO2_surf>0:
#            gl[n, s] = 1
#        elif all(k < 263 for k in temp_av):
#            gl[n,s] = 2
#        else:
#            gl[n,s] = 3
        
    if n==6:
        n=n-6
        s+=1
    else:
        n+=1

xi = np.linspace(1.0, 1.60, 200)
yi = np.linspace(0.01, 3, 200)

sa_plot = np.reshape(sem_ax_plot, (len(sem_ax_plot[0])*len(sem_ax_plot),1))
sa_plot = sa_plot.tolist()
pc_plot = np.reshape(pco2_plot, (len(pco2_plot[0])*len(pco2_plot),1))
pc_plot = pc_plot.tolist()
gl0 = np.reshape(gl, (len(gl[0])*len(gl),1))
gl0=np.array(gl0).tolist()

x = list(chain.from_iterable(sa_plot))
y = list(chain.from_iterable(pc_plot))
xic = list(chain.from_iterable(gl0))

## Griddata
zi = griddata(x, y, xic, xi, yi, interp='linear')
zi = np.around(zi)

size =10

plt.figure(1)
plt.contourf(xi,yi,zi,levels=[0,1,2,3],cmap='coolwarm', alpha=0.8)
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.0,1.60])
plt.ylim([0,3])
plt.title('Earth-like planet - Warm start 23.5 obl')
plt.savefig('Warm_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(2)
levels=10
CS = plt.contourf(sem_ax_plot, pco2_plot, CO2_ice, levels=[1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,5,1e1,50,100],norm=colors.LogNorm(),cmap='viridis',vmin=1e-3,vmax=1e2)
cb = plt.colorbar(CS,ticks=[1e-3, 1e-2, 1e-1, 1,1e1,1e2])
cb.set_label('Atmospheric CO$_{2}$ loss to surface [%]')
plt.scatter(sem_ax_plot,pco2_plot, marker='o',s=5,color='k')
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.title('Earth-like planet - Warm start 23.5 obl')
plt.xlim([1.0,1.60])
plt.ylim([0,3])
plt.savefig('WarmCO2_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(3)
CS = plt.contourf(sem_ax_plot, pco2_plot, Delta_T , levels=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140],cmap='viridis',vmin=0,vmax=140)
cb = plt.colorbar(CS)
cb.set_label('$T_{eq} - T_{pole}$ [K]')
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.0,1.60])
plt.ylim([0,3])
plt.title('Earth-like planet - Warm start 23.5 obl')
plt.savefig('DeltaTWarm_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(4)
CS = plt.contourf(sem_ax_plot, pco2_plot, T_min, levels=[150,175,200,225,250,275,300,325,350,375,400],cmap='viridis',vmin=150,vmax=400)
cb = plt.colorbar(CS)
cb.set_label('$T_{min}$ [K]')
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.0,1.60])
plt.ylim([0,3])
plt.title('Earth-like planet - Warm start 23.5 obl')
plt.savefig('TminWarm_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

'''
# Cold start
icoldflag = ["1"]
ocean = ["0.7"]
igeog = ["1"]
obl = ["23.5"]
a0 = ["1.00","1.05","1.10","1.15","1.20","1.25","1.30","1.35","1.40","1.46"]
pco2i = ["3.0","2.5","2.0","1.5","1.0","0.5","0.01"]

sem_ax = 10
pco2 = 7

sem_ax_plot = np.linspace(1.0,1.45,sem_ax)
sem_ax_plot = np.reshape(sem_ax_plot, (sem_ax))
sem_ax_plot = np.array([sem_ax_plot,]*pco2)

pco2_plot = np.linspace(3,0.01,pco2)
pco2_plot = np.array([pco2_plot,]*sem_ax).transpose()

with open("input_ebm.dat", "r") as f:
    main = f.read()
    
splitfile = main.split('\n')

gl = np.zeros((len(pco2i),len(a0)),dtype=np.float32)
CO2_ice = np.zeros((len(pco2i),len(a0)),dtype=np.float32)
Delta_T = np.zeros((len(pco2i),len(a0)),dtype=np.float32)
T_min = np.zeros((len(pco2i),len(a0)),dtype=np.float32)

s=0 
n=0
var = ["icoldflag","obl","a0","pco2i","ocean","igeog"]

for i in itertools.product(icoldflag,obl,a0,pco2i,ocean,igeog):
    a,b,c,d,e,f = i        
    print i
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
        
        with open("input_ebm.dat", "w") as f:
            f.write('\n'.join(splitfile))
        
        if j==0:
            subprocess.call(['make clean'], shell=True)
            subprocess.call(['make'], shell=True)
            
        subprocess.call(['./driver'],shell=True)

    
        df = pd.read_csv('out/model.out', header=None)
        df = pd.DataFrame(df[0].str.split(' ').tolist())
        
        # CO2 surface ice
        for m in range(len(df)):
            if df.iloc[m,2] == "atmospheric" and df.iloc[m,10]!=None:
                CO2_ice[n,s] = df.iloc[m,10]
                break
            elif df.iloc[m,2] == "atmospheric" and df.iloc[m,10]==None:
                CO2_ice[n,s] = df.iloc[m,9]
                break
        print CO2_ice[n,s]
        
        #Average temperature from text file
        temp_av = df.iloc[56:72,14]
        temp_av = temp_av.append(df.iloc[72:74,15])
        temp_av = temp_av.append(df.iloc[74:76,16])
        temp_av = temp_av.append(df.iloc[76:92,15])
        temp_av = np.array(temp_av, dtype = np.float32)
        
        #Minimum temperature from text file
        temp_min = df.iloc[56:72,20]
        temp_min = temp_min.append(df.iloc[72:74,21])
        temp_min = temp_min.append(df.iloc[74:76,22])
        temp_min = temp_min.append(df.iloc[76:92,21])
        temp_min = np.array(temp_min, dtype = np.float32)
        
        # 1: Snowball planet, 3: Ice-free planets, 2: All the rest
        if all(i < 263 for i in temp_av) and CO2_ice[n,s]>0:
            gl[n, s] = 1 # blue
        elif all(k > 263 for k in temp_av):
            gl[n,s] = 3 # red
        else:
            gl[n,s] = 2 # grey
            
        Delta_T[n,s] = np.max(temp_av)-np.min(temp_av)
        T_min[n,s] = np.min(temp_min)

        # Similar to Turbet    
        # 1: Snowball planet, 2: Ice-free planets, 3: All the rest
#        if all(i < 263 for i in temp_av) and CO2_surf>0:
#            gl[n, s] = 1
#        elif all(k < 263 for k in temp_av):
#            gl[n,s] = 2
#        else:
#            gl[n,s] = 3
        
    if n==6:
        n=n-6
        s+=1
    else:
        n+=1

xi = np.linspace(1.0, 1.45, 200)
yi = np.linspace(0.01, 3, 200)

sa_plot = np.reshape(sem_ax_plot, (len(sem_ax_plot[0])*len(sem_ax_plot),1))
sa_plot = sa_plot.tolist()
pc_plot = np.reshape(pco2_plot, (len(pco2_plot[0])*len(pco2_plot),1))
pc_plot = pc_plot.tolist()
gl0 = np.reshape(gl, (len(gl[0])*len(gl),1))
gl0=np.array(gl0).tolist()

x = list(chain.from_iterable(sa_plot))
y = list(chain.from_iterable(pc_plot))
xic = list(chain.from_iterable(gl0))

## Griddata
zi = griddata(x, y, xic, xi, yi, interp='linear')
zi = np.around(zi)

size =10

plt.figure(1)
plt.contourf(xi,yi,zi,levels=[0,1,2,3],cmap='coolwarm', alpha=0.8)
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.0,1.45])
plt.ylim([0,3])
plt.title('Earth-like planet - Cold start 23.5 obl')
plt.savefig('Cold_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(2)
levels=10
CS = plt.contourf(sem_ax_plot, pco2_plot, CO2_ice, levels=[1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,5,1e1,50,100],norm=colors.LogNorm(),cmap='viridis',vmin=1e-3,vmax=1e2)
cb = plt.colorbar(CS,ticks=[1e-3,1e-2, 1e-1, 1,1e1,1e2])
cb.set_label('Atmospheric CO$_{2}$ loss to surface (%)')
plt.scatter(sem_ax_plot,pco2_plot, marker='o',s=5,color='k')
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.title('Earth-like planet - Cold start 23.5 obl')
plt.xlim([1.0,1.45])
plt.ylim([0,3])
plt.savefig('ColdCO2_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(3)
CS = plt.contourf(sem_ax_plot, pco2_plot, Delta_T,levels=[0,10,20,30,40,50,60,70,80,90,100],cmap='viridis',vmin=0,vmax=100)
cb = plt.colorbar(CS)
cb.set_label('$T_{eq} - T_{pole}$ [K]')
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.0,1.45])
plt.ylim([0,3])
plt.title('Earth-like planet - Cold start 23.5 obl')
plt.savefig('DeltaTCold_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()

plt.figure(4)
CS = plt.contourf(sem_ax_plot, pco2_plot, T_min,levels=[150,175,200,225,250,275,300,325,350,375,400],cmap='viridis',vmin=150,vmax=400)
cb = plt.colorbar(CS)
cb.set_label('$T_{min}$ [K]')
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.0,1.45])
plt.ylim([0,3])
plt.title('Earth-like planet - Cold start 23.5 obl')
plt.savefig('TminCold_23.5_igeog1.pdf', bbox_inches='tight',format='pdf')
plt.show()
'''