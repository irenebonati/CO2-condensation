#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:43:08 2019

@author: irenebonati
"""

# This is a script created to plot partial CO2 pressures against semi-major axes
# of Earth-like planets. The regime plot shows blue and red regions, where blue
# regions indicate ice-ball planets with CO2 condensation, and red regions denote
# ice-free bodies. Green regions --> all the rest 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from itertools import chain

pco2 = 7
pco2warm = 7
sem_ax = 8
sem_axwarm = 10

sem_ax_plot = np.linspace(1.10,1.45,sem_ax)
sem_ax_plot = np.reshape(sem_ax_plot, (sem_ax))
sem_ax_plot = np.array([sem_ax_plot,]*pco2)

sem_ax_plotwarm = np.linspace(1.10,1.55,sem_axwarm)
sem_ax_plotwarm = np.reshape(sem_ax_plotwarm, (sem_axwarm))
sem_ax_plotwarm = np.array([sem_ax_plotwarm,]*pco2)

pco2_plot = np.linspace(3,0,pco2)
pco2_plot = np.array([pco2_plot,]*sem_ax).transpose()

pco2_plotwarm = np.linspace(3,0,pco2warm)
pco2_plotwarm = np.array([pco2_plotwarm,]*sem_axwarm).transpose()

gl = np.zeros((pco2,sem_ax),dtype=np.float32)

s=0 
n=0

pressures = np.linspace(3,0,7)
semax = np.linspace(1.10,1.45,8)
semax = np.around(semax, decimals=3)

semaxwarm = np.linspace(1.10,1.55,10)
semaxwarm = np.around(semaxwarm, decimals=3)
          
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

os.chdir("Cold_start/0/")

for i in range(len(semax)):
    n=0
    for j in range(len(pressures)):
        os.chdir("a" + str(semax[i]) +"_p"+str(pressures[j])+"/out")

        df = pd.read_csv('model.out', header=None)
        df = pd.DataFrame(df[0].str.split(' ').tolist())

        # Average temperature from text file
        temp_av = df.iloc[56:72,14]
        temp_av = temp_av.append(df.iloc[72:74,15])
        temp_av = temp_av.append(df.iloc[74:76,16])
        temp_av = temp_av.append(df.iloc[76:92,15])
        temp_av = np.array(temp_av, dtype = np.float32)

        # 1: Snowball planet, 2: Ice-free planets, 3: All the rest
        if all(i < 263 for i in temp_av):# and CO2_surf>0:
            gl[n, s] = 1
        elif all(k > 263 for k in temp_av):
            gl[n,s] = 3
        else:
            gl[n,s] = 2
            
        os.chdir("../../")

        n+=1
    s+=1
    
# FIGURE   
os.chdir("../../")

xi = np.linspace(1.1, 1.45, 200)
yi = np.linspace(3.3e-4, 3, 200)

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
#for i in range(len(sem_ax_plot[0])):
#    for j in range(len(pco2_plot)):
#        if gl[j,i] == 1:
#            plt.scatter(sem_ax_plot[j,i],pco2_plot[j,i],color='dodgerblue', s=size)
#        elif gl[j,i] == 2:
#            plt.scatter(sem_ax_plot[j,i],pco2_plot[j,i],color='limegreen', s=size)
#        else: 
#            plt.scatter(sem_ax_plot[j,i],pco2_plot[j,i],color='orangered', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([1.1,1.45])
plt.ylim([0,3])
plt.title('Cold start - Obliquity: 0$^{\circ}$')
plt.savefig('Cold_0new.pdf', bbox_inches='tight',format='pdf')
plt.show()


#s=0 
#n=0
#
#os.chdir("Cold_start/23.5/")
#
#for i in range(len(semax)):
#    n=0
#    for j in range(len(pressures)):
#        os.chdir("a" + str(semax[i]) +"_p"+str(pressures[j])+"/out")
#
#        df = pd.read_csv('model.out', header=None)
#        df = pd.DataFrame(df[0].str.split(' ').tolist())
#
#        # Average temperature from text file
#        temp_av = df.iloc[56:72,14]
#        temp_av = temp_av.append(df.iloc[72:74,15])
#        temp_av = temp_av.append(df.iloc[74:76,16])
#        temp_av = temp_av.append(df.iloc[76:92,15])
#        temp_av = np.array(temp_av, dtype = np.float32)
#
#        # 1: Snowball planet, 2: Ice-free planets, 3: All the rest
#        if all(i < 263 for i in temp_av):# and CO2_surf>0:
#            gl[n, s] = 1
#        elif all(k > 263 for k in temp_av):
#            gl[n,s] = 3
#        else:
#            gl[n,s] = 2
#            
#        os.chdir("../../")
#
#        n+=1
#    s+=1
#    
## FIGURE   
#os.chdir("../../")
#
#sa_plot = np.reshape(sem_ax_plot, (len(sem_ax_plot[0])*len(sem_ax_plot),1))
#sa_plot = sa_plot.tolist()
#pc_plot = np.reshape(pco2_plot, (len(pco2_plot[0])*len(pco2_plot),1))
#pc_plot = pc_plot.tolist()
#gl0 = np.reshape(gl, (len(gl[0])*len(gl),1))
#gl0=np.array(gl0).tolist()
#
#x = list(chain.from_iterable(sa_plot))
#y = list(chain.from_iterable(pc_plot))
#xic = list(chain.from_iterable(gl0))
#
### Griddata
#zi = griddata(x, y, xic, xi, yi, interp='linear')
#zi = np.around(zi)
#
#plt.figure(2)
#plt.contourf(xi,yi,zi,levels=[0,1,2,3],cmap='coolwarm', alpha=0.8)
#plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
##for i in range(len(sem_ax_plot[0])):
##    for j in range(len(pco2_plot)):
##        if gl[j,i] == 1:
##            plt.scatter(sem_ax_plot[j,i],pco2_plot[j,i],color='dodgerblue', s=size)
##        elif gl[j,i] == 2:
##            plt.scatter(sem_ax_plot[j,i],pco2_plot[j,i],color='limegreen', s=size)
##        else: 
##            plt.scatter(sem_ax_plot[j,i],pco2_plot[j,i],color='orangered', s=size)
#plt.xlabel('Semi-major axis [AU]')
#plt.ylabel('CO$_{2}$ partial pressure [bar]')
#plt.xlim([1.1,1.45])
#plt.ylim([0,3])
#plt.title('Cold start - Obliquity: 23.5$^{\circ}$')
#plt.savefig('Cold_23.5new.pdf', bbox_inches='tight',format='pdf')
#plt.show()


#s=0 
#n=0
#gl = np.zeros((pco2,sem_axwarm),dtype=np.float32)
#
#os.chdir("Warm_start/0/")
#
#for i in range(len(semaxwarm)):
#    n=0
#    for j in range(len(pressures)):
#        os.chdir("a" + str(semaxwarm[i]) +"_p"+str(pressures[j])+"/out")
#
#        df = pd.read_csv('model.out', header=None)
#        df = pd.DataFrame(df[0].str.split(' ').tolist())
#        
#        # CO2 surface ice
#        CO2_surf = df.iloc[94,10]
#        if df.iloc[94,10]==None:
#            CO2_surf = df.iloc[94,9]
#        CO2_surf = np.array(CO2_surf, dtype = np.float32)
#
#        # Average temperature from text file
#        temp_av = df.iloc[56:72,14]
#        temp_av = temp_av.append(df.iloc[72:74,15])
#        temp_av = temp_av.append(df.iloc[74:76,16])
#        temp_av = temp_av.append(df.iloc[76:92,15])
#        temp_av = np.array(temp_av, dtype = np.float32)
#
#        # 1: Snowball planet, 2: Ice-free planets, 3: All the rest
#        if all(i < 263 for i in temp_av) and CO2_surf>0:
#            gl[n, s] = 1
#        elif all(k > 263 for k in temp_av):
#            gl[n,s] = 3
#        else:
#            gl[n,s] = 2
#            
#        os.chdir("../../")
#
#        n+=1
#    s+=1
#    
## FIGURE   
#os.chdir("../../")
#
#xi = np.linspace(1.1, 1.55, 200)
#yi = np.linspace(3.3e-4, 3, 200)
#
#sa_plotwarm = np.reshape(sem_ax_plotwarm, (len(sem_ax_plotwarm[0])*len(sem_ax_plotwarm),1))
#sa_plotwarm = sa_plotwarm.tolist()
#pc_plotwarm = np.reshape(pco2_plotwarm, (len(pco2_plotwarm[0])*len(pco2_plotwarm),1))
#pc_plotwarm = pc_plotwarm.tolist()
#gl0 = np.reshape(gl, (len(gl[0])*len(gl),1))
#gl0=np.array(gl0).tolist()
#
#x = list(chain.from_iterable(sa_plotwarm))
#y = list(chain.from_iterable(pc_plotwarm))
#xic = list(chain.from_iterable(gl0))
#
## Griddata
#zi = griddata(x, y, xic, xi, yi, interp='linear')
#zi = np.around(zi)
#
#plt.figure(3)
#plt.contourf(xi,yi,zi,levels=[0,1,2,3],cmap='coolwarm', alpha=0.8)
#plt.scatter(sem_ax_plotwarm,pco2_plotwarm,color='black', s=size)
##for i in range(len(sem_ax_plotwarm[0])):
##    for j in range(len(pco2_plotwarm)):
##        if gl[j,i] == 1:
##            plt.scatter(sem_ax_plotwarm[j,i],pco2_plotwarm[j,i],color='dodgerblue', s=size)
##        elif gl[j,i] == 2:
##            plt.scatter(sem_ax_plotwarm[j,i],pco2_plotwarm[j,i],color='limegreen', s=size)
##        else: 
##            plt.scatter(sem_ax_plotwarm[j,i],pco2_plotwarm[j,i],color='orangered', s=size)
#plt.xlabel('Semi-major axis [AU]')
#plt.ylabel('CO$_{2}$ partial pressure [bar]')
#plt.title('Warm start - Obliquity: 0$^{\circ}$')
#plt.xlim([1.1,1.55])
#plt.ylim([0,3])
#plt.savefig('Warm_0new.pdf', bbox_inches='tight',format='pdf')
#plt.show()

#s=0 
#n=0
#
#os.chdir("Warm_start/23.5/")
#
#for i in range(len(semaxwarm)):
#    n=0
#    for j in range(len(pressures)):
#        os.chdir("a" + str(semaxwarm[i]) +"_p"+str(pressures[j])+"/out")
#
#        df = pd.read_csv('model.out', header=None)
#        df = pd.DataFrame(df[0].str.split(' ').tolist())
#        
#        # CO2 surface ice
#        CO2_surf = df.iloc[94,10]
#        if df.iloc[94,10]==None:
#            CO2_surf = df.iloc[94,9]
#        CO2_surf = np.array(CO2_surf, dtype = np.float32)
#
#        # Average temperature from text file
#        temp_av = df.iloc[56:72,14]
#        temp_av = temp_av.append(df.iloc[72:74,15])
#        temp_av = temp_av.append(df.iloc[74:76,16])
#        temp_av = temp_av.append(df.iloc[76:92,15])
#        temp_av = np.array(temp_av, dtype = np.float32)
#
#        # 1: Snowball planet, 2: Ice-free planets, 3: All the rest
#        if all(i < 263 for i in temp_av) and CO2_surf>0:
#            gl[n, s] = 1
#        elif all(k > 263 for k in temp_av):
#            gl[n,s] = 3
#        else:
#            gl[n,s] = 2
#            
#        os.chdir("../../")
#
#        n+=1
#    s+=1
#    
## FIGURE   
#os.chdir("../../")
#
#sa_plotwarm = np.reshape(sem_ax_plotwarm, (len(sem_ax_plotwarm[0])*len(sem_ax_plotwarm),1))
#sa_plotwarm = sa_plotwarm.tolist()
#pc_plotwarm = np.reshape(pco2_plotwarm, (len(pco2_plotwarm[0])*len(pco2_plotwarm),1))
#pc_plotwarm = pc_plotwarm.tolist()
#gl0 = np.reshape(gl, (len(gl[0])*len(gl),1))
#gl0=np.array(gl0).tolist()
#
#x = list(chain.from_iterable(sa_plotwarm))
#y = list(chain.from_iterable(pc_plotwarm))
#xic = list(chain.from_iterable(gl0))
#
### Griddata
#zi = griddata(x, y, xic, xi, yi, interp='linear')
#zi = np.around(zi)
#
#plt.figure(4)
#plt.contourf(xi,yi,zi,levels=[0,1,2,3],cmap='coolwarm', alpha=0.8)
#plt.scatter(sem_ax_plotwarm,pco2_plotwarm,color='black', s=size)
##for i in range(len(sem_ax_plotwarm[0])):
##    for j in range(len(pco2_plotwarm)):
##        if gl[j,i] == 1:
##            plt.scatter(sem_ax_plotwarm[j,i],pco2_plotwarm[j,i],color='dodgerblue', s=size)
##        elif gl[j,i] == 2:
##            plt.scatter(sem_ax_plotwarm[j,i],pco2_plotwarm[j,i],color='limegreen', s=size)
##        else: 
##            plt.scatter(sem_ax_plotwarm[j,i],pco2_plotwarm[j,i],color='orangered', s=size)
#plt.xlabel('Semi-major axis [AU]')
#plt.ylabel('CO$_{2}$ partial pressure [bar]')
#plt.xlim([1.1,1.55])
#plt.ylim([0,3])
#plt.title('Warm start - Obliquity: 23.5$^{\circ}$')
#plt.savefig('Warm_23.5new.pdf', bbox_inches='tight',format='pdf')
#plt.show()
#
#
#
