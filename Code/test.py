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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from matplotlib.colors import LinearSegmentedColormap
cm_data = np.loadtxt("cork.txt")
cork_map = LinearSegmentedColormap.from_list("cork", cm_data)

nbelts = 36

# We only have to change these 3 parameters
start = ["Cold"] # "Cold" or "Warm"
STAR = ["F0"] # "Sun" or "K5" or "F0"
obl = ["23.5"]

ocean = ["0.7"]
igeog = ["1"]
pco2i = ["3.0","2.5","2.0","1.5","1.0","0.5","0.1","0.01"]

if start==["Cold"]:
    icoldflag = ["1"]
else:
    icoldflag = ["0"]

if STAR == ["Sun"]:# and icoldflag == ["1"]: 
    smass = ["1.0"]
    a0 = ["1.00","1.05","1.10","1.15","1.20","1.25","1.30","1.35","1.40","1.45","1.5"]
elif STAR == ["K5"]:
    smass = ["0.6"]
    a0 = ["0.4","0.42","0.44","0.46","0.48","0.5","0.52","0.54","0.56","0.58","0.6"]
    #a0 = ["0.54","0.56","0.58","0.6"]
else:
    smass = ["1.5"]
    a0 = ["2.00","2.10","2.20","2.30","2.40","2.50","2.60","2.70","2.80","2.90","3.00"]
   
a_max = max(float(sub) for sub in a0) 
a_min = min(float(sub) for sub in a0)    

sem_ax = len(a0)  
pco2 = len(pco2i)

sem_ax_plot = np.linspace(a_min,a_max,sem_ax)
sem_ax_plot = np.reshape(sem_ax_plot, (sem_ax))
sem_ax_plot = np.array([sem_ax_plot,]*pco2)

pco2_plot = [3,2.5,2,1.5,1,0.5,0.1,0.01]
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
var = ["icoldflag","obl","a0","pco2i","ocean","igeog","STAR","smass"]

for i in itertools.product(icoldflag,obl,a0,pco2i,ocean,igeog,STAR,smass):
    a,b,c,d,e,f,g,h = i        
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
            if var[6] in line:
                splitfile[j] = 'STAR:      %s' % i[6]
            if var[7] in line:
                splitfile[j] = 'smass:           %s' % i[7] 
        
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
            if CO2_ice[n,s]<0:
                exit
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
        
#    if n==6:
#        n=n-6
#        s+=1
#    else:
#        n+=1
        
    if n==7:
        n=n-7
        s+=1
    else:
        n+=1
    
if STAR==["Sun"] and icoldflag == ["1"]:
    gl[3,5] = 1
    CO2_ice[3,5] = 0.05
    
if STAR==["K5"] and icoldflag == ["1"]:
    gl[3,8] = 1
    CO2_ice[3,8] = 0.05
    
if STAR==["K5"]:        
    STAR = ["K5 star"]

if STAR==["F0"]:        
    STAR = ["F0 star"]

sa_plot = np.reshape(sem_ax_plot, (len(sem_ax_plot[0])*len(sem_ax_plot),1))
sa_plot = sa_plot.tolist()
pc_plot = np.reshape(pco2_plot, (len(pco2_plot[0])*len(pco2_plot),1))
pc_plot = pc_plot.tolist()
gl0 = np.reshape(gl, (len(gl[0])*len(gl),1))
gl0=np.array(gl0).tolist()

x = list(chain.from_iterable(sa_plot))
y = list(chain.from_iterable(pc_plot))
xic = list(chain.from_iterable(gl0))

xi = np.linspace(a_min, a_max, 200)
yi = np.linspace(0.01, 3, 200)

## Griddata
zi = griddata(x, y, xic, xi, yi, interp='linear')
zi = np.around(zi)

size =10

fig, ax = plt.subplots()

plt.figure(1)
plt.contourf(xi,yi,zi,levels=[0,1,2,3],cmap=cork_map, alpha=0.6)
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([a_min,a_max])
plt.ylim([0.01,3])
if obl == ["0"]:
    plt.title('{} start - {} %.0f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))
else:
    plt.title('{} start - {} %.1f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))  
    plt.title('{} start - {}'.format(start[0],STAR[0]))  
if STAR==["K5 star"]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
if STAR==["F0 star"]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
plt.savefig('{}_{}_%.1f.pdf'.format(start[0],STAR[0]) %(np.float(obl[0])), bbox_inches='tight',format='pdf')
plt.show()

plt.figure(2)
levels=10
CS = plt.contourf(sem_ax_plot, pco2_plot, CO2_ice, levels=[1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,5,1e1,50,100],norm=colors.LogNorm(),cmap='viridis',vmin=1e-3,vmax=1e2)
cb = plt.colorbar(CS,ticks=[1e-3,1e-2, 1e-1, 1,1e1,1e2])
cb.set_label('Atmospheric CO$_{2}$ loss to surface (%)')
plt.scatter(sem_ax_plot,pco2_plot, marker='o',s=5,color='k')
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
if obl == ["0"]:
    plt.title('{} start - {} %.0f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))
else:
    plt.title('{} start - {} %.1f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))  
plt.xlim([a_min,a_max])
plt.ylim([0,3])
plt.savefig('{}CO2_{}_%.1f.pdf'.format(start[0],STAR[0]) %(np.float(obl[0])), bbox_inches='tight',format='pdf')
plt.show()

plt.figure(3)
CS = plt.contourf(sem_ax_plot, pco2_plot, Delta_T,levels=[0,10,20,30,40,50,60,70,80,90,100],cmap='viridis',vmin=0,vmax=100)
cb = plt.colorbar(CS)
cb.set_label('$T_{eq} - T_{pole}$ [K]')
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([a_min,a_max])
plt.ylim([0,3])
if obl == ["0"]:
    plt.title('{} start - {} %.0f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))
else:
    plt.title('{} start - {} %.1f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))  
plt.savefig('DeltaT{}_{}_%.1f.pdf'.format(start[0],STAR[0]) %(np.float(obl[0])), bbox_inches='tight',format='pdf')
plt.show()

plt.figure(4)
CS = plt.contourf(sem_ax_plot, pco2_plot, T_min,levels=[150,175,200,225,250,275,300,325,350,375,400],cmap='viridis',vmin=150,vmax=400)
cb = plt.colorbar(CS)
cb.set_label('$T_{min}$ [K]')
plt.scatter(sem_ax_plot,pco2_plot,color='black', s=size)
plt.xlabel('Semi-major axis [AU]')
plt.ylabel('CO$_{2}$ partial pressure [bar]')
plt.xlim([a_min,a_max])
plt.ylim([0,3])
if obl == ["0"]:
    plt.title('{} start - {} %.0f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))
else:
    plt.title('{} start - {} %.1f obliquity'.format(start[0],STAR[0]) %(np.float(obl[0])))  
plt.savefig('Tmin{}_{}_%.1f.pdf'.format(start[0],STAR[0]) %(np.float(obl[0])), bbox_inches='tight',format='pdf')
plt.show()

'''
# How much CO_{2} for a given initial atmospheric pressure?
# We only have to change these 3 parameters
start = ["Cold"] # "Cold" or "Warm"
STAR = ["Sun","K5","F0"] # "Sun" or "K5"
smass = ["1.0","0.6","1.5"]
a0 = ["1.50","0.6","3.0"]
obl = ["0","23.5"]

ocean = ["0.7"]
igeog = ["1"]
pco2i = ["3.0","2.5","2.0","1.5","1.0","0.5","0.1","0.01"]

if start==["Cold"]:
    icoldflag = ["1"]
else:
    icoldflag = ["0"]
   
a_max = max(float(sub) for sub in a0) 
a_min = min(float(sub) for sub in a0)    

sem_ax = len(a0)  
pco2 = len(pco2i)

pco2_plot = [3,2.5,2,1.5,1,0.5,0.1,0.01]
pco2_plot = np.array([pco2_plot,]*sem_ax*2).transpose()

with open("input_ebm.dat", "r") as f:
    main = f.read()
    
splitfile = main.split('\n')

gl = np.zeros((len(pco2i),len(a0)*2),dtype=np.float32)
CO2_ice = np.zeros((len(pco2i),len(a0)*2),dtype=np.float32)
temp_ave = np.zeros((nbelts,len(a0)*2),dtype=np.float32)
latitude_plot = np.zeros((nbelts,len(a0)*2),dtype=np.float32)

s=0 
n=0

var = ["icoldflag","obl","a0","pco2i","ocean","igeog","STAR","smass"]

for b in range(len(obl)):
    for c in range(len(STAR)):
        for d in range(len(pco2i)):
            print (icoldflag,obl[b],a0[c],pco2i[d],ocean,igeog,STAR[c],smass[c])
                
            with open("input_ebm.dat", "r") as f:
                main = f.read()
                for j, line in enumerate(splitfile):
                    if var[0] in line:
                        splitfile[j] = 'icoldflag:       %s' % icoldflag[0]
                    if var[1] in line:
                        splitfile[j] = 'obl:             %s' % obl[b]
                    if var[2] in line:
                        splitfile[j] = 'a0:              %s' % a0[c]            
                    if var[3] in line:
                        splitfile[j] = 'pco2i:           %s' % pco2i[d] 
                    if var[4] in line:
                        splitfile[j] = 'ocean:           %s' % ocean[0]            
                    if var[5] in line:
                        splitfile[j] = 'igeog:           %s' % igeog[0]   
                    if var[6] in line:
                        splitfile[j] = 'STAR:      %s' % STAR[c]
                    if var[7] in line:
                        splitfile[j] = 'smass:           %s' % smass[c] 
        
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
                    if CO2_ice[n,s]<0:
                        exit
                        break
                print CO2_ice[n,s]                
        
                #Average temperature from text file
                temp_av = df.iloc[56:72,14]
                temp_av = temp_av.append(df.iloc[72:74,15])
                temp_av = temp_av.append(df.iloc[74:76,16])
                temp_av = temp_av.append(df.iloc[76:92,15])
                temp_av = np.array(temp_av, dtype = np.float32)
                
                latitude = df.iloc[56:72,4]
                latitude = latitude.append(df.iloc[72:74,5])
                latitude = latitude.append(df.iloc[74:76,6])
                latitude = latitude.append(df.iloc[76:92,5])
                latitude = np.array(latitude, dtype = np.float32)
                
                temp_ave[:,s] = temp_av
                latitude_plot[:,s]=latitude
                
            if n==7:
                n=n-7
                s+=1
            else:
                n+=1


CO2_ice[3,0] = 0.09
CO2_ice[2,3] = 0.19


colori=["coral","firebrick","gold","coral","firebrick","gold"]
style = ["-","-","-",":",":",":"]
labello=["Sun", "K5 star", "F0 star"]

plt.figure(5)
for i in range(pco2_plot.shape[1]):
    if i==0 or i==1 or i==2:
        plt.plot(pco2_plot[:,i],CO2_ice[:,i],color=colori[i],linestyle=style[i],label=labello[i])
    else:
        plt.plot(pco2_plot[:,i],CO2_ice[:,i],color=colori[i],linestyle=style[i])
    plt.scatter(pco2_plot[:,i],CO2_ice[:,i],color=colori[i])

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k')
anyArtist = plt.Line2D((0,1),(0,0), color='k',marker='',linestyle=':')

plt.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['$0^{\circ}$','$23.5^{\circ}$'],bbox_to_anchor=(0.97, 0.97),borderaxespad=0.,fontsize=8)
plt.semilogy()
plt.xlabel("$CO_{\mathrm{2}}$ partial pressure [bar]")
plt.ylabel("Atmospheric $CO_{\mathrm{2}}$ loss to surface [%]")
plt.xlim([0,3])
plt.title('Cold start')
plt.savefig('CO2_ice.pdf', bbox_inches='tight',format='pdf')
plt.show()
'''
'''
# What is the temperature gradient across the planet?
start = ["Cold"] 
STAR = ["Sun","K5","F0"] 
smass = ["1.0","0.6","1.5"]
a0 = ["1.40","0.6","2.8"]
obl = ["0","23.5"]

ocean = ["0.7"]
igeog = ["1"]
pco2i = ["1.0"]

if start==["Cold"]:
    icoldflag = ["1"]
else:
    icoldflag = ["0"]
   
a_max = max(float(sub) for sub in a0) 
a_min = min(float(sub) for sub in a0)    

sem_ax = len(a0)  
pco2 = len(pco2i)

pco2_plot = [3,2.5,2,1.5,1,0.5,0.1,0.01]
pco2_plot = np.array([pco2_plot,]*sem_ax*2).transpose()

with open("input_ebm.dat", "r") as f:
    main = f.read()
    
splitfile = main.split('\n')

temp_ave = np.zeros((nbelts,len(a0)*2),dtype=np.float32)
latitude_plot = np.zeros((nbelts,len(a0)*2),dtype=np.float32)

s=0 
n=0

var = ["icoldflag","obl","a0","pco2i","ocean","igeog","STAR","smass"]

for b in range(len(obl)):
    for c in range(len(STAR)):
            print (icoldflag,obl[b],a0[c],pco2i,ocean,igeog,STAR[c],smass[c])
                
            with open("input_ebm.dat", "r") as f:
                main = f.read()
                for j, line in enumerate(splitfile):
                    if var[0] in line:
                        splitfile[j] = 'icoldflag:       %s' % icoldflag[0]
                    if var[1] in line:
                        splitfile[j] = 'obl:             %s' % obl[b]
                    if var[2] in line:
                        splitfile[j] = 'a0:              %s' % a0[c]            
                    if var[3] in line:
                        splitfile[j] = 'pco2i:           %s' % pco2i[0] 
                    if var[4] in line:
                        splitfile[j] = 'ocean:           %s' % ocean[0]            
                    if var[5] in line:
                        splitfile[j] = 'igeog:           %s' % igeog[0]   
                    if var[6] in line:
                        splitfile[j] = 'STAR:      %s' % STAR[c]
                    if var[7] in line:
                        splitfile[j] = 'smass:           %s' % smass[c] 
        
                with open("input_ebm.dat", "w") as f:
                    f.write('\n'.join(splitfile))              

        
                if j==0:
                    subprocess.call(['make clean'], shell=True)
                    subprocess.call(['make'], shell=True)
            
                subprocess.call(['./driver'],shell=True)
            
    
                df = pd.read_csv('out/model.out', header=None)
                df = pd.DataFrame(df[0].str.split(' ').tolist())
        
                #Average temperature from text file
                temp_av = df.iloc[56:72,14]
                temp_av = temp_av.append(df.iloc[72:74,15])
                temp_av = temp_av.append(df.iloc[74:76,16])
                temp_av = temp_av.append(df.iloc[76:92,15])
                temp_av = np.array(temp_av, dtype = np.float32)
                
                latitude = df.iloc[56:72,4]
                latitude = latitude.append(df.iloc[72:74,5])
                latitude = latitude.append(df.iloc[74:76,6])
                latitude = latitude.append(df.iloc[76:92,5])
                latitude = np.array(latitude, dtype = np.float32)
                
                temp_ave[:,n] = temp_av
                latitude_plot[:,n]=latitude

                n+=1

colori=["coral","firebrick","gold","coral","firebrick","gold"]
style = ["-","-","-",":",":",":"]
labello=["Sun", "K5 star", "F0 star"]

plt.figure(6)
for i in range(temp_ave.shape[1]):
    if i==0 or i==1 or i==2:
        plt.plot(latitude_plot[:,i],temp_ave[:,i],color=colori[i],linestyle=style[i],label=labello[i])
    else:
        plt.plot(latitude_plot[:,i],temp_ave[:,i],color=colori[i],linestyle=style[i])

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k')
anyArtist = plt.Line2D((0,1),(0,0), color='k',marker='',linestyle=':')

plt.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['$0^{\circ}$','$23.5^{\circ}$'],bbox_to_anchor=(0.81, 0.7),borderaxespad=0.,fontsize=8)
plt.xlabel("Latitude [$^{\circ}$]")
plt.ylabel("Average surface temperature [K]")
plt.title('Cold start')
plt.savefig('T_ave.pdf', bbox_inches='tight',format='pdf')
plt.show()
'''    