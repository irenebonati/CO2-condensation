# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#
#Fe = [20,30,40,50,60,70,80] #20,30,40,50,60,70,80
#Masses = [0.8,1,1.2,1.5,1.8,2]
#MF08 = [2.38,2.75,2.94,3.31,3.67,2.90,2.80]
#MF1 = [2.96,3.19,3.46,3.89,4.37,4.70,5]
#Fe12 = [20,30,40,50,60]
#MF12 = [3.22,3.55,3.64,4.28,5] 
#MF15 = [3.50,3.93,4.24,5]
#Fe18 = [20,30,40,50]
#MF18 = [3.56,4.13,4.60,5]
#Fe2 = [20,30,40]
#MF2 = [3.75,4.35,5.]
#
#plt.plot(Fe,MF08,marker=".",linestyle='--',color='gold',label='0.8 $M_{\mathrm{E}}$')
#plt.plot(Fe,MF1,marker=".",linestyle='--',color='yellowgreen',label='1 $M_{\mathrm{E}}$')
#plt.plot(Fe12, MF12,marker=".",linestyle='--',color='limegreen',label='1.2 $M_{\mathrm{E}}$')
#plt.plot(Fe18,MF15,marker=".",linestyle='--',color='mediumaquamarine',label='1.5 $M_{\mathrm{E}}$')
#plt.plot(Fe18, MF18,marker=".",linestyle='--',color='teal',label='1.8 $M_{\mathrm{E}}$')
#plt.plot(Fe2, MF2,marker=".",linestyle='--',label='2 $M_{\mathrm{E}}$')
##plt.plot(Fe,MF08,Fe,MF1, Fe12, MF12, Fe18,MF15, Fe18, MF18, Fe2, MF2,linestyle='--')
#plt.ylabel('Magnetic field lifetime (Gyr)')
#plt.xlabel('Fe content (wt%)')
#plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.02))
#plt.ylim([2,5.1])
#plt.savefig('MF.pdf', bbox_inches='tight',format='pdf')
#plt.show()
#
#Fe = [20,30,40,50,60,70,80] #20,30,40,50,60,70,80
#MF08 = [0.6,0.78,1.1,1.2,1.36,1.25,1.2]
#
#MF1 = [0.7,1,1.5,1.8,2.0,2.1,2.3]
#MF12 = [1.1,1.3,1.63,2.1,2.3,2.6,2.85] 
#MF15 = [1.25,1.55,2.12,2.45,3.3,3.65,3.8,]
#MF18 = [1.55,2,2.8,3.3,4,4.15,4.44]
#MF2 = [1.9,2.3,3.2,4.,4.45,5,5.4]
#
#
#plt.plot(Fe,MF08,marker=".",linestyle='--',color='gold',label='0.8 $M_{\mathrm{E}}$')
#plt.plot(Fe,MF1,marker=".",linestyle='--',color='yellowgreen',label='1 $M_{\mathrm{E}}$')
#plt.plot(Fe, MF12,marker=".",linestyle='--',color='limegreen',label='1.2 $M_{\mathrm{E}}$')
#plt.plot(Fe,MF15,marker=".",linestyle='--',color='mediumaquamarine',label='1.5 $M_{\mathrm{E}}$')
#plt.plot(Fe, MF18,marker=".",linestyle='--',color='teal',label='1.8 $M_{\mathrm{E}}$')
#plt.plot(Fe, MF2,marker=".",linestyle='--',label='2 $M_{\mathrm{E}}$')
##plt.plot(Fe,MF08,Fe,MF1, Fe12, MF12, Fe18,MF15, Fe18, MF18, Fe2, MF2,linestyle='--')
#plt.ylabel('Magnetic moment (relative to present Earth)')
#plt.xlabel('Fe content (wt%)')
##plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.02))
#plt.ylim([0,5.7])
#plt.savefig('MF_strength.pdf', bbox_inches='tight',format='pdf')
#plt.show()


age_Earth = 4.55e9
timesteps = 456
f = 0.055
age = 1e9
t = np.linspace(age_Earth,0, timesteps)
def sqrt_growth(age):
    t = np.linspace(age_Earth,0, timesteps)
    a = np.zeros_like(t)
    for i in range(len(t)):
        if t[i]>age:
            a[i] = 0.
        else: 
            t[i] = -(t[i] - age)
            #a[i] = (f/np.sqrt(age)) * np.sqrt(t[i])
            
            a[i] = (f/np.sqrt(age)) * np.sqrt(t[i]) 
    return a

def lin_growth(age):
    t = np.linspace(age_Earth,0, timesteps)
    for i in range(len(t)):
        if t[i]>age:
            b = 0.
        else:
            b = (-f/age)*t + f         
    return b

test = sqrt_growth(age)
print (test)

plt.plot(t,test)
plt.show()



