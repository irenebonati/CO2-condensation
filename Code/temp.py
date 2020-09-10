# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sympy as sp
import numpy as np

u_Re = np.exp(-(Q_Re*dn_liq/dx*x))
#self.C_Re187[i] = self.C_Re187[0] * np.exp((Q_Re*(dn_liq/dt) - lam_Re)*t)

x = sp.symbols('x')
result = sp.integrate(x **2, x)
result = sp.integrate(u_Re, x)
print (result)


