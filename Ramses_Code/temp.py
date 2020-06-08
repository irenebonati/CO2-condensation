# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,0.1)

# The lines to plot
y1 = 4 - 2*x
y2 = 3 - 0.5*x
y3 = 1 -x

# The upper edge of polygon (min of lines y1 & y2)
y4 = np.minimum(y1, y2)

# Set y-limit, making neg y-values not show in plot
plt.ylim(0, 5)

# Plotting of lines
plt.plot(x, y1,
         x, y2,
         x, y3,
         x, y4)

# Filling between line y3 and line y4
plt.fill_between(x, y3, y4, color='grey', alpha='0.5')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1:, 1])

y=np.random.uniform(0,1,30)
x=np.arange(30)

ax1.set_ylabel('Plot 1')
ax1.plot(x,y)
ax1.fill_between(x,y,0.5,where=y>0.5,interpolate=True)

ax2.set_ylabel('Plot 2')
ax2.plot(y,x)
ax2.fill_betweenx(x,y, x2=0.5, where=x>0.5,interpolate=True)
ax2.set_ylim(30,0)

plt.show()