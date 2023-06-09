from cProfile import label
from sqlite3 import Connection
from textwrap import fill

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

version = 'ver2'

fig = plt.figure(figsize=(10, 7))
ax =plt.axes()


domain = patches.Rectangle(xy=(0, 0), width=92, height=77, fc='c', label='PML')
basalt_box = patches.Rectangle(xy=(1, 1), width=90, height=75, fc='k', label='Basalt')
tube_box = patches.Rectangle(xy=(31, 31), width=30, height=15, fc='w', label='Tube')
src_horizontal = patches.Arrow(x=38.5, y=32, dx=15, dy=0, width=1, fc='r', label='Source horizontal')
src_vertical = patches.Arrow(x=46, y=41, dx=0, dy=10, width=1, fc='b', label='Source vertical') 

src_x = 38.5
src_y = 32
src_point = patches.Circle(xy=(src_x, src_y), radius=0.5, fc='g', label='Source point')


ax.add_patch(domain) 
ax.add_patch(basalt_box)
ax.add_patch(tube_box)
ax.add_patch(src_horizontal)
#ax.add_patch(src_vertical)
ax.add_patch(src_point)

time = [0.58, 1.03, 1.9, 2.33, 5.13, 6]
tx_time = np.ones_like(time) * 0.088
print(tx_time)

tau = time - tx_time
print(tau)

for i in range(len(tau)):
    range_i = patches.Circle(xy=(src_x, src_y), radius=tau[i]*30/2, ec='g', fill=False, label='tau='+str(time[i]))
    ax.add_patch(range_i)

# set legend at the outside of the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

#plt.title(version, fontsize=20)
ax.set_xlabel('x [m]', fontsize=16)
ax.set_ylabel('y [m]', fontsize=16)

plt.axis('scaled') # 
ax.set_aspect('equal') # this will make x and y axis scaled equally

plt.show()