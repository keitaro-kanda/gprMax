from cProfile import label
from sqlite3 import Connection
from textwrap import fill

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

version = 'ver2'

fig = plt.figure(figsize=(10, 7))
ax =plt.axes()


domain = patches.Rectangle(xy=(0, 0), width=92, height=92, fc='c', label='PML')
basalt_box = patches.Rectangle(xy=(1, 1), width=90, height=90, fc='k', label='Basalt')
tube_box = patches.Rectangle(xy=(31, 31), width=30, height=30, fc='w', label='Tube')
src_horizontal = patches.Arrow(x=41, y=46, dx=10, dy=0, width=1, fc='r', label='Source horizontal')
src_vertical = patches.Arrow(x=46, y=41, dx=0, dy=10, width=1, fc='b', label='Source vertical') 
src_point = patches.Circle(xy=(41, 46), radius=0.5, fc='g', label='Source point')


ax.add_patch(domain) 
ax.add_patch(basalt_box)
ax.add_patch(tube_box)
ax.add_patch(src_horizontal)
ax.add_patch(src_vertical)
ax.add_patch(src_point)

time = [0.759, 1.09, 1.29, 1.43, 1.76, 2.09, 2.33]
tx_time = np.ones_like(time) * 0.088
print(tx_time)

tau = time - tx_time
print(tau)

for i in range(5):
    range_i = patches.Circle(xy=(41, 46), radius=tau[i]*30/2, ec='g', fill=False, label='Range'+str(i+1))
    ax.add_patch(range_i)

# set legend at the outside of the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

plt.title(version, fontsize=20)
ax.set_xlabel('x [m]', fontsize=16)
ax.set_ylabel('y [m]', fontsize=16)

plt.axis('scaled') # 
ax.set_aspect('equal') # this will make x and y axis scaled equally

plt.show()