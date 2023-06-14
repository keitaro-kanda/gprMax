from cProfile import label
from sqlite3 import Connection
from textwrap import fill

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tight_layout

version = 'ver2'

fig = plt.figure(facecolor='gray',figsize=(17, 7), tight_layout=True)
ax =plt.axes()


domain = patches.Rectangle(xy=(0, 0), width=132, height=77, fc='c', label='PML')
basalt_box_1 = patches.Rectangle(xy=(1, 1), width=130, height=30, fc='k', label='Basalt')
basalt_box_2 = patches.Rectangle(xy=(1, 46), width=130, height=30, fc='k')

tube_box = patches.Rectangle(xy=(1, 31), width=130, height=15, fc='w', label='Vacuum')
tube_dent = patches.Rectangle(xy=(71, 26), width=5, height=5, fc='w')

src_horizontal = patches.Arrow(x=56, y=32, dx=20, dy=0, width=1, fc='r', label='Source horizontal')
src_vertical = patches.Arrow(x=66, y=45, dx=0, dy=-13, width=1, fc='b', label='Source vertical') 

src_x = 66
src_y = 38.5
src_point = patches.Circle(xy=(src_x, src_y), radius=0.5, fc='g', label='Source point')


ax.add_patch(domain) 
ax.add_patch(basalt_box_1)
ax.add_patch(basalt_box_2)

ax.add_patch(tube_box)
ax.add_patch(tube_dent)

#ax.add_patch(src_horizontal)
ax.add_patch(src_vertical)
#ax.add_patch(src_point)


time = [0.759, 1.09, 1.29, 1.43, 1.76, 2.2, 2.33,  2.5, 2.92, 5.73]
tx_time = np.ones_like(time) * 0.088
print(tx_time)

tau = time - tx_time
print(tau)

for i in range(len(tau)):
    range_i = patches.Circle(xy=(src_x, src_y), radius=tau[i]*30/2, ec='g', fill=False, label='tau='+str(time[i]))
    ax.add_patch(range_i)

# set legend at the outside of the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=20)

#plt.title(version, fontsize=20)
ax.set_xlabel('x [m]', fontsize=16)
ax.set_ylabel('y [m]', fontsize=16)

plt.axis('scaled') # 
ax.set_aspect('equal') # this will make x and y axis scaled equally

plt.show()