from cProfile import label
from sqlite3 import Connection
from textwrap import fill

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tight_layout

version = 'ver7'

fig = plt.figure(facecolor='gray',figsize=(17, 7), tight_layout=True)
ax =plt.axes()


domain = patches.Rectangle(xy=(0, 0), width=10, height=10, ec='k',fill=False, label='domain')
basalt_box_1 = patches.Rectangle(xy=(0, 0), width=10, height=2, fc='k', label=r'$\varepsilon_r=6.0$')
basalt_box_2 = patches.Rectangle(xy=(0, 2), width=10, height=2, fc='grey', label=r'$\varepsilon_r=4.0$')
basalt_box_3 = patches.Rectangle(xy=(0, 9), width=10, height=1, fc='k')

src_horizontal = patches.Arrow(x=56, y=32, dx=20, dy=0, width=1, fc='r', label='Source horizontal')
src_vertical = patches.Arrow(x=66, y=45, dx=0, dy=-13, width=1, fc='b', label='Source vertical') 

src_x = 5
src_y = 5
src_point = patches.Circle(xy=(src_x, src_y), radius=0.05, fc='r', label='Source point')


ax.add_patch(domain) 
ax.add_patch(basalt_box_1)
ax.add_patch(basalt_box_2)

ax.add_patch(basalt_box_3)

#ax.add_patch(src_horizontal)
#ax.add_patch(src_vertical)
ax.add_patch(src_point)


time = [0.076, 0.28, 0.35, 0.41, 0.62]
tx_time = 0.009
tx_time_array = np.ones_like(time) * 0.009
#ax.text(domain.get_width+1, 1, 'tx time: '+str(tx_time), fontsize=20)

tau = time - tx_time_array
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