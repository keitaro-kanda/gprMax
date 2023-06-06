from cProfile import label
from textwrap import fill

import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
ax =plt.axes()


domain = patches.Rectangle(xy=(0, 0), width=72, height=52, fc='c', label='PML')
basalt_box = patches.Rectangle(xy=(1, 1), width=70, height=50, fc='k', label='Basalt')
tube_box = patches.Rectangle(xy=(21, 21), width=30, height=10, fc='grey', label='Tube')
src_point = patches.Circle(xy=(36, 26), radius=0.5, fc='m', label='Source')
range_1 = patches.Circle(xy=(36, 26), radius=4.95, ec='r', fill=False, label='Phase: +-')
range_2 = patches.Circle(xy=(36, 26), radius=19.5, ec='r', fill=False)
range_3 = patches.Circle(xy=(36, 26), radius=31.5, ec='r', fill=False)
range_4 = patches.Circle(xy=(36, 26), radius=11.4, ec='b', fill=False, label='Phase: -+')
range_5 = patches.Circle(xy=(36, 26), radius=16.5, ec='b', fill=False)
range_6 = patches.Circle(xy=(36, 26), radius=22.5, ec='b', fill=False)

ax.add_patch(domain) 
ax.add_patch(basalt_box)
ax.add_patch(tube_box)
ax.add_patch(src_point)
ax.add_patch(range_1)
ax.add_patch(range_2)
ax.add_patch(range_3)
ax.add_patch(range_4)
ax.add_patch(range_5)
ax.add_patch(range_6)
# set legend at the outside of the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

plt.axis('scaled') # 
ax.set_aspect('equal') # this will make x and y axis scaled equally

plt.show()