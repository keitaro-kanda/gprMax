import argparse
import os

import h5py
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='get epsilon_r map from .h5 file', 
                                 usage='cd gprMax; python -m tools.get_epsilon_map file_name -closeup')
parser.add_argument('file_name', help='.h5 file name')
args = parser.parse_args()


class epsilon_map():
    def __init__(self):
        self.h5_file_name = ''
    
    # read .h5 file
    def read_h5_file(self):
        self.h5file = h5py.File(self.h5_file_name, 'r')

        # read ID
        ID = self.h5file['ID'][:, 0, 0, 0]
    
    def get_epsilon_map(self):
        # read epsilon_r
        self.h5_data = self.h5file['data'][:, :, 0]
        self.h5_data = np.rot90(self.h5_data)


        z_num = self.h5_data.shape[0] # number of grids in z axis
        x_num = self.h5_data.shape[1] # number of grids in x axis
        self.epsilon_map = np.zeros((z_num, x_num))

        # get epsilon_r map
        self.epsilon_vacuum = 1 # epsilon_r of vacuum
        self.epsilon_regolith = 4 # epsilon_r of regolith
        self.epsilon_basalt6 = 6 # epsilon_r of basalt
        self.epsilon_basalt7 = 7 # epsilon_r of basalt

        for i in tqdm(range(self.h5_data.shape[1])):
            for j in range(self.h5_data.shape[0]):
                if self.h5_data[j, i] == 1:
                    self.epsilon_map[j, i] = self.epsilon_vacuum
                elif self.h5_data[j, i] == 2:
                    self.epsilon_map[j, i] = self.epsilon_regolith
                elif self.h5_data[j, i] == 3:
                    self.epsilon_map[j, i] = self.epsilon_basalt6
                elif self.h5_data[j, i] == 4:
                    self.epsilon_map[j, i] = self.epsilon_basalt7
                else:
                    print('error, input correct ID')

map = epsilon_map()
map.h5_file_name = args.file_name
map.read_h5_file()
map.get_epsilon_map()


# =====output dir path=====
input_path = os.path.dirname(map.h5_file_name)
output_path = os.path.join(input_path, 'map')
if not os.path.exists(output_path):
    os.mkdir(output_path)


# =====save map as txt file=====
#np.savetxt(output_path+'/' + 'h5_data.csv', map.h5_data, delimiter=',')
np.savetxt(output_path+'/' + 'epsilon_map.txt', map.epsilon_map, delimiter=',')


# =====plot map=====
spatial_grid =  0.05 # spatial grid size [m]


fig = plt.figure(figsize=(5, 5*map.epsilon_map.shape[0]/map.epsilon_map.shape[1]))
ax = fig.add_subplot(111)

plt.imshow(map.epsilon_map,
        extent=[0, map.epsilon_map.shape[1] * spatial_grid, map.epsilon_map.shape[0] * spatial_grid, 0],
        cmap='binary')

plt.xlabel('x (m)', size=14)
plt.ylabel('z (m)', size=14)
ax.set_title('epsilon_r distribution', size=18)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='epsilon_r')

plt.savefig(output_path+'/' + 'epsilon_map.png')
plt.show()
