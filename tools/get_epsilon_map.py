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

"""
# read .h5 file
h5_file_name = args.file_name
h5file = h5py.File(h5_file_name, 'r') 

# check h5 file dataset
def check_file_structure():
    def PrintAllObjects(name):
        print(name)

    h5file.visit(PrintAllObjects)
#check_file_structure()

def get_ID():
    # read ID
    ID = h5file['ID'][:, 0, 0, 0]
    print(ID.shape)
    print(ID)


#get_ID()
"""

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

        for i in tqdm(range(self.h5_data.shape[1])):
            for j in range(self.h5_data.shape[0]):
                if self.h5_data[j, i] == 1:
                    self.epsilon_map[j, i] = self.epsilon_vacuum
                elif self.h5_data[j, i] == 2:
                    self.epsilon_map[j, i] = self.epsilon_regolith
                elif self.h5_data[j, i] == 3:
                    self.epsilon_map[j, i] = self.epsilon_basalt6
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
np.savetxt(output_path+'/' + 'h5_data.txt', map.h5_data, fmt='%.3f')
np.savetxt(output_path+'/' + 'epsilon_map.txt', map.epsilon_map, fmt='%.3f')


# =====plot map=====
spatial_grid =  0.05 # spatial grid size [m]


fig = plt.figure(figsize=(10, 6))
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

"""
def plot(map, x_resolution, z_resolution, file_name):
    #save epsilon_r map as txt file
    input_path = os.path.dirname(h5_file_name)
    output_path = os.path.join(input_path, 'map_fig')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    np.savetxt(output_path+'/' + file_name + '.txt', map, fmt='%.3f')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    plt.imshow(map,
            extent=[0, map.shape[1] * x_resolution, map.shape[0] * z_resolution, 0],
            cmap='binary')
    
    plt.xlabel('x (m)', size=14)
    plt.ylabel('z (m)', size=14)
    plt.title('epsilon_r distribution', size=18)

    if args.closeup == True:
        plt.xlim(150, 200)
        plt.ylim(265, 220)
    
    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax, label='epsilon_r')

    if args.closeup == True:
        plt.savefig(output_path+'/' + file_name + ' _closeup.png')
    else:
        plt.savefig(output_path+'/' + file_name + '.png')

    plt.show()

plot(epsilon_map, migration_step, migration_step, 'epsilon_map4mig')
plot(geometry_data, resolution, resolution, 'epsilon_map_from_h5')
"""

"""
def plot_background_img(map, x_resolution, z_resolution, file_name):
    input_path = os.path.dirname(h5_file_name)
    output_path = os.path.join(input_path, 'map_fig')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.imshow(map,
            extent=[0, map.shape[1] * x_resolution, map.shape[0] * z_resolution, 0],
            cmap='binary')
    
    ax.axis('off')

    plt.savefig(output_path+'/' + file_name + '.png',
                bbox_inches='tight', pad_inches=0)
    plt.show()

plot_background_img(geometry_data, args.resolution, args.resolution, 'epsilon_map4backimg')
"""