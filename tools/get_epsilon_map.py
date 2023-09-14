import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='get epsilon_r map from .h5 file', 
                                 usage='cd gprMax; python -m tools.get_epsilon_map file_name')
parser.add_argument('file_name', help='.h5 file name')
args = parser.parse_args()

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

def get_epsilon_map():
    permittivity_map = np.zeros([300, 550])
    print(permittivity_map.shape)
    # read epsilon_r
    h5_data = h5file['data'][:, :, 0]
    h5_data = np.rot90(h5_data)
    print(h5_data.shape)

    z_num = permittivity_map.shape[0]
    x_num = permittivity_map.shape[1]

    # convert epsilon_r map
    for i in tqdm(range(z_num)):
        for j in range(x_num):
            if i*10 >= h5_data.shape[0] or j*10 >= h5_data.shape[1]:
                break
            elif h5_data[i*10, j*10] == 1:
                permittivity_map[i, j] = 1
            elif h5_data[i*10, j*10] == 2:
                permittivity_map[i, j] = 4
            elif h5_data[i*10, j*10] == 3:
                permittivity_map[i, j] = 6
            else:
                print('error')
    
    return permittivity_map, z_num, x_num

epsilon_map, axis0_index_num, axis1_index_num = get_epsilon_map()

def plot(map, z_num, x_num):
    #save epsilon_r map as txt file
    output_path = os.path.dirname(h5_file_name)
    np.savetxt(output_path+'/epsilon_map.txt', map, fmt='%.3f')

    plt.imshow(map,
            #extent=[0, x_num * 0.1, z_num * 0.1, 0],
            cmap='binary')
    plt.colorbar()
    plt.show()

plot(epsilon_map, axis0_index_num, axis1_index_num)