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
parser.add_argument('-closeup', action='store_true', help='resolution of migration grid', default=False)
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
    # read epsilon_r
    h5_data = h5file['data'][:, :, 0]
    h5_data = np.rot90(h5_data)
    print('h5_data shape:')
    print(h5_data.shape)


    migration_grid_size = 1
    resolution = 0.1
    resolution_ratio = int(migration_grid_size / resolution)

    geometry_size_z = 300
    geometry_size_x = 550
    permittivity_map = 6 * np.ones([np.int(geometry_size_z/migration_grid_size), np.int(geometry_size_x/migration_grid_size)])
    print('permittivity_map shape:')
    print(permittivity_map.shape)

    z_num = permittivity_map.shape[0]
    x_num = permittivity_map.shape[1]

    # convert epsilon_r map
    for i in tqdm(range(z_num)):
        for j in range(x_num):
            if i*resolution_ratio >= h5_data.shape[0] or j*resolution_ratio >= h5_data.shape[1]:
                break
            elif h5_data[i*resolution_ratio, j*resolution_ratio] == 1:
                permittivity_map[i, j] = 1
            elif h5_data[i*resolution_ratio, j*resolution_ratio] == 2:
                permittivity_map[i, j] = 6
            elif h5_data[i*resolution_ratio, j*resolution_ratio] == 3:
                permittivity_map[i, j] = 6
            else:
                print('error')
    
    for i in tqdm(range(h5_data.shape[1])):
        for j in range(h5_data.shape[0]):
            if h5_data[j, i] == 1:
                h5_data[j, i] = 1
            elif h5_data[j, i] == 2:
                h5_data[j, i] = 4
            elif h5_data[j, i] == 3:
                h5_data[j, i] = 6
            else:
                print('error')
    
    # extract boudary
    
    return h5_data, permittivity_map, migration_grid_size, resolution

geometry_data, epsilon_map, migration_step, resolution = get_epsilon_map()
print('epsilon_map shape:')
print(epsilon_map.shape)

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