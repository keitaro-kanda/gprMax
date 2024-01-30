import argparse
import os

import h5py
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from tqdm import tqdm
import json

#* Parse command line arguments
parser = argparse.ArgumentParser(description='get epsilon_r map from .h5 file',
                                 usage='cd gprMax; python -m tools.plot_geometry json_file -closeup')
parser.add_argument('json_file', help='json file name')
parser.add_argument('-closeup', action='store_true', help='closeup of the plot', default=False)
args = parser.parse_args()

#* load json file
with open (args.json_file) as f:
    params = json.load(f)


#* load epsilon from materials.txt file
material_path = params['material_file']
with open(material_path, 'r') as f:
    epsilon_list = []

    for line in f:
        # 行を空白やタブなどで分割してリストにする
        values = line.split()
        
        # 2列目の数字を取得してリストに追加
        if len(values) >= 2:  # 行に少なくとも2つの要素があることを確認
            epsilon_list.append(float(values[1]))  # 2番目の要素を取得し、floatに変換してリストに追加



class epsilon_map():
    def __init__(self):
        self.h5_file_name = ''
    
    #* read .h5 file
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


        for i in tqdm(range(self.h5_data.shape[1])):
            for j in range(self.h5_data.shape[0]):
                self.epsilon_map[j, i] = epsilon_list[int(self.h5_data[j, i])]


map = epsilon_map()
map.h5_file_name = params['h5_file']
map.read_h5_file()
map.get_epsilon_map()


# =====output dir path=====
input_path = os.path.dirname(map.h5_file_name)
output_path = os.path.join(input_path, 'map')
if not os.path.exists(output_path):
    os.mkdir(output_path)


# =====save map as txt file=====
#! file size is too large, so don't save
#np.savetxt(output_path+'/' + 'h5_data.csv', map.h5_data, delimiter=',')
#np.savetxt(output_path+'/' + 'epsilon_map.txt', map.epsilon_map, delimiter=',')


# =====plot map=====
spatial_grid =  params['grid_size'] # spatial grid size [m]


fig = plt.figure(figsize=(5, 5*map.epsilon_map.shape[0]/map.epsilon_map.shape[1]))
ax = fig.add_subplot(111)

vacuum_thickness = params['domain_z'] - params['ground_depth']
plt.imshow(map.epsilon_map,
        extent=[0, map.epsilon_map.shape[1] * spatial_grid,
                map.epsilon_map.shape[0] * spatial_grid - vacuum_thickness, -vacuum_thickness],
        cmap='binary')

if args.closeup:
    y_start = 9
    y_end = 40
    ax.set_ylim(y_end, y_start)

ax.set_yticks(np.arange(-vacuum_thickness, map.epsilon_map.shape[0] * spatial_grid - vacuum_thickness + 1, 5))

plt.xlabel('x (m)', size=14)
plt.ylabel('z (m)', size=14)
ax.set_title('epsilon_r distribution', size=18)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='epsilon_r')

plt.savefig(output_path+'/' + 'epsilon_map.png')
if args.closeup:
    plt.savefig(output_path +'/closeup' + str(y_start) + '-' + str(y_end) + '.png')
plt.show()
