import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import json
import argparse
import mpl_toolkits.axes_grid1 as axgrid1
from tools.k_plot_geometry import epsilon_map


#* Parse command line arguments
parser = argparse.ArgumentParser(description='Plot geometry map and imaging result',
                                usage='cd gprMax; python -m tools.plot_geometry_imaging jsonfile -theory')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('-theory', action='store_true', help='option: use theoretical value for t0 and Vrms')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* load geometry data
map = epsilon_map()
map.h5_file_name = params['geometry_settings']['h5_file']
map.read_h5_file()
map.get_epsilon_map()
geometry = map.epsilon_map


#* cut vacuum area
antenna_hight = params['antenna_settings']['antenna_height']


#* load imaging result data
path_imaging = params['imaging_result_csv']
if args.theory:
    path_imaging = path_imaging.replace('.csv', '_theory.csv')
imaging_result = np.loadtxt(path_imaging, delimiter=',')
#* add vacuum area to imaging result
ground_depth = params['geometry_settings']['ground_depth']
grid_size = params['geometry_settings']['grid_size']
vacuum_thickness = int(geometry.shape[0] - (ground_depth + antenna_hight) / grid_size) # number of grids, not in [m]
imaging_resolution = params['imaging_resolution']
imaging_result = np.pad(imaging_result, ((int(vacuum_thickness*grid_size/imaging_resolution), 0), (0, 0)), 'constant', constant_values=0)


#* plot
fig,ax = plt.subplots(1, 2, tight_layout=True, figsize=(10, 5*geometry.shape[0]/geometry.shape[1]))
fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

#* plot geometry
img1 = ax[0].imshow(geometry,
                extent=[0, geometry.shape[1]*grid_size, geometry.shape[0]*grid_size, 0],
                cmap='binary', # recommended: 'jet', 'binary'
                aspect=1)

ax[0].set_yticks(np.arange(0, geometry.shape[0]*grid_size, 10))
#ax[0].set_yticks(np.arange(-vacuum_thickness*grid_size, geometry.shape[0] * grid_size - vacuum_thickness*grid_size + 1, 5))
ax[0].set_title('geometry', size = fontsize_large)

delvider = axgrid1.make_axes_locatable(ax[0])
cax1 = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(img1, cax=cax1).set_label('epsilon_r', size=fontsize_medium)


#* plot imaging result
img2 = ax[1].imshow(imaging_result,
                extent = [0, imaging_result.shape[1]*params['imaging_resolution'],
                        imaging_result.shape[0]*params['imaging_resolution'], 0],
                cmap='jet', # recommended: 'jet', 'gray'
                aspect=1, norm=colors.LogNorm(vmin=1e-5, vmax=np.amax(imaging_result)))

ax[1].set_yticks(np.arange(0, imaging_result.shape[0]*params['imaging_resolution'], 10))
#ax[1].set_yticks(np.arange(-vacuum_thickness*grid_size, imaging_result.shape[0] * imaging_resolution - vacuum_thickness*grid_size + 1, 5))
ax[1].set_title('imaging result', size = fontsize_large)

delvider = axgrid1.make_axes_locatable(ax[1])
cax2 = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(img2, cax=cax2).set_label('amplitude', size=fontsize_medium)

fig.supxlabel('x (m)', size = fontsize_medium)
fig.supylabel('z (m)', size = fontsize_medium)

output_dir_path = os.path.dirname(path_imaging)
if args.theory:
    plt.savefig(output_dir_path + '/imaging_result_geometry_theory.png')
else:
    plt.savefig(output_dir_path + '/imaging_result_geometry.png')

plt.show()
