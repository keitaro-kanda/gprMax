import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import json
import argparse
import mpl_toolkits.axes_grid1 as axgrid1


#* Parse command line arguments
parser = argparse.ArgumentParser(description='Plot geometry map and imaging result',
                                 usage='cd gprMax; python -m tools.plot_geometry_imaging jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* load geometry data
path_geometry = params['geometry_txt']
geometry = np.loadtxt(path_geometry, delimiter=',')


# cut vacuum area
antenna_hight = params['antenna_height']
ground_depth = params['ground_depth']
vacuum_thicness = int(geometry.shape[0] - (ground_depth + antenna_hight) / params['grid_size'])
geometry = geometry[vacuum_thicness: , :]



#* load imaging result data
path_imaging = params['imaging_result_csv']
imaging_result = np.loadtxt(path_imaging, delimiter=',')



#* plot
fig,ax = plt.subplots(1, 2,
                        tight_layout=True, figsize=(10, 5*geometry.shape[0]/geometry.shape[1]))

# plot geometry
img1 = ax[0].imshow(geometry,
                extent=[0, geometry.shape[1]*params['grid_size'], geometry.shape[0]*params['grid_size'], 0],
                cmap='binary', aspect=1)

ax[0].set_title('geometry', size=18)

delvider = axgrid1.make_axes_locatable(ax[0])
cax1 = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(img1, cax=cax1, label='epsilon_r')


# plot imaging result
img2 = ax[1].imshow(imaging_result,
                extent = [0, imaging_result.shape[1]*params['imaging_resolution'],
                        imaging_result.shape[0]*params['imaging_resolution'], 0],
                cmap='gray', # recommended: 'jet', 'gray'
                aspect=1, norm=colors.LogNorm(vmin=1e-3, vmax=np.amax(imaging_result)))

ax[1].set_title('imaging result', size=18)

delvider = axgrid1.make_axes_locatable(ax[1])
cax2 = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(img2, cax=cax2, label='correlation')

fig.supxlabel('x (m)', size=14)
fig.supylabel('z (m)', size=14)

output_dir_path = os.path.dirname(path_imaging)
plt.savefig(output_dir_path + '/imaging_result_geometry.png')

plt.show()
