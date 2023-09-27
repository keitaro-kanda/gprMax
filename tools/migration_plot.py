import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from PIL import Image
from tqdm import tqdm

from tools.outputfiles_merge import get_output_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot migration',
                                    usage='cd gprMax; python -m tools.migration_plot migration_result file_type')
parser.add_argument('migration_result', help='migration resutl txt file name')
parser.add_argument('file_type', choices=['raw', 'pulse_comp'], help='file type')
args = parser.parse_args()

# load files
file_name = args.migration_result
migration_result = np.loadtxt(file_name)
print(migration_result.shape)
print(np.amax(migration_result))
rx = str(file_name.split('_')[-1].split('.')[0])
output_dir_path = os.path.dirname(file_name)

xgrid_num = migration_result.shape[1]
zgrid_num = migration_result.shape[0]



# set parameters
x_resolution = 1
z_resolution = 1



# =====plot=====
"""
if args.file_type == 'raw':
        migration_result_standardize = migration_result / np.amax(migration_result) * 100

elif args.file_type == 'pulse_comp':
        migration_result_standardize = np.zeros_like(migration_result)
        for i in tqdm(range(xgrid_num), desc = 'calculate power'):
                for j in range(zgrid_num):
                        if migration_result[j, i] == 0:
                                migration_result_standardize[j, i] = 10 * np.log10(1e-10 / np.amax(migration_result))
                        else:
                                migration_result_standardize[j, i] = 10 * \
                                        np.log10(np.abs(migration_result[j, i]) / np.abs(np.amax(migration_result)))
"""


# =====plot=====
fig = plt.figure(figsize=(10, 7), facecolor='w', edgecolor='w')
ax = fig.add_subplot(211)

if args.file_type == 'raw':
        plt.imshow(migration_result,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                aspect=z_resolution/x_resolution, cmap='seismic', vmin=-10, vmax=10, alpha=0.5)
elif args.file_type == 'pulse_comp':
        plt.imshow(migration_result,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                cmap='rainbow', vmin=-50, vmax=0)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label = 'power [dB]')

ax.set_xlabel('Horizontal distance [m]', size=14)
ax.set_ylabel('Depth form surface [m]', size=14)

if args.file_type == 'raw':
        edge_color = 'gray'
elif args.file_type == 'pulse_comp':
        edge_color = 'white'

ax.set_title('Migration result rx' + str(rx), size=18)


# 地形のプロット
rille_apex_list = [(0, 10), (25, 10), 
                (175, 260), (375, 260),
                (525, 10), (550, 10)]
rille = patches.Polygon(rille_apex_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(rille)

surface_hole_tube_list = [(40, 35), (250, 35),
                        (250, 60), (200, 60),
                        (200, 77), (350, 77),
                        (350, 60), (300, 60),
                        (300, 35), (515, 35)]
tube = patches.Polygon(surface_hole_tube_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(tube)


# =====seve plot=====
plt.savefig(output_dir_path + '/migration_result' + str(rx) + '.png', bbox_inches='tight', dpi=300)
plt.show()