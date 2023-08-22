import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot migration',
                                    usage='cd gprMax; python -m tools.migration jsonfile')
parser.add_argument('file_name', help='migration resutl txt file name')
args = parser.parse_args()

# load files
file_name = args.file_name
migration_result = np.loadtxt(file_name)
rx = str(file_name.split('_')[-1].split('.')[0])

output_dir_path = os.path.dirname(file_name)



migration_result_percent = migration_result / np.amax(migration_result)* 100


# 定数の設定
spatial_step = 1 # [m]
x_resolution = 1 # [m]

fig = plt.figure(figsize=(13, 14), facecolor='w', edgecolor='w')
#plt.tight_layout()

ax = fig.add_subplot(211)
plt.imshow(migration_result_percent,
        aspect=spatial_step/x_resolution, cmap='seismic', vmin=-10, vmax=10)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax)

ax.set_title('Migration result ' + rx, size=20)
ax.set_xlabel('Horizontal distance [m]', size=20)
ax.set_ylabel('Depth form surface [m]', size=20)
#ax.set_xticks(np.linspace(0, xgrid_num, 10), np.linspace(0, xgrid_num*x_resolution, 10))
#ax.set_yticks(np.linspace(0, zgrid_num, 10), np.linspace(0, zgrid_num*spatial_step, 11))


ax = fig.add_subplot(212)
plt.imshow(migration_result_percent,
        aspect=spatial_step/x_resolution, cmap='seismic', vmin=-10, vmax=10)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax)

ax.set_xlabel('Horizontal distance [m]', size=20)
ax.set_ylabel('Depth form surface [m]', size=20)
#ax.set_xticks(np.linspace(0, xgrid_num, 10), np.linspace(0, xgrid_num*x_resolution, 10))
#ax.set_yticks(np.linspace(0, zgrid_num, 10), np.linspace(0, zgrid_num*spatial_step, 11))
ax.set_title('Migration result ' + rx, size=20)
# 地形のプロット
rille_apex_list = [(0, 10), (25, 10), (125, 260), (425, 260), (525, 10), (550, 10)]
rille = patches.Polygon(rille_apex_list, ec='gray', linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(rille)

surface_hole_tube_list = [(35, 35), (250, 35), (250, 60), (175, 60), (175, 77),
                        (375, 77), (375, 60), (300, 60), (300, 35), (515, 35)]
tube = patches.Polygon(surface_hole_tube_list, ec='gray', linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(tube)

#fig.suptitle('Migration result ' + str(rx), size=20)
#fig.supxlabel('Horizontal distance [m]', size=20)
#fig.supylabel('Depth [m]', size=20)


# plotの保存
plt.savefig(output_dir_path+'/migration_result_' + str(rx) + '.png', bbox_inches='tight', dpi=300)

plt.show()