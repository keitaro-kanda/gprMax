import os

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np

file_path = os.path.dirname('kanda/domain_550x270/rille_hole_tube/B_out/array/out_files/migration/migration_result_rx1.txt')
file_name = 'migration_result_rx'

spatial_step = 1
x_resolution = 1

for i in range(1, 51, 1):
    mig_data = np.loadtxt(file_path + '/' + file_name + str(i) + '.txt')
    
    if i ==1:
        axis0 = mig_data.shape[0]
        axis1 = mig_data.shape[1]
        migration_merge = np.zeros([axis0, axis1])
    
    migration_merge = migration_merge + mig_data

migration_result_percent = migration_merge / np.amax(migration_merge) * 100
#plt.figure(figsize=(18, 15), facecolor='w', edgecolor='w')
#plt.imshow(migration_result_percent,
#        aspect=spatial_step/x_resolution, cmap='seismic', vmin=-25, vmax=25)
#plt.colorbar()
#plt.xlabel('Horizontal distance [m]', size=20)
#plt.ylabel('Depth form surface [m]', size=20)
#plt.xticks(np.arange(0, axis1, 5), np.arange(0, 10, 1))
#plt.yticks(np.arange(0, axis0, 100), np.arange(0, 8, 1))
#plt.title('Migration merged', size=20)

fig = plt.figure(figsize=(15, 12), facecolor='w', edgecolor='w')
ax = fig.add_subplot(111)
plt.imshow(migration_result_percent,
        aspect=spatial_step/x_resolution, cmap='seismic', vmin=-10, vmax=10)
plt.grid()

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax)

ax.set_xlabel('Horizontal distance [m]', size=20)
ax.set_ylabel('Depth form surface [m]', size=20)
#ax.set_xticks(np.linspace(0, xgrid_num, 10), np.linspace(0, xgrid_num*x_resolution, 10))
#ax.set_yticks(np.linspace(0, zgrid_num, 10), np.linspace(0, zgrid_num*spatial_step, 11))
ax.set_title('Migration merged', size=20)


plt.savefig(file_path + '/migration_merged.png')
plt.show()