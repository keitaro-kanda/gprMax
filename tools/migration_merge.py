import os

import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.dirname('kanda/domain_10x10/test_3cyliner/B-scan/array/B_out/migration/migration_result_rx1.txt')
file_name = 'migration_result_rx'


for i in range(1, 32, 1):
    mig_data = np.loadtxt(file_path + '/' + file_name + str(i) + '.txt')
    
    if i ==1:
        axis0 = mig_data.shape[0]
        axis1 = mig_data.shape[1]
        migration_merge = np.zeros([axis0, axis1])
    
    migration_merge = migration_merge + mig_data * 1/31

migration_result_percent = migration_merge / np.amax(migration_merge) * 100
plt.figure(figsize=(18, 15), facecolor='w', edgecolor='w')
plt.imshow(migration_result_percent,
        aspect='auto', cmap='seismic', vmin=-25, vmax=25)
plt.colorbar()
plt.xlabel('Horizontal distance [m]', size=20)
plt.ylabel('Depth form surface [m]', size=20)
plt.xticks(np.arange(0, axis1, 5), np.arange(0, 10, 1))
plt.yticks(np.arange(0, axis0, 100), np.arange(0, 8, 1))
plt.title('Migration Merged', size=20)


plt.grid()

plt.savefig(file_path + '/migration_merged.png')
plt.show()