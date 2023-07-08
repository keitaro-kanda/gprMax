from calendar import c

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

file_name = 'kanda/domain_10x10/rock/B-scan/0.2step_n40/10x10_rock_merged.out'

# .outファイルの読み込み
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()

for rx in range(1, nrx + 1):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')


# outputdataの行を移動平均
outputdata_ave = np.zeros(outputdata.shape)
for i in range(outputdata.shape[0]): # 列
    for j in range(outputdata.shape[1] - 2): # 行
        outputdata_ave[i, j] = (outputdata[i, j-2] + outputdata[i, j-1] + outputdata[i, j] + \
                                outputdata[i, j+1] + outputdata[i, j+2]) / 5

# 観測の間隔
src_step = 0.2 #[m]

fig = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='w')

plt.imshow(outputdata_ave,
             extent=[0, outputdata_ave.shape[1] * src_step, outputdata_ave.shape[0] * dt, 0], 
            interpolation='nearest', aspect='auto', cmap='seismic', vmin=-1, vmax=1)


plt.xlabel('Trace number')
plt.ylabel('Time [s]')

# クローズアップのON/OFF
closeup = True # True or False
if closeup:
    plt.ylim(1.5e-7, 0)
    plt.minorticks_on( )

if closeup:
    plt.title('{}'.format('closeup'))

# Grid properties
ax = fig.gca()
ax.grid(which='both', axis='both', linestyle='-.')

cb = plt.colorbar()
cb.set_label('Field strength percentage [%]')

plt.show( )

print(outputdata.shape)
print(outputdata_ave.shape)