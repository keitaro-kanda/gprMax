from calendar import c

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

# 読み込むファイルの指定
filename_original = 'kanda/inner_tube/ver8/B-scan/v8_A_35_x4_02_original/inner_v8_merged.out'
filename_onlysurface = 'kanda/inner_tube/ver8/B-scan/v8_A_35_x4_02_singlelayer/inner_v8_merged.out'
filename_array = [filename_original, filename_onlysurface]


# .outファイルの読み込み
for filename in filename_array:
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    f.close()

    for rx in range(1, nrx + 1):
        if filename == filename_array[0]:
            outputdata_1, dt = get_output_data(filename, rx, 'Ez')
        elif filename == filename_array[1]:
            outputdata_2, dt = get_output_data(filename, rx, 'Ez')


# 地下エコー要素の抽出
outputdata_extract = outputdata_1 - outputdata_2

outputdata_norm = outputdata_extract / np.amax(np.abs(outputdata_1)) * 100

fig = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='w')


# 観測の方向
radar_direction = 'horizontal' # horizontal or vertical

# プロット
if radar_direction == 'horizontal':
    plt.imshow(outputdata_norm, 
             extent=[0, outputdata_norm.shape[1], outputdata_norm.shape[0] * dt, 0], 
            interpolation='nearest', aspect='auto', cmap='seismic', vmin=-2, vmax=2)
    plt.xlabel('Trace number')
    plt.ylabel('Time [s]')
    closeup = True # True or False
    if closeup:
        plt.ylim(1.0e-7, 0)
        plt.minorticks_on( )
else:
# Create a plot rotated 90 degrees and then reversed up and down.
    plt.imshow(outputdata_norm.T[::-1],
            extent=[0, outputdata_norm.shape[0] * dt, 0, outputdata_norm.shape[1]], 
            interpolation='nearest', aspect='auto', cmap='seismic', vmin=-10, vmax=10)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Trace number')

plt.title('{}'.format(filename))

# Grid properties
ax = fig.gca()
ax.grid(which='both', axis='both', linestyle='-.')

cb = plt.colorbar()
cb.set_label('Field strength percentage [%]')

plt.show( )

output_dir = 'kanda/inner_tube/ver8/B-scan/extraction/'
fig.savefig(output_dir + 'extract_original_singlelayer.png', dpi=150, format='png',
            bbox_inches='tight', pad_inches=0.1)