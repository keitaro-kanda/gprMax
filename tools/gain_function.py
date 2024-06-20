import numpy as np
import argparse
import json
import os

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm

from tools.outputfiles_merge import get_output_data
from itertools import combinations
import matplotlib as mpl
from tools.calc_Vrms_from_geometry import calc_Vrms
from mpl_toolkits.axes_grid1 import ImageGrid


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='gain_function.py',
    description='Process gain function',
    epilog='End of help message',
    usage='python -m tools.gain_function jsonfile'
)
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* load B-scan data
#* Chech wheter if json file has 'out_file' key or 'txt_Bscan_file' key
if 'out_file' in params:
    data_path = params['out_file']
    data, dt = get_output_data(data_path, 1, 'Ez')
    print('input is out_file')
elif 'txt_Bscan_file' in params:
    data_path = params['original_info']['original_out_file']
    data, dt = get_output_data(data_path, 1, 'Ez')
    data = np.loadtxt(params['txt_Bscan_file'])
    print('input is extracted B-scan data txt file')
else:
    raise ValueError('Invalid key: out_file or txt_Bscan_file')
print('data shape: ', data.shape)
print('dt: ', dt)
t = np.arange(0, data.shape[0] * dt, dt) # [s]
print(np.max(t))


#* gain function
def gain(Ascan):
    #* setting parameters
    c = 3e8 # [m/s]
    f = params['pulse_info']['center_frequency'] # [Hz]
    eps_r = 6
    loss_tangent = 0.001
    wabelength = c / f / np.sqrt(eps_r)
    alpha = np.pi / wabelength * np.sqrt(eps_r) * loss_tangent

    r = t * c / np.sqrt(eps_r) / 2
    Gain = r**2 * np.exp(2 * alpha * r)
    #plt.plot(t*1e9, Gain)
    #plt.yscale('log')
    #plt.grid()
    #plt.show()

    gained_Ascan = Ascan * Gain
    return gained_Ascan


#* use gain function to data
gained_data = np.zeros(data.shape)
for i in tqdm(range(data.shape[1])):
    gained_data[:, i] = gain(data[:, i])


#* Normalize
data = data / np.amax(np.abs(data))
gained_data = gained_data / np.amax(np.abs(gained_data))
print('original data shape: ', data.shape)
print('gained data shape: ', gained_data.shape)


#* plot
font_large = 20
font_medium = 18
font_small = 16
color_max = 0.01
color_min = -0.01
"""
fig = plt.figure(figsize=(15, 7))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2),
                axes_pad=0.15, cbar_mode='single', cbar_location='right', cbar_pad=0.15)

ax1 = grid[0]
im1 = ax1.imshow(data, aspect='auto', cmap='seismic',
            vmin=color_min, vmax=color_max,
            extent=[0, data.shape[1]*2, data.shape[0]*dt*1e9, 0])
ax1.set_title('original data', fontsize=font_large)
ax1.grid()
ax1.tick_params(labelsize=font_medium)

ax = grid[1]
im = ax.imshow(gained_data, aspect='auto', cmap='seismic',
            vmin=color_min, vmax=color_max,
            extent=[0, data.shape[1]*2, data.shape[0]*dt*1e9, 0])
ax.set_title('Gain compensation', fontsize=font_large)
ax.grid()
ax.tick_params(labelsize=font_medium)

cbar = grid.cbar_axes[0].colorbar(im1)
cbar.ax.axis('off')
cbar.set_label('Normalized amplitude', fontsize=font_medium)
cbar.ax.tick_params(labelsize=font_medium)

fig.supxlabel('Offset [m]', fontsize=font_large)
fig.supylabel('Time [ns]', fontsize=font_large)
#plt.tight_layout()
plt.show()
"""
antenna_step = params['antenna_settings']['rx_step'] * 2
fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

color_max = 0.01
color_min = -0.01

ax[0].imshow(data, aspect='auto', cmap='seismic',
            vmin=color_min, vmax=color_max,
            #extent=[0, data.shape[1]*antenna_step, data.shape[0]*dt*1e9, 0]
            )
ax[0].set_title('original data', fontsize=font_large)
ax[0].grid()
ax[0].tick_params(labelsize=font_medium)

ax[1].imshow(gained_data, aspect='auto', cmap='seismic',
            vmin=color_min, vmax=color_max,
            #extent=[0, data.shape[1]*antenna_step, data.shape[0]*dt*1e9, 0]
            )
ax[1].set_title('Gain compensation', fontsize=font_large)
ax[1].grid()
ax[1].tick_params(labelsize=font_medium)

"""
fig.colorbar(ax[0].imshow(data, aspect='auto', cmap='seismic',
            vmin=color_min, vmax=color_max,
            extent=[0, data.shape[1]*antenna_step, data.shape[0]*dt*1e9, 0]), ax=ax[1],
            location='right', orientation='vertical', label='Normalized amplitude')
"""


fig.supxlabel('Offset [m]', fontsize=font_large)
fig.supylabel('Time [ns]', fontsize=font_large)
plt.tight_layout()
plt.show()