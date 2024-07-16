import numpy as np
import argparse
import json
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from tools.outputfiles_merge import get_output_data
import matplotlib as mpl


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


src_start = params['antenna_settings']['src_start'] # [m]
src_step = params['antenna_settings']['src_step'] # [m]
rx_start = params['antenna_settings']['rx_start'] # [m]
rx_step = params['antenna_settings']['rx_step'] # [m]


#* setting parameters
c = 3e8 # [m/s]
f = params['pulse_info']['center_frequency'] # [Hz]
eps_r = 6
loss_tangent = 0.001
wabelength = c / f / np.sqrt(eps_r)
alpha = np.pi / wabelength * np.sqrt(eps_r) * loss_tangent


# Precompute depth and gain factors
depth = t * c / np.sqrt(eps_r) / 2
gain_cache = {}

def gain(Ascan, trace_num):
    if trace_num not in gain_cache:
        tx_posi = src_start + trace_num * src_step  # [m]
        rx_posi = rx_start + trace_num * rx_step  # [m]
        offset = np.abs(tx_posi - rx_posi)  # [m]
        r = np.sqrt(offset**2 + depth**2)
        gain_cache[trace_num] = r**2 * np.exp(2 * alpha * r)
    return Ascan * gain_cache[trace_num]



#* Plot gain function
def plot_gain_func():
    dummy = np.ones((data.shape[0], 1))
    plt.figure(figsize=(10, 10))
    gain_func = gain(dummy, 0)
    plt.plot(gain_func, label=f'offset: {0 * rx_step * 2}')
    plt.grid()
    plt.legend()
    plt.show()
plot_gain_func()


def process_gain_func():
    #* use gain function to data
    gained_data = np.zeros(data.shape)
    for i in tqdm(range(data.shape[1])):
        gained_data[:, i] = gain(data[:, i], i)
    output_dir = os.path.join(os.path.dirname(args.jsonfile), 'gain')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    np.savetxt(output_dir + '/gained_data.txt', gained_data, delimiter=' ')


    #* Normalize
    data = data / np.amax(np.abs(data)) * 100
    gained_data = gained_data / np.amax(np.abs(gained_data)) * 100
    print('original data shape: ', data.shape)
    print('gained data shape: ', gained_data.shape)
    print(np.amax(data))


    font_large = 20
    font_medium = 18
    font_small = 16
    color_max = 1 # [%]
    color_min = -1 # [%]

    grid_kw = dict(left=0.1, right=0.9, bottom=0.1, top=0.9)
    antenna_step = params['antenna_settings']['rx_step'] * 2
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, gridspec_kw=grid_kw, tight_layout=True)


    ax[0].imshow(data, aspect='auto', cmap='seismic',
                vmin=color_min, vmax=color_max,
                extent=[0, data.shape[1]*antenna_step, data.shape[0]*dt*1e9, 0],
                )
    ax[0].set_title('original data', fontsize=font_large)
    ax[0].grid(which='both', axis='both', linestyle='-.')
    ax[0].tick_params(labelsize=font_medium)

    ax[1].imshow(gained_data, aspect='auto', cmap='seismic',
                vmin=color_min, vmax=color_max,
                extent=[0, data.shape[1]*antenna_step, data.shape[0]*dt*1e9, 0]
                )
    ax[1].set_title('Gain compensation', fontsize=font_large)
    ax[1].grid(which='both', axis='both', linestyle='-.')
    ax[1].tick_params(labelsize=font_medium)


    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8]) # [left, bottom, width, height
    mappable = mpl.cm.ScalarMappable(cmap='seismic')
    fig.colorbar(mappable, cax=cax,  orientation='vertical',).set_label('Normalized amplitude', fontsize=font_large)
    cax.tick_params(labelsize=font_medium)

    """
    fig.colorbar(ax[0].imshow(data, aspect='auto', cmap='seismic',
                vmin=color_min, vmax=color_max,
                extent=[0, data.shape[1]*antenna_step, data.shape[0]*dt*1e9, 0]), ax=ax[1],
                location='right', orientation='vertical', label='Normalized amplitude')
    """


    fig.supxlabel('Offset [m]', fontsize=font_large)
    fig.supylabel('Time [ns]', fontsize=font_large)

    plt.savefig(output_dir + '/gain_function.png')
    plt.show()
#process_gain_func()