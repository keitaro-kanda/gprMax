"""
This tool is used to estimate the Vrms from the CMP observation data.
Input mergedout file must be CMP observation data.
"""
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


#* Parse command line arguments
parser = argparse.ArgumentParser(usage='cd gprMax; python -m tools.estimate_Vrms_CMP jsonfile')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'calc'])
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* load data
data_path = params['out_file']
data, dt = get_output_data(data_path, 1, 'Ez')


time_window = params['time_window'] # [ns]
time_step = params['time_step'] # [ns]
t0_array_ns = np.arange(0 , time_window, time_step) # [ns], array for extent
t0_array = t0_array_ns * 1e-9 # [s]
c = 3e8 # [m/s]
Vrms_array_percent = np.arange(0.01, 1.01, 0.01) # [/c], array for extent
Vrms_array = c * Vrms_array_percent # [m/s]


src_start = params['antenna_settings']['src_start'] # [m]
src_step = params['antenna_settings']['src_step'] # [m]
rx_start = params['antenna_settings']['rx_start'] # [m]
rx_step = params['antenna_settings']['rx_step'] # [m]

def get_amplitude(Vrms_ind, t0_ind, trace_num):
    #* get value of Vrms and t0
    t0 = t0_array[t0_ind] # [s]
    Vrms = Vrms_array[Vrms_ind]# [m/s]

    #* calculate offset
    tx_posi = src_start + trace_num * src_step # [m]
    rx_posi = rx_start + trace_num * rx_step # [m]
    offset = np.abs(tx_posi - rx_posi) # [m]

    #* calculate delaytime
    total_delay_time = np.sqrt(t0**2 + (offset / Vrms)**2)
    if total_delay_time > time_window * 1e-9:
        Amplitude = 0
    else:
        Amplitude = data[int(total_delay_time / dt), trace_num]
    return Amplitude


def calc_correration():
    correration_map = np.zeros((len(t0_array), len(Vrms_array)))

    for v in tqdm(range(len(Vrms_array))):
        for t in range(len(t0_array)):
            Amp_vt = []
            for trace_num in range(data.shape[1]):
                Amp_vt.append(get_amplitude(v, t, trace_num))
            correration_map[t, v] = np.sum(np.abs([a * b for a, b in combinations(Amp_vt, 2)]))
    return correration_map


output_dir = os.path.dirname(data_path)
output_dir_name = 'Vrms_estimation'
output_dir_path = os.path.join(output_dir, output_dir_name)
#* In case calculate and plot
if args.plot_type == 'calc':
    corr_map = calc_correration()
    #* save
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    np.savetxt(os.path.join(output_dir_path, 'corr_map.txt'), corr_map)
#* In case plot only
elif args.plot_type == 'plot':
    corr_map_file_path = params['corr_map_txt']
    corr_map = np.loadtxt(corr_map_file_path)
#* In case invalid plot type
else:
    raise ValueError('Invalid plot type')

#* normalize
corr_map = corr_map / np.max(corr_map)


#* plot
fig = plt.figure(figsize=(12, 10), tight_layout=True)
ax = fig.add_subplot(111)

fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

plt.imshow(corr_map,
            cmap = 'jet', aspect='auto',interpolation='nearest',
            extent=[Vrms_array_percent[0], Vrms_array_percent[-1], t0_array_ns[-1], t0_array_ns[0]],
            norm = colors.LogNorm(vmin=1e-7, vmax=1)
            )


ax.set_xlabel('Vrms [/c]', fontsize=fontsize_medium)
ax.set_ylabel('t0 [ns]', fontsize=fontsize_medium)
ax.tick_params(labelsize=fontsize_small)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
cax.tick_params(labelsize=16)

plt.savefig(os.path.join(output_dir_path, 'corr_map.png'))
plt.show()