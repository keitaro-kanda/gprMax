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
from tools.calc_Vrms_from_geometry import calc_Vrms


#* Parse command line arguments
parser = argparse.ArgumentParser(usage='cd gprMax; python -m tools.estimate_Vrms_CMP jsonfile')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'calc', 'select']) # 'plot': plot only, 'calc': calculate and plot, 'select': plot only with selected data area
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)
#* load geometry json file
with open(params['geometry_settings']['geometry_json']) as f:
    geometry_params = json.load(f)


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



#* prepare arrays
time_window = params['time_window'] # [ns]
time_step = params['time_step'] # [ns]
t0_array_ns = np.arange(0 , time_window, time_step) # [ns], array for extent
t0_array = t0_array_ns * 1e-9 # [s]
c = 3e8 # [m/s]
Vrms_array_percent = np.arange(0.01, 1.01, 0.01) # [/c], array for extent
Vrms_array = c * Vrms_array_percent # [m/s]


#* need for calculation of  offset in get_apmlitude function
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


#* prepare output directory
output_dir = os.path.dirname(args.jsonfile)
if args.plot_type == 'select':
    output_dir_name = 'Vrms_estimation/selected'
else:
    output_dir_name = 'Vrms_estimation'
output_dir_path = os.path.join(output_dir, output_dir_name)
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)


#* run the tool
#! =====select area-----
select_t0_start = 1675 # [ns]
select_t0_end = 1725 # [ns]
select_Vrms_start = 0 # [/c]
select_Vrms_end = 1 # [/c]
#! ---------------------
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
#* In case plot only with selected data area
elif args.plot_type == 'select':
    corr_map = np.loadtxt(params['corr_map_txt'])
    corr_map = corr_map[int(select_t0_start/time_step):int(select_t0_end/time_step)
                        , int(select_Vrms_start/0.01):int(select_Vrms_end/0.01)] # select data area
    corr_map = corr_map / np.max(corr_map) # normalize
#* In case invalid plot type
else:
    raise ValueError('Invalid plot type')
#* normalize
corr_map = corr_map / np.max(corr_map)


#* get theoretical value of Vrms and t0 from geometry
calc_Vrms_geometry = calc_Vrms(params['geometry_settings']['geometry_json'])
layer_thickness, internal_permittivity, internal_velovity = calc_Vrms_geometry.load_params_from_json()
t0_theory = calc_Vrms_geometry.calc_t0(layer_thickness, internal_velovity) # [s]
Vrms_theory = calc_Vrms_geometry.calc_Vrms(layer_thickness, internal_velovity, t0_theory) # [/c]


#* plot
fig = plt.figure(figsize=(12, 10), tight_layout=True)
ax = fig.add_subplot(111)

fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

#* show theoretical values
ax.scatter(Vrms_theory, t0_theory * 1e9,
        c = 'black', s = 70, marker = 'P', edgecolor = 'white',
        label = 'Theoretical values')


#* show correlation map
if args.plot_type == 'select':
    bounds = np.array([0, 0.25, 0.50, 0.75, 1.0])
    cmap = mpl.colors.ListedColormap([plt.cm.Blues(int(255*i/3)) for i in range(4)])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(corr_map,
            cmap = cmap, aspect='auto', interpolation='nearest',
            extent=[select_Vrms_start, select_Vrms_end, select_t0_end, select_t0_start],
            norm = norm
            )
    ax.minorticks_on()
    ax.grid(which='both', linestyle='--', linewidth=0.5)
else:
    plt.imshow(corr_map,
                cmap = 'jet', aspect='auto',interpolation='nearest',
                extent=[Vrms_array_percent[0], Vrms_array_percent[-1], t0_array_ns[-1], t0_array_ns[0]],
                norm = colors.LogNorm(vmin=1e-5, vmax=1)
                )


ax.legend(fontsize=fontsize_small, loc='lower right')
ax.set_xlabel('Vrms [/c]', fontsize=fontsize_medium)
ax.set_ylabel('t0 [ns]', fontsize=fontsize_medium)
ax.tick_params(labelsize=fontsize_small)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Amplitude', fontsize=18)
cax.tick_params(labelsize=16)

if args.plot_type == 'select':
    plt.savefig(output_dir_path + '/corr_map_selected_' + str(select_t0_start) + '_' + str(select_t0_end) + '.png')
    #plt.savefig(os.path.join(output_dir_path, 'corr_map_selected_',str(select_t0_start), str(select_t0_end), '.png' ))
else:
    plt.savefig(os.path.join(output_dir_path, 'corr_map.png'))
plt.show()