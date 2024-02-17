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
import matplotlib as mpl
from scipy.optimize import fsolve
from multiprocessing import Pool
from multiprocessing import freeze_support



#* Parse command line arguments
parser = argparse.ArgumentParser(description='Processing estimate Vrms in De Pue et al. method',
                                 usage='cd gprMax; python -m tools.estimate_Vrms_DePue jsonfile plot_type')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'mask', 'select', 'calc'])
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)



#* Open output file and read number of outputs (receivers)
data_path = params['out_file']
data = h5py.File(data_path, 'r')
nrx = data.attrs['nrx']
data.close()
data_dir_path = os.path.dirname(data_path)



#* load data
data_list = []
#if args.select_points == False:
for i in range(1, nrx+1):
    data, dt = get_output_data(data_path, i, 'Ez')
    data_list.append(data)



#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum
epsilon_0 = 1 # vacuum permittivity



#* set calculation parameters
RMS_velocity = np.arange(0.01, 1.01, 0.01) # percentage to speed of light, 0% to 100%
vertical_delay_time = np.arange(0, params['time_window'], params['time_step']) # 2-way travel time in vertical direction, don't contans air travel time[ns]



#* load parameters from json file
antenna_step = params['src_step'] # antenna distance step, [m]
rx_start = params['rx_start'] # rx start position, [m]
src_start = params['src_start'] # src start position, [m]
src_end = params['src_end'] # src end position, [m]
src_move_times = nrx # the number of src move times
pulse_width = int(params['pulse_length'] / dt) # [data point]
transmit_delay = params['transmitting_delay'] # [ns]
transmit_delay_point = int(transmit_delay / dt) # [data point]
antenna_height = params['antenna_height'] # [m]



#* difine equations to solve
def DePue_eq9(y1, y0, z0, z1, v1):
    sin0 = ((y0 - y1) / 2) / ((y0 - y1)**2 + z0**2) # De Pue eq(9b)
    sin1 = (y1 / 2) / (y1**2 + z1**2) # De Pue eq(9c)

    return sin0 / sin1 - c / 1



#* make function to calcurate semblance
def calc_semblance(Vrms_ind, t0_ind, i): # i: in range(nrx)
    rx_posi = rx_start + i * antenna_step

    Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
    t0 = vertical_delay_time[t0_ind] * 10**(-9) # [s]

    semblance_array = []
    depth = Vrms * t0 / 2 # [m]


    for src in range(src_move_times):
        src_posi = src_start + src * antenna_step # [m]
        offset = np.abs(src_posi - rx_posi) # [m]

        # calculate offset_ground
        offset_ground = np.linspace(0, offset, 100) # [m]
        offset_ground_solution = fsolve(DePue_eq9, 0.1, args=(offset, antenna_height, depth, Vrms), maxfev=2000)


        # calculate delay time
        delay_time = np.sqrt((offset - offset_ground_solution)**2 + 4 * antenna_height**2) / c \
            + t0  + offset_ground_solution**2 / Vrms**2 # [s]
        total_delay_time = delay_time + transmit_delay * 10**(-9) # [s]
        total_delay_point = int(total_delay_time / dt) # data point


        if total_delay_point >= len(data_list[i]):
            semblance_array.append(0)
            continue
        else:
            semblance_array.append(data_list[i][total_delay_point, src])

    return semblance_array



def calc_semblance_helper(args):
    v, t, i = args
    return calc_semblance(v, t, i)



#* roop calc_semblance
def roop():
    semblance_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))

    # Use multiprocessing to parallelize the outer loop
    with Pool() as pool:
        results = list(tqdm(pool.imap(calc_semblance_helper,
                        [(v, t, i) for v in range(len(RMS_velocity)) for t in range(len(vertical_delay_time)) for i in range(nrx)]),
                        total=len(RMS_velocity)*len(vertical_delay_time), desc='calc_semblance'))


    # Reshape the results back into semblance_map
    for v in range(len(RMS_velocity)):
        for t in range(len(vertical_delay_time)):
            index = v * len(vertical_delay_time) * nrx + t * nrx
            semblance_map[t, v] = np.array(results[index:index + nrx]).sum() ** 2

    #for v in tqdm(range(len(RMS_velocity)), desc='calc_semblance'):
    #    for t in tqdm(range(len(vertical_delay_time)), desc='v=' + str(v)):
    #        semblance_map[t, v] = (np.array([calc_semblance(v, t, i) for i in range(nrx)]).sum())**2

    return semblance_map



#* make output directory
output_dir_path = data_dir_path + '/Vrms_DePue'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)



#* load only or calculate and save
"""
select area [ns]
"""
select_start = 60
select_end = 100
select_startx = 0
select_endx = 1.0
"""
select area [ns]
"""

if __name__ == '__main__':
    freeze_support()

    #* plot only
    if args.plot_type == 'plot':
        Vt_map = np.loadtxt(params['semblance_txt'], delimiter=',')


    #* make select plot
    elif args.plot_type == 'select':
        Vt_map = np.loadtxt(params['semblance_txt'], delimiter=',') # load data
        Vt_map = Vt_map[int(select_start/params['time_step']): int(select_end/params['time_step']), int(select_startx/0.02): int(select_endx/0.02)] # select area
        Vt_map = Vt_map / np.amax(Vt_map) # normalize by max value in selected area

    #* calculate and save as txt file
    elif args.plot_type == 'calc':
        Vt_map = roop()
        print(np.amax(Vt_map))
        Vt_map = Vt_map / np.amax(Vt_map) # normalize
        np.savetxt(output_dir_path + '/semblance_map.txt', Vt_map, delimiter=',')
    else:
        print('error, input plot, mask, or calc')



    #* plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    if args.plot_type == 'select':
        bounds = np.array([0, 0.25, 0.50, 0.75, 1.0])
        cmap = mpl.colors.ListedColormap([plt.cm.Blues(int(255*i/3)) for i in range(4)])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(Vt_map, cmap=cmap, aspect='auto', interpolation='nearest',
            extent=[select_startx, select_endx, select_end, select_start],
            norm=norm
        )
        ax.minorticks_on()


    else:
        plt.imshow(Vt_map, cmap='jet', aspect='auto', interpolation='nearest',
                extent=[0, RMS_velocity[-1], Vt_map.shape[0]*params['time_step'], 0],
                norm=colors.LogNorm(vmin=1e-7, vmax=0.1) #! default: vmin=1e-7, vmax=0.1
                #norm=colors.LogNorm(vmin=1e-7, vmax=0.5)
        )

    ax.set_xlabel('RMS velocity [/c]', fontsize=20)
    ax.set_ylabel('Vertical delay time [ns]', fontsize=20)
    ax.grid(color='gray', linestyle='--', which='both', linewidth=0.5)
    ax.tick_params(labelsize=18)

    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax, label='Cross-correlation')
    cax.tick_params(labelsize=18)


    #* save plot
    if args.plot_type == 'select' and args.closeup == False:
        output_dir_path = output_dir_path + '/select'
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        plt.savefig(output_dir_path + '/semblance_map_select' + str(select_start) + '-' + str(select_end) + '.png')
    else:
        plt.savefig(output_dir_path + '/semblance_map.png')
    plt.show()