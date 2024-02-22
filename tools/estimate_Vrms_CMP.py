import numpy as np
import argparse
import json
import os

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm

import matplotlib as mpl
from multiprocessing import Pool


#* Parse command line arguments
parser = argparse.ArgumentParser(description='Processing De Pue et al. method',
                                usage='cd gprMax; python -m tools.Su_method jsonfile plot_type -closeup')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'select', 'calc'])
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)



#* check dt
data_path = params['out_file']
data = h5py.File(data_path, 'r')
dt = data.attrs['dt']
data.close()


#* load CMP txt data
data_path = params['cmp_file']
data = np.loadtxt(data_path, delimiter=',')
obs_num = data.shape[1]


#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum
epsilon_0 = 1 # vacuum permittivity



#* set calculation parameters
RMS_velocity = np.arange(0.01, 1.01, 0.01) # percentage to speed of light, 0% to 100%
vertical_delay_time = np.arange(0, params['time_window'], params['time_step']) # 2-way travelt time in vertical direction, [ns]



#* load parameters from json file
antenna_step = params['src_step'] # antenna distance step, [m]
src_end = params['src_end'] # src end position, [m]
pulse_width = int(params['pulse_length'] / dt) # [data point]
transmit_delay = params['transmitting_delay'] # [ns]
transmit_delay_point = int(transmit_delay / dt) # [data point]
antenna_height = params['antenna_height'] # [m]



#* difine equations to solve
def DePue_eq9(y1, y0, z0, z1, v1):
    # De Pue eq(9b)
    lateral_air = (y0 - y1) / 2
    sin0 = lateral_air / np.sqrt(lateral_air**2 + z0**2)
    # De Pue eq(9c)
    lateral_ground = (y1 / 2)
    sin1 = lateral_ground / np.sqrt(lateral_ground**2 + z1**2)

    return sin0 / sin1 - c / v1



#* make function to calcurate semblance
def calc_semblance(Vrms_ind, t0_ind, i): # i: in range(obs_num)
    Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
    t0 = vertical_delay_time[t0_ind] * 10**(-9) # [s]
    depth = Vrms * t0 / 2 # [m]


    # calculate ground offset
    #offset = antenna_step * 2 * (i + 1) # [m]
    rx_posi = antenna_step * i # [m]
    tx_posi = src_end - antenna_step * i # [m]
    offset = np.abs(rx_posi - tx_posi) # [m]
    if offset == 0:
        offset_ground_solution = 0
    else:
        offset_ground = np.linspace(offset/100, offset, 100)
        DePue_eq9_result = DePue_eq9(offset_ground, offset, antenna_height, depth, Vrms)
        offset_ground_solution = offset_ground[np.argmin(np.abs(DePue_eq9_result))]

    # calculate delay time
    delay_time = np.sqrt((offset - offset_ground_solution)**2 + 4 * antenna_height**2) / c \
            + t0  + offset_ground_solution**2 / Vrms**2
    total_delay_time = delay_time + transmit_delay * 10**(-9) # [s]
    total_delay_point = int(total_delay_time / dt) # data point


    # search data
    if total_delay_point >= data.shape[0]:
        echo_intensity = 0
    else:
        echo_intensity = np.abs(data[total_delay_point, i])

    return echo_intensity



#* roop calc_semblance
def roop():
    semblance_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))

    for v in tqdm(range(len(RMS_velocity)), desc='calc_semblance'):
        for t in tqdm(range(len(vertical_delay_time)), desc='v=' + str(v)):
            power_list = []
            for i in range(obs_num):
                semblance_value = calc_semblance(v, t, i)
                power_list.append(np.sum(semblance_value))
            #semblance_value = sum([sum(calc_semblance(v, t, i)) for i in range(int(nrx/2))])**2
            #semblance_map[t, v] = semblance_value`
            semblance_map[t, v] = np.sum(power_list)**2

    return semblance_map


"""
Palarellel processing
"""
def worker_func(args):
    v, t, RMS_velocity, vertical_delay_time, obs_num = args
    power_list = [np.sum(calc_semblance(v, t, i)) for i in range(obs_num)]
    return t, v, np.sum(power_list)**2

def calc_semblance_parallel(RMS_velocity, vertical_delay_time, obs_num):
    semblance_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))
    args = [(v, t, RMS_velocity, vertical_delay_time, obs_num) \
            for v in range(len(RMS_velocity)) for t in range(len(vertical_delay_time))]

    with Pool() as pool:
        results = list(tqdm(pool.imap(worker_func, args), total=len(args), desc='Calculating Semblance'))
    
    for t, v, semblance_value in results:
        semblance_map[t, v] = semblance_value

    return semblance_map




# Ensure this part is under the __main__ guard
if __name__ == '__main__':


    #* make output directory
    output_dir_path = os.path.join(os.path.dirname(args.jsonfile), 'Vrms')
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)



    #* load only or calculate and save
    """
    select area [ns]
    """
    select_start = 60
    select_end = 100
    select_startx = 0
    select_endx = 0.6
    """
    select area [ns]
    """

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
        #Vt_map = roop()
        Vt_map = calc_semblance_parallel(RMS_velocity, vertical_delay_time, obs_num)
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
                norm=colors.LogNorm(vmin=1e-10, vmax=0.1) #! default: vmin=1e-7, vmax=0.1
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
    if args.plot_type == 'select':
        output_dir_path = output_dir_path + '/select'
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        plt.savefig(output_dir_path + '/semblance_map_select' + str(select_start) + '-' + str(select_end) + '.png')
    else:
        plt.savefig(output_dir_path + '/semblance_map.png')
    plt.show()