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
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.Su_method jsonfile plot_type -closeup')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'mask', 'select', 'calc'])
parser.add_argument('-closeup', action='store_true', help='closeup of the plot', default=False)
parser.add_argument('-CMP', action='store_true', help='analyse as CMP observation', default=False)
parser.add_argument('--select-points', action='store_true', help='select points', default=False)
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* Open output file and read number of outputs (receivers)
#? h5pyを使ってデータを開ける意味はあまりないかも？nrx取得できるだけなのかな．
data_path = params['data']
data = h5py.File(data_path, 'r')
#if args.select_points == False:
nrx = data.attrs['nrx']
#elif args.select_points == True:
#    nrx = 9
data.close()
data_dir_path = os.path.dirname(data_path)

#* load data
data_list = []
#if args.select_points == False:
for i in range(1, nrx+1):
    data, dt = get_output_data(data_path, i, 'Ez')
    data_list.append(data)
#elif args.select_points == True:
point_num = 17
start_point = 13 # the number of points to start
end_point = start_point + point_num - 1
#    for i in range(start_point, end_point +1, 1):
#        data, dt = get_output_data(data_path, i, 'Ez')
#        data_list.append(data)



#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum
epsilon_0 = 1 # vacuum permittivity



#* set calculation parameters
RMS_velocity = np.arange(0.02, 1.02, 0.02) # percentage to speed of light, 0% to 100%
vertical_delay_time = np.arange(0, params['time_window'], params['time_step']) # 2-way travelt time in vertical direction, [ns]



#* load parameters from json file
antenna_step = params['antenna_settings']['src_step'] # antenna distance step, [m]
rx_start = params['antenna_settings']['rx_start'] # rx start position, [m]
src_start = params['antenna_settings']['src_start'] # src start position, [m]
src_end = params['antenna_settings']['src_end'] # src end position, [m]
src_move_times = params['antenna_settings']['src_move_times'] # the number of src move times
pulse_width = int(params['pulse_length'] / dt) # [data point]
transmit_delay = params['transmitting_delay'] # [ns]
transmit_delay_point = int(transmit_delay / dt) # [data point]

#path_num = nrx**2

#* make corr function
def calc_corr(Vrms_ind, tau_ver_ind, i): # i: in range(nrx)
    if args.CMP == True:
        rx_ind = i # rx: 0 -> nrx-1
        tx_ind = i # tx: nrx-1 -> 0

        # calculate offset
        rx_posi = rx_ind * antenna_step # [m]
        tx_posi = tx_ind * antenna_step # [m]
        offset = np.abs(rx_posi - tx_posi) # [m]

        Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
        tau_ver = vertical_delay_time[tau_ver_ind] * 1e-9 # [s]

        # calculate total delay time t
        total_delay = int((np.sqrt((offset / Vrms)**2 + tau_ver**2) / dt))  + transmit_delay_point # [data point]
        Amp_array = np.zeros(nrx) # 取り出した強度を入れる配列を用意しておく
        if total_delay >= len(data_list):
                Amp_array[rx_ind] = 0
        else:
            Amp_array[i] = np.abs(data_list[rx_ind][total_delay, tx_ind]) # Ez intensity in (t, src) at i-th rx

        return Amp_array # 1D array


    else:
        rx_posi = rx_start + i * antenna_step # [m]

        Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
        tau_ver = vertical_delay_time[tau_ver_ind] * 1e-9 # [s]

        Amp_array = [] # 取り出した強度を入れる配列を用意しておく

        #* use all observation points
        if args.select_points == False:
            for src in range(src_move_times):

                src_posi = src_start + (src-1) * antenna_step # [m]
                offset = np.abs(rx_posi - src_posi) # [m]

                # calculate total delay time t
                if Vrms == 0:
                    total_delay = transmit_delay_point
                else:
                    total_delay = int((np.sqrt((offset / Vrms)**2 + tau_ver**2) / dt))  \
                        + transmit_delay_point # [data point]

                # calculate shifted hyperbola
                #Sx = (offset**2 / Vrms**2) - 2 * tau_ver * ()

                if total_delay >= len(data_list):
                    Amp_array.append(0)
                else:
                    Amp_array.append(np.abs(data_list[i][total_delay, src])) # Ez intensity in (t, src) at i-th rx


        #* use selected observation points
        elif args.select_points == True:
            for src in range(start_point, end_point + 1, 1):

                src_posi = src_start + (src-1) * antenna_step
                offset = np.abs(rx_posi - src_posi) # [m]

                # calculate total delay time t
                if Vrms == 0:
                    total_delay = transmit_delay_point
                else:
                    total_delay = int((np.sqrt((offset / Vrms)**2 + tau_ver**2) / dt))  \
                        + transmit_delay_point # [data point]

                # calculate shifted hyperbola
                #Sx = (offset**2 / Vrms**2) - 2 * tau_ver * ()

                if total_delay >= len(data_list[i]):
                    Amp_array.append(0)
                else:
                    Amp_array.append(np.abs(data_list[i][total_delay, src])) # Ez intensity in (t, src) at i-th rx

        return np.array(Amp_array) # 1D array


#* caluculate corr roop
def roop_corr():
    corr_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))

    for v in tqdm(range(len(RMS_velocity))):
        for t in range(len(vertical_delay_time)):
            if args.CMP == True:
                Amp_at_vt = np.array([calc_corr(v, t, rx) for rx in range(src_move_times)]) # 1D array
                corr_matrix = np.abs([a *b for a, b in combinations(Amp_at_vt, 2)]) # 1D array

            else:
                if args.select_points == False:
                    Amp_at_vt = np.array([calc_corr(v, t, rx) for rx in range(nrx)])
                elif args.select_points == True:
                    Amp_at_vt = np.array([calc_corr(v, t, rx) for rx in range(start_point, end_point + 1, 1)]) # 1D array
                corr_matrix = np.abs(Amp_at_vt[:, None] * Amp_at_vt)
                #corr_map[t, v] = np.sum(calc_corr(v, t, 11)) # 1D array

            corr_map[t, v] = np.sum(corr_matrix)
            #corr_map[t, v] = np.sum(Amp_at_vt)

    #corr_map = corr_map / path_num / (path_num - 1) # normalize
    return corr_map


#* make output directory
if args.CMP == True:
    output_dir_path = data_dir_path + '/Vrms_CMP'
elif args.select_points == True:
    output_dir_path = data_dir_path + '/Vrms_select_points'
else:
    output_dir_path = data_dir_path + '/Vrms'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)


#* load only or calculate and save

"""
select area [ns]
"""
select_start = 530
select_end = 580
select_startx = 0
select_endx = 1.0
"""
select area [ns]
"""

if args.plot_type == 'plot':
    Vt_map = np.loadtxt(params['corr_map_txt'], delimiter=',')

elif args.plot_type == 'mask':
    Vt_map = np.loadtxt(params['corr_map_txt'], delimiter=',')
    # 1e-6以下の値を0に置き換える
    #Vt_map[Vt_map < 1e-6] = 0

    max_values_col = np.argmax(Vt_map, axis=1)  # 各行の最大値を取得
    max_val = np.max(Vt_map, axis=1)  # 各行の最大値を取得
    threshold = 0.5 * max_val  # 最大値の50%
    for row in range(Vt_map.shape[0]):

        #Vt_map[row][Vt_map[row] >= threshold[row]] = 1e-4  # 50%以下の値を0に置き換える
        Vt_map[row][Vt_map[row] < threshold[row]] = 0  # 50%以下の値を0に置き換える
        #Vt_map[row][max_values_col[row]] = 1
        #Vt_map[row][Vt_map[row] < max_val[row]] = 0  # 50%以下の値を0に置き換える


#* select Vt_map area
elif args.plot_type == 'select':
    if args.CMP == True:
        Vt_map = np.loadtxt(params['cmp_corr_map_txt'], delimiter=',')
    else:
        Vt_map = np.loadtxt(params['corr_map_txt'], delimiter=',') # load data
    Vt_map = Vt_map[int(select_start/params['time_step']): int(select_end/params['time_step']), int(select_startx/0.02): int(select_endx/0.02)] # select area
    Vt_map = Vt_map / np.amax(Vt_map) # normalize by max value in selected area


elif args.plot_type == 'calc':
    Vt_map = roop_corr()
    print(np.amax(Vt_map))
    Vt_map = Vt_map / np.amax(Vt_map) # normalize
    np.savetxt(output_dir_path + '/corr_map.txt', Vt_map, delimiter=',')
else:
    print('error, input plot, mask, or calc')


#* plot

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)

#* show true value points
Vrms_theory = params['Vrms_theory'] # [/c]
t0_theory = params['t0_theory'] # [ns]
plt.scatter(Vrms_theory, t0_theory, c='k', s=70, marker='P', edgecolors='w', label='Theoretical value')
plt.legend(fontsize=18, loc='lower right')


if args.plot_type == 'select':
    bounds = np.array([0, 0.25, 0.50, 0.75, 1.0])
    cmap = mpl.colors.ListedColormap([plt.cm.Blues(int(255*i/3)) for i in range(4)])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(Vt_map, cmap=cmap, aspect='auto', interpolation='nearest',
        extent=[select_startx, select_endx, select_end, select_start],
        #norm=colors.LogNorm(vmin=1e-2, vmax=1)
        norm=norm
    )
    ax.minorticks_on()
    # countour
    #cont = ax.contour(Vt_map, 3, colors='k', linewidths=1, linestyles=['-', '--', '-.'],
    #        extent=[select_startx, select_endx, select_start, select_end],)
    #ax.clabel(cont, inline=True, fontsize=10, fmt='%.2f')

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



#* for closeup option
if args.closeup == True:
    x_start = 0
    x_end = 1
    y_start = 60
    y_end = 100
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')
cax.tick_params(labelsize=18)

#* save plot
if args.plot_type == 'mask' and args.closeup == False:
    plt.savefig(output_dir_path + '/corr_map_mask.png')
elif args.plot_type == 'select' and args.closeup == False:
    output_dir_path = output_dir_path + '/select'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    plt.savefig(output_dir_path + '/corr_map_select' + str(select_start) + '-' + str(select_end) + '.png')
elif args.closeup == True:
    output_dir_path = output_dir_path + '/closeup'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    plt.savefig(output_dir_path + '/corr_map_closeup' + str(y_start) + '-' + str(y_end) + '.png')
else:
    plt.savefig(output_dir_path + '/corr_map.png')
plt.show()
