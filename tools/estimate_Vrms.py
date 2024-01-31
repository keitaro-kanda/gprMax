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


#* Parse command line arguments
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.Su_method jsonfile plot_type -closeup')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'mask', 'select', 'calc'])
parser.add_argument('-closeup', action='store_true', help='closeup of the plot', default=False)
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* Open output file and read number of outputs (receivers)
#? h5pyを使ってデータを開ける意味はあまりないかも？nrx取得できるだけなのかな．
data_path = params['out_file']
data = h5py.File(data_path, 'r')
nrx = data.attrs['nrx']
data.close()
data_dir_path = os.path.dirname(data_path)

#* load data
data_list = []
for i in range(1, nrx+1):
    data, dt = get_output_data(data_path, i, 'Ez')
    data_list.append(data)



#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum
epsilon_0 = 1 # vacuum permittivity



#* set calculation parameters
RMS_velocity = np.arange(0.01, 1.01, 0.02) # percentage to speed of light, 0% to 100%
vertical_delay_time = np.arange(0, params['time_window'], params['time_step']) # 2-way travelt time in vertical direction, [ns]



#* load parameters from json file
antenna_step = params['src_step'] # antenna distance step, [m]
rx_start = params['rx_start'] # rx start position, [m]
src_start = params['src_start'] # src start position, [m]
src_end = params['src_end'] # src end position, [m]
src_move_times = params['src_move_times'] # number of src moving times
pulse_width = int(params['pulse_length'] / dt) # [data point]
transmit_delay = params['transmitting_delay'] # [ns]
transmit_delay_point = int(transmit_delay / dt) # [data point]

path_num = nrx**2

#* make corr function
def calc_corr(Vrms_ind, tau_ver_ind, i):
    rx_posi = rx_start + i * antenna_step # [m]

    Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
    tau_ver = vertical_delay_time[tau_ver_ind] * 1e-9 # [s]

    Amp_array = np.zeros(nrx) # 取り出した強度を入れる配列を用意しておく

    for src in range(src_move_times):

        src_posi = src_start + (src-1) * antenna_step # [m]
        offset = np.abs(rx_posi - src_posi) # [m]

        # calculate total delay time t
        total_delay = int((np.sqrt((offset / Vrms)**2 + tau_ver**2) / dt))  + transmit_delay_point # [data point]

        # calculate shifted hyperbola
        #Sx = (offset**2 / Vrms**2) - 2 * tau_ver * ()

        if total_delay >= len(data_list[i]):
            Amp_array[src] = 0
        else:
            Amp_array[src] = np.abs(data_list[i][total_delay, src]) # Ez intensity in (t, src) at i-th rx


    return Amp_array # 1D array


#* caluculate corr roop
def roop_corr():
    corr_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))

    for v in range(len(RMS_velocity)):
        for t in tqdm(range(len(vertical_delay_time)), str(v*2/100) + 'c'):
            Amp_at_vt = np.array([calc_corr(v, t, rx) for rx in range(nrx)]) # 2D array
            #corr_map[t, v] = np.sum(calc_corr(v, t, 11)) # 1D array
            corr_matrix = np.abs(Amp_at_vt[:, None] * Amp_at_vt)
            corr_map[t, v] = np.sum(corr_matrix)

    corr_map = corr_map / path_num / (path_num - 1) # normalize
    return corr_map


#* make output directory
output_dir_path = data_dir_path + '/Vrms'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)


#* load only or calculate and save

"""
select area [ns]
"""
select_start = 350
select_end = 370
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
    
    #! トップ5のみ残す
    """
    for row in Vt_map:
        indices_to_keep = np.argsort(row)[-5:]  # トップ5のインデックスを取得
        row[~np.isin(np.arange(len(row)), indices_to_keep)] = 0  # top5以外の要素を0に置き換える
    # 1e-6以下の値を0に置き換える
    #Vt_map[Vt_map < 1e-6] = 0
    np.savetxt(output_dir_path + '/corr_map_mask.txt', Vt_map, delimiter=',')
    """
    #! トップ5のみ残す


elif args.plot_type == 'select':

    #* select Vt_map area
    Vt_map = np.loadtxt(params['corr_map_txt'], delimiter=',') # load data
    Vt_map = Vt_map[int(select_start/params['time_step']): int(select_end/params['time_step']), :] # select area
    Vt_map = Vt_map / np.amax(Vt_map) # normalize by max value in selected area

    """
    # Vt_mapの最大値の50%以下の値を0に置き換える
    max_val = np.amax(Vt_map)  # get max value in selected area
    for row in range(Vt_map.shape[0]):
        Vt_map[row][Vt_map[row] < 0.5*max_val] = 0
    """



elif args.plot_type == 'calc':
    Vt_map = roop_corr()
    Vt_map = Vt_map / np.amax(Vt_map) # normalize
    np.savetxt(output_dir_path + '/corr_map.txt', Vt_map, delimiter=',')
else:
    print('error, input plot, mask, or calc')


#* plot

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
if args.plot_type == 'select':
    plt.imshow(Vt_map, cmap='plasma', aspect='auto', interpolation='nearest',
        extent=[0, RMS_velocity[-1], select_end, select_start],
        norm=colors.LogNorm(vmin=1e-2, vmax=1)
)
    # countour
    cont = ax.contour(Vt_map, 3, colors='k', linewidths=1, linestyles=['-', '--', '-.'],
            extent=[0, RMS_velocity[-1], select_start, select_end],)
    ax.clabel(cont, inline=True, fontsize=10, fmt='%.2f')

else:
    plt.imshow(Vt_map, cmap='jet', aspect='auto', interpolation='nearest',
            extent=[0, RMS_velocity[-1], Vt_map.shape[0]*params['time_step'], 0],
            norm=colors.LogNorm(vmin=1e-7, vmax=0.1) #! default: vmin=1e-7, vmax=0.1
            #norm=colors.LogNorm(vmin=1e-7, vmax=0.5)
    )

ax.set_xlabel('RMS velocity [/c]')
ax.set_ylabel('Vertical delay time [ns]')
ax.grid(color='gray', linestyle='--')

#* for closeup option
if args.closeup == True:
    x_start = 0
    x_end = 1
    y_start = 260
    y_end = 300
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')

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

