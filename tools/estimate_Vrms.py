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
import itertools


#* Parse command line arguments
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.Su_method jsonfile plot_type -closeup')
parser.add_argument('jsonfile', help='json file path')
parser.add_argument('plot_type', choices=['plot', 'mask', 'calc'])
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
vertical_delay_time = np.arange(0, 1501, 1) # 2-way travelt time in vertical direction, [ns]



#* load parameters from json file
antenna_step = params['src_step'] # antenna distance step, [m]
rx_start = params['rx_start'] # rx start position, [m]
src_start = params['src_start'] # src start position, [m]
src_end = params['src_end'] # src end position, [m]
src_move_times = params['src_move_times'] # number of src moving times
pulse_width = int(params['pulse_length'] / dt) # [data point]
transmit_delay = int(params['transmitting_delay'] / dt) # [data point]

path_num = nrx**2

#* make corr function
def corr(Vrms_ind, tau_ver_ind, i):
    rx_posi = rx_start + i * antenna_step # [m]

    Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
    tau_ver = vertical_delay_time[tau_ver_ind] * 1e-9 # [s]

    Amp_array = np.zeros(nrx) # 取り出した強度を入れる配列を用意しておく

    for src in range(src_move_times):

        src_posi = src_start + (src-1) * antenna_step # [m]
        offset = np.abs(rx_posi - src_posi) # [m]

        total_delay = int((np.sqrt((offset / Vrms)**2 + tau_ver**2) / dt))  + transmit_delay # [data point]

        if total_delay >= len(data_list[i]):
            Amp_array[src] = 0
        else:
            Amp_array[src] = np.abs(data_list[i][total_delay, src])


    return Amp_array # 1D array


#* caluculate corr roop
def corr_roop():
    corr_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))

    for v in range(len(RMS_velocity)):
        for t in tqdm(range(len(vertical_delay_time)), str(v*2/100) + 'c'):
            Amp_at_vt = np.array([corr(v, t, rx) for rx in range(nrx)]) # 2D array

            corr_matrix = np.abs(Amp_at_vt[:, None] * Amp_at_vt)
            corr_map[t, v] = np.sum(corr_matrix)

    corr_map = corr_map / path_num / (path_num - 1) # normalize
    return corr_map


#* make output directory
output_dir_path = data_dir_path + '/Vrms'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)


#* load only or calculate and save
if args.plot_type == 'plot':
    Vt_map = np.loadtxt(params['corr_map_txt'], delimiter=',')
    print(np.amax(Vt_map))
    Vt_map = Vt_map / np.amax(Vt_map) # normalize

elif args.plot_type == 'mask':
    Vt_map = np.loadtxt(params['corr_map_txt'], delimiter=',')
    Vt_map = Vt_map / np.amax(Vt_map) # normalize
    for row in Vt_map:
        indices_to_keep = np.argsort(row)[-5:]  # トップ5のインデックスを取得
        row[~np.isin(np.arange(len(row)), indices_to_keep)] = 0  # top5以外の要素を0に置き換える
    # 1e-6以下の値を0に置き換える
    Vt_map[Vt_map < 1e-6] = 0
    np.savetxt(output_dir_path + '/corr_map_mask.txt', Vt_map, delimiter=',')

elif args.plot_type == 'calc':
    Vt_map = corr_roop()
    Vt_map = Vt_map / np.amax(Vt_map) # normalize
    np.savetxt(output_dir_path + '/corr_map.txt', Vt_map, delimiter=',')
else:
    print('error, input plot, mask, or calc')


#* plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

plt.imshow(Vt_map, cmap='jet', aspect='auto', interpolation='nearest',
        extent=[RMS_velocity[0], RMS_velocity[-1], vertical_delay_time[-1], vertical_delay_time[0]],
        norm=colors.LogNorm(vmin=1e-10, vmax=5e-4))

ax.set_xlabel('RMS velocity [/c]')
ax.set_ylabel('Vertical delay time [ns]')
ax.grid(color='gray', linestyle='--')

"""
===== for closeup option =====
"""
if args.closeup == True:
    x_start = 0.4
    x_end = 0.7
    y_start = 100
    y_end = 200
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')

if args.plot_type == 'mask' and args.closeup == False:
    plt.savefig(output_dir_path + '/corr_map_mask.png')
elif args.closeup == True:
    plt.savefig(output_dir_path + '/corr_map_closeup' + str(y_start) + '-' + str(y_end) + '.png')
else:
    plt.savefig(output_dir_path + '/corr_map.png')
plt.show()


"""
#* caluculate and plot
def calc_corrmap_tarver():
    corr_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))
    for RX in range(4, 5):
        for v in tqdm(range(len(RMS_velocity)), desc='RX' + str(RX+1)):
            for t in range(len(vertical_delay_time)):
                corr_map[t, v] = corr(v, t, RX)


        #* plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        plt.imshow(corr_map, cmap='jet', aspect='auto', interpolation='nearest',
                extent=[RMS_velocity[0], RMS_velocity[-1], vertical_delay_time[-1], vertical_delay_time[0]],
                norm=colors.LogNorm(vmin=1e-5, vmax=1e3))

        ax.set_xlabel('RMS velocity [/c]')
        ax.set_ylabel('Vertical delay time [ns]')
        ax.set_title('Correlation map for rx' + str(RX + 1))
        ax.grid(color='gray', linestyle='--')

        delvider = axgrid1.make_axes_locatable(ax)
        cax = delvider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(cax=cax, label='Cross-correlation')

        plt.savefig(output_dir_path + '/corr_map_rx' + str(RX + 1) + '.png')
        #plt.show()
calc_corrmap_tarver()
"""




#! 深さとRMS速度で相関計算してみるやつ↓↓
interface_depth = np.arange(0, 150, 0.1) # [m]

def corr_depth_Vrms(depth_ind, Vrms_ind, rx):
    rx_posi = rx_start + rx * antenna_step # [m]

    Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
    depth = interface_depth[depth_ind] # [m]

    Amp_list= [] # 取り出した強度を入れる配列を用意しておく
    for src in range(nrx-1):

        src_posi = src_start + src * antenna_step # [m]
        offset = np.abs(rx_posi - src_posi) # [m]

        total_delay = np.sqrt( (2 * depth)**2 + offset**2) / Vrms # [s]
        total_delay = int(total_delay / dt) + transmit_delay # [data point]

        if total_delay >= len(data_list[rx]):
            Amp_list.append(0)
        else:
            #Amp = np.sum(np.abs(data_list[rx][total_delay-int(pulse_width/2): total_delay+int(pulse_width/2), src])) \
            #    / pulse_width
            Amp = np.abs(data_list[rx][total_delay, src])
            Amp_list.append(Amp)

    # Amp_timesから2つ選んで積をとり，その和を求める
    correlation =  sum(Amp1 * Amp2 for Amp1, Amp2 in itertools.combinations(Amp_list, 2))
    return correlation



#* caluculate and plot
def calc_corrmap_depth():
    corr_map = np.zeros((len(interface_depth), len(RMS_velocity)))
    for RX in range(1):
        for v in tqdm(range(len(RMS_velocity)), desc='RX' + str(RX+1)):
            for d in range(len(interface_depth)):
                corr_map[d, v] = corr(v, d, RX)


        #* plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        plt.imshow(corr_map, cmap='jet', aspect='auto', interpolation='nearest',
                extent=[RMS_velocity[0], RMS_velocity[-1], interface_depth[-1], interface_depth[0]],
                norm=colors.LogNorm(vmin=1e-5, vmax=1e3))

        ax.set_xlabel('RMS velocity [/c]')
        ax.set_ylabel('depth [m]')
        ax.set_title('Correlation map for rx' + str(RX + 1))
        ax.grid(color='gray', linestyle='--')

        delvider = axgrid1.make_axes_locatable(ax)
        cax = delvider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(cax=cax, label='Cross-correlation')

        plt.savefig(output_dir_path + '/corr_map_rx' + str(RX + 1) + '.png')
        plt.show()
#calc_corrmap_depth()




#! ↓なにこれ
""""
def calc_delay_time(rx_posi_index, src_posi_index):
    delay_time_array = np.zeros((len(RMS_velocity), len(vertical_delay_time)))
    rx_posi = rx_start + rx_posi_index * antenna_step
    src_posti = src_start + src_posi_index * antenna_step
    offset = np.abs(rx_posi - src_posti) # [m]


    for i in range(len(RMS_velocity)):
        Vrms = RMS_velocity[i] * c # [m/s]
        for j in range(len(vertical_delay_time)):
            tau_ver = vertical_delay_time[j] * 1e-6

            delay_time_array[j, i] = int(np.sqrt((offset / Vrms)**2 + (tau_ver)**2)/ dt) # index number

    return delay_time_array


#* とあるVrms, tau_verにおける相関計算
def corr(Vrms, tau_ver):
    # 取り出した強度を入れる配列を用意しておく
    Amp = []

    rx = 0 # rx1のこと
    for src in range(10):
        delay_time = calc_delay_time(rx, src)
        Amp.append(data_list[rx][delay_time, src])
    
    # Ampから2つ選んで積をとり，その和を求める
    cross_corr = sum(Amp1 *  Amp2 for Amp1, Amp2 in itertools.combinations(Amp, 2))

    return cross_corr

#! ↓なにこれ
#* 試しにrx1の場合で作ってみる
def calc_Vrms_inrx1():
    # 取り出した強度を入れる配列を用意しておく
    Amp = []

    rx = 0 # rx1のこと
    for src in range(10):
        delay_time = calc_delay_time(rx, src)
        Amp.append(data_list[rx][delay_time, src])
    
    # Ampから2つ選んで積をとり，その和を求める
    cross_corr = sum(Amp1 *  Amp2 for Amp1, Amp2 in itertools.combinations(Amp, 2))

    return cross_corr

corr_result = calc_Vrms_inrx1()
print(corr_result.shape)
"""
