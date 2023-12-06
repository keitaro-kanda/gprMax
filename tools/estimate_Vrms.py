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
                                 usage='cd gprMax; python -m tools.Su_method outfile')
parser.add_argument('outfile', help='.out file path')
args = parser.parse_args()


#* Open output file and read number of outputs (receivers)
#? h5pyを使ってデータを開ける意味はあまりないかも？nrx取得できるだけなのかな．
data_path = args.outfile
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
RMS_velocity = np.arange(0.01, 1.01, 0.01) # percentage to speed of light, 0% to 100%
vertical_delay_time = np.arange(0, 100, 0.1) # 2-way travelt time in vertical direction, [ns]


#! jsonに書いたほうがいいかも
antenna_step = 2.4 # antenna distance step, [m]
rx_start = 3
src_start = 3
#!!!!!!!!!!!

corr_map = np.zeros((len(vertical_delay_time), len(RMS_velocity)))

def corr(Vrms_ind, tau_ver_ind):
    rx = 0 # rx1のこと
    rx_posi = rx_start + rx * antenna_step # [m]

    Vrms = RMS_velocity[Vrms_ind] * c # [m/s]
    tau_ver = vertical_delay_time[tau_ver_ind] * 1e-9 # [s]

    for src in range(10):
        Amp_times = [] # 取り出した強度を入れる配列を用意しておく

        src_posi = src_start + src * antenna_step # [m]
        offset = np.abs(rx_posi - src_posi) # [m]

        total_delay = np.sqrt((offset / Vrms)**2 + tau_ver**2) # index number
        total_delay = int(total_delay / dt)
        if total_delay > len(data_list[rx]):
            Amp_times.append(0)
        else:
            Amp = (np.abs(data_list[rx][total_delay, src]) + np.abs(data_list[rx][total_delay+1, src]) \
                + np.abs(data_list[rx][total_delay-1, src]))/3
            Amp_times.append(np.abs(data_list[rx][total_delay, src]))
    corr_map[tau_ver_ind, Vrms_ind] = np.sum(Amp_times)


    return corr_map

for p in tqdm(range(len(RMS_velocity))):
    for t in range(len(vertical_delay_time)):
        corr_resutl = corr(p, t)


#* plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

plt.imshow(corr_map, cmap='viridis', aspect='auto', interpolation='nearest',
           extent=[RMS_velocity[0], RMS_velocity[-1], vertical_delay_time[-1], vertical_delay_time[0]])

ax.set_xlabel('RMS velocity [/c]')
ax.set_ylabel('Vertical delay time [ns]')
ax.set_title('Correlation map for rx1')

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')

plt.savefig(data_dir_path + '/corr_map.png')
plt.show()


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
