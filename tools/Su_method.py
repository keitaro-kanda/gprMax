import argparse
import itertools
import json
import os
from cProfile import label
from math import tau
from re import A

import h5py
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from requests import get
from tqdm import tqdm

from tools.outputfiles_merge import get_output_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.Su_method jsonfile')
parser.add_argument('jsonfile', help='name of json file')
args = parser.parse_args()



# =====load files=====
with open (args.jsonfile) as f:
    params = json.load(f)


# Open output file and read number of outputs (receivers)
data_path = params['out_file']
data = h5py.File(data_path, 'r')
nrx = data.attrs['nrx']
data.close()
data_dir_path = os.path.dirname(data_path)
# =====load files=====


# =====set physical constants=====
c = 299792458 # [m/s], speed of light in vacuum
epsilon_0 = 1 # vacuum permittivity
# =====set physical constants=====



domain_x = params['domain_x']
ground_depth = params['ground_depth']
antenna_height = params['antenna_height']
domain_z = ground_depth + antenna_height # アンテナ高さをz=0とする
domain_array = np.zeros((domain_z, domain_x))


rx1, dt = get_output_data(data_path, 1, 'Ez') # shape=(29679, 10)
rx2, dt = get_output_data(data_path, 2, 'Ez')
rx3, dt = get_output_data(data_path, 3, 'Ez')
rx4, dt = get_output_data(data_path, 4, 'Ez')
rx5, dt = get_output_data(data_path, 5, 'Ez')
rx6, dt = get_output_data(data_path, 6, 'Ez')
rx7, dt = get_output_data(data_path, 7, 'Ez')
rx8, dt = get_output_data(data_path, 8, 'Ez')
rx9, dt = get_output_data(data_path, 9, 'Ez')
rx10, dt = get_output_data(data_path, 10, 'Ez')


# number of tx and rx
antenna_num = params['src_move_times']
rx_start = params['rx_start']
rx_step = params['rx_step']
src_start = params['src_start']
src_step = params['src_step']

rx_position_list = []
src_position_list = []
for i in range(1, antenna_num+1):
    rx_position_list.append(rx_start + rx_step * (i-1))
    src_position_list.append(src_start + src_step * (i-1))


L_ref2rx = np.zeros((domain_z, domain_x, antenna_num)) 
L_src2ref = np.zeros((domain_z, domain_x, antenna_num))
#print(L_rx2ref[1, 1, 1:])

for x in tqdm(range(domain_x)):
    for z in range(domain_z):
        # calculate distance between rx and (x,z)
        for i in range(antenna_num):
            L_ref2rx[z, x, i] = np.sqrt((rx_position_list[i]-x)**2 + (z**2))
            L_src2ref[z, x, i] = np.sqrt((src_position_list[i]-x)**2 + (z**2))

# L_src2ref, L_ref2rxは3次元の配列
# i番目のsrcにおいて，位置(x,z)に対する距離をL_src2ref[z, x, i]に格納している
# i番目のrxにおいて，位置(x,z)に対する距離をL_ref2rx[z, x, i]に格納している

tau_ref2rx = L_ref2rx / (c / np.sqrt(4)) # tau: [s], 試しに誘電率4で計算
tau_src2ref = L_src2ref / (c / np.sqrt(4)) # tau: [s], 試しに誘電率4で計算

# calculate cross-correlation
cross_corr = np.zeros((domain_z, domain_x))


def calc_corr():
    # forループの準備
    corr_list = []
    cross_corr = np.zeros((domain_z, domain_x))
    # 受信がrx1のみの場合
    for x in tqdm(range(domain_x)):
        for z in range(domain_z):
            print('x = ', x, ', z = ', z)
            tau = 5e-9 + tau_src2ref[z, x, 1:] + tau_ref2rx[z, x, 0] # tx1で送信の場合の遅れ時間[s]
            print('tau = ', int(tau[0]/dt), int(tau[1]/dt), int(tau[2]/dt), int(tau[3]/dt), int(tau[4]/dt), int(tau[5]/dt), int(tau[6]/dt), int(tau[7]/dt), int(tau[8]/dt))
            #  tauに対応するrx1の振幅を取得
            Amp =  np.zeros(antenna_num-1)
            Amp[0] = rx1[int(tau[0]/dt), 1] # rx1, src2
            Amp[1] = rx1[int(tau[1]/dt), 2] # rx1, src3
            Amp[2] = rx1[int(tau[2]/dt), 3] # rx1, src4
            Amp[3] = rx1[int(tau[3]/dt), 4] # rx1, src5
            Amp[4] = rx1[int(tau[4]/dt), 5] # rx1, src6
            Amp[5] = rx1[int(tau[5]/dt), 6] # rx1, src7
            Amp[6] = rx1[int(tau[6]/dt), 7] # rx1, src8
            Amp[7] = rx1[int(tau[7]/dt), 8] # rx1, src9
            Amp[8] = rx1[int(tau[8]/dt), 9] # rx1, src10
            print('Amp = ', Amp)

            # calculate cross-correlation
            for pair in itertools.combinations(Amp, 2):
                corr_list.append(np.abs(pair[0]) * np.abs(pair[1]))
            cross_corr[z, x] = np.sum(corr_list)
            #print('cross_corr = ', cross_corr[z, x])
                
    cross_corr = cross_corr / antenna_num / (antenna_num - 1) # 平均化
    return cross_corr

corr = calc_corr()

# plot
fig = plt.figure(facecolor='w', edgecolor='w')
ax = fig.add_subplot(111)
plt.imshow(corr, 
        extent=[0, corr.shape[1], corr.shape[0]-antenna_height, -antenna_height], 
        aspect=1, cmap='jet')

ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_title('Cross-correlation')

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')

plt.savefig(data_dir_path + '/cross_corr.png')
plt.show()

"""
        for rx in range(1, antenna_num+1):
            rx = rx_start + rx_step * (rx-1)

            # calculate distance between rx and (x,z)
            L_rx_ref = np.sqrt((rx-x)**2 + (z**2))
            tau_rx_ref = L_rx_ref / c

            for src in range(1, antenna_num+1):
                src = src_start + src_step * (src-1)
                # calculate distance between src and (x,z)
                L_src_ref = np.sqrt((src-x)**2 + (z**2))
                tau_src_ref = L_src_ref / c
        """
