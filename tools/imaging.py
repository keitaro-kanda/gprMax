import argparse
import json
import os

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from tqdm import tqdm

from tools.outputfiles_merge import get_output_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.imaging jsonfile velocity_structure')
parser.add_argument('jsonfile', help='name of json file')
parser.add_argument('velocity_structure', choices=['y', 'n'], help='whether to use velocity structure or not')
args = parser.parse_args()



#* load jason file
with open (args.jsonfile) as f:
    params = json.load(f)


#* Open output file and read number of outputs (receivers)
data_path = params['out_file']
data = h5py.File(data_path, 'r')
nrx = data.attrs['nrx']
data.close()
data_dir_path = os.path.dirname(data_path)
# =====load files=====


#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum
epsilon_0 = 1 # vacuum permittivity


#* load setting from json file
domain_x = params['domain_x']
ground_depth = params['ground_depth']
antenna_height = params['antenna_height']
domain_z = ground_depth + antenna_height # アンテナ高さをz=0とする
domain_array = np.zeros((domain_z, domain_x))

# number of tx and rx
antenna_num = params['src_move_times']
rx_start = params['rx_start']
rx_step = params['rx_step']
src_start = params['src_start']
src_step = params['src_step']



#* load data
data_list = []
for i in range(1, nrx+1):
    data, dt = get_output_data(data_path, i, 'Ez')
    data_list.append(data)



#* calc array position
rx_position_list = []
src_position_list = []
for i in range(1, antenna_num+1):
    rx_position_list.append(rx_start + rx_step * (i-1))
    src_position_list.append(src_start + src_step * (i-1))


imaging_resolution = params['imaging_resolution'] # [m]
imaging_grid_x = int(domain_x / imaging_resolution) # number of grid in x direction
imaging_grid_z = int(domain_z / imaging_resolution) # number of grid in z direction


#* caluculate propagation path length for each sets of rx and src
"""
L_src2ref, L_ref2rxは3次元の配列
i番目のsrcにおいて，位置(x,z)に対する距離をL_src2ref[z, x, i]に格納している
i番目のrxにおいて，位置(x,z)に対する距離をL_ref2rx[z, x, i]に格納している
"""
L_ref2rx = np.zeros((imaging_grid_z, imaging_grid_x, antenna_num))
L_src2ref = np.zeros((imaging_grid_z, imaging_grid_x, antenna_num))

for x_index in tqdm(range(imaging_grid_x), desc='calculating path length'):
    x = x_index * imaging_resolution
    for z_index in range(imaging_grid_z):
        z = z_index * imaging_resolution
        # calculate distance between rx and (x,z)
        for i in range(antenna_num):
            L_ref2rx[z_index, x_index, i] = np.sqrt((rx_position_list[i]-x)**2 + (z**2))
            L_src2ref[z_index, x_index, i] = np.sqrt((src_position_list[i]-x)**2 + (z**2))



#* calculate propagation time for each sets of rx and src
if args.velocity_structure == 'n':
    """
    assume that epsilon_r = 4 in all grid
    """
    tau_ref2rx = L_ref2rx / (c / np.sqrt(4)) # tau: [s] from ref to rx
    tau_src2ref = L_src2ref / (c / np.sqrt(4)) # tau: [s] from src to ref

elif args.velocity_structure == 'y':
    """
    Consider the results of V_RMS estimation
    """
    #! ただし，２層分のV_RMSにしか対応していない
    layer1_Vrms = params['V_RMS_1'] * c # [m/s]
    layer1_thickness = int(params['tau_ver_1'] * layer1_Vrms / 2 / imaging_resolution) # grind number
    layer2_Vrms = params['V_RMS_2'] * c # [m/s]
    layer2_thickness = params['tau_ver_2'] * layer2_Vrms / 2 # [m]

    tau_ref2rx = np.vstack((L_ref2rx[0:layer1_thickness, :, :] / layer1_Vrms,
                        L_ref2rx[layer1_thickness:, :, :] / layer2_Vrms)) # tau: [s] from ref to rx
    tau_src2ref = np.vstack((L_src2ref[0:layer1_thickness, :, :] / layer1_Vrms,
                        L_src2ref[layer1_thickness:, :, :] / layer2_Vrms))

else:
    print('error, input y or n')



#* calculate cross-correlation
cross_corr = np.zeros((domain_z, domain_x))

def calc_Amp(z, x, i): # i: 0~antenna_num-1を入力する
    # i番目のrxにおける遅れ時間配列（1D）を作成
    # 簡単のため，i番目のsrcで送信してi番目のrxで受診するのもOKとする
    tau = 8.73e-9 + tau_src2ref[z, x, :] + tau_ref2rx[z, x, i] # i番目のrxで受信する場合の遅れ時間[s]
    tau_index = (tau / dt).astype(int) # tauをインデックス番号に変換

    # i番目のrxにおける振幅配列（1D）を作成
    Amp_array = np.zeros(antenna_num)# １次元配列, 行：rxインデックス, 列：srcインデックス

    for src in range(antenna_num):
        # i番目のrxにおける，src番目のsrcで送信したときの振幅を取得
        Amp_array[src] = data_list[i-1][tau_index[src], src]

        """
        rx_data, dt = get_output_data(data_path, i+1, 'Ez')
        Amp_array[src] = rx_data[tau_index, src]
        """
    return Amp_array # 1次元配列を返す


#* caluculate cross-correlation for each grids
def calc_corr():
    # forループの準備
    cross_corr = np.zeros((imaging_grid_z, imaging_grid_x))
    #Amp_at_xz = np.zeros((antenna_num, antenna_num))
    for x in tqdm(range(imaging_grid_x), desc='calculating cross-correlation'):
        for z in range(imaging_grid_z):

            Amp_at_xz = np.array([calc_Amp(z, x, rx) for rx in range(antenna_num)])
            # ↑Amp_at_xzは1次元配列Amp_arrayを結合した2次元配列

            corr_matrix = np.abs(Amp_at_xz[:, None] * Amp_at_xz)
            cross_corr[z, x] = np.sum(corr_matrix)

    path_num = antenna_num * (antenna_num - 1)
    corr_xz = cross_corr / path_num/ (path_num - 1) # 平均化

    return corr_xz


#* 関数の実行
corr = calc_corr()


#* make output dir and save data as txt file
output_dir_path = data_dir_path + '/imaging'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)

np.savetxt(output_dir_path + '/imaging_result.csv', corr, delimiter=',')


#* plot
fig = plt.figure(figsize=(5, 5*corr.shape[0]/corr.shape[1]) ,facecolor='w', edgecolor='w')
ax = fig.add_subplot(111)
plt.imshow(corr, 
        extent=[0, corr.shape[1] * imaging_resolution, 
                corr.shape[0]*imaging_resolution-antenna_height, -antenna_height], 
        aspect=1, cmap='jet', norm=colors.LogNorm(vmin=1e-5, vmax=np.amax(corr)))

ax.set_xlabel('x [m]', fontsize=14)
ax.set_ylabel('z [m]', fontsize=14)
ax.set_title('Imaging by Su+(2022) method', fontsize=14)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')

plt.savefig(output_dir_path + '/imaging_result.png')
plt.show()

