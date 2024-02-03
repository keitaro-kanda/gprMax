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
                                 usage='cd gprMax; python -m tools.imaging jsonfile velocity_structure -plot')
parser.add_argument('jsonfile', help='name of json file')
parser.add_argument('velocity_structure', choices=['y', 'n'], help='whether to use velocity structure or not')
parser.add_argument('-plot', action='store_true', help='option: plot only')
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
            L_ref2rx[z_index, x_index, i] = np.sqrt((rx_position_list[i]-x)**2 + (z**2)) # [m]
            L_src2ref[z_index, x_index, i] = np.sqrt((src_position_list[i]-x)**2 + (z**2)) # [m]




#* calculate propagation time for each sets of rx and src
if args.velocity_structure == 'n':
    #! setting constant Vrms
    constant_Vrms = 0.5 # [/c]
    #! setting constant Vrms

    tau_ref2rx = L_ref2rx / (c * constant_Vrms) # tau: [s] from ref to rx
    tau_src2ref = L_src2ref / (c * constant_Vrms) # tau: [s] from src to ref

elif args.velocity_structure == 'y':
    """
    Consider the results of V_RMS estimation
    """
    # check the number of layers and load parameters of each layers
    num_layers = len(params['V_RMS'])  # number of layers
    layer_info = np.zeros((num_layers, 2))  # layer_info[i, 0]: V_RMS, layer_info[i, 1]: tau_ver
    for i in range(num_layers):
        #V_rms_key = f'V_RMS_{i+1}'
        #tau_ver_key = f'tau_ver_{i+1}'

        area_Vrms = params['V_RMS'][i] * c  # [m/s]
        #area_thickness = params['tau_ver'][i] * area_Vrms / 2
        #print(f"Layer {i+1} - depth: {area_thickness}")
        area_thickness = int(params['tau_ver'][i] * area_Vrms / 2 / imaging_resolution)  # grid number, 'grid' means imaging grid

        layer_info[i, 0] = area_Vrms # [m/s]
        layer_info[i, 1] = area_thickness # grid number, 'grid' means imaging grid

    # calculate tau_ref2rx and tau_src2ref for each resion
    tau_ref2rx = np.zeros((imaging_grid_z, imaging_grid_x, antenna_num)) # [s], size is same as imaging grid
    tau_src2ref = np.zeros((imaging_grid_z, imaging_grid_x, antenna_num)) # [s], size is same as imaging grid
    for i in range(num_layers):
        # from surface to first boundary
        if i == 0:
            area_depth = int(layer_info[i, 1]) # intをつけないとなぜか動かない
            tau_ref2rx[0:area_depth, :, :] = L_ref2rx[0:area_depth, :, :] / layer_info[i, 0] # [s]
            tau_src2ref[0:area_depth, :, :] = L_src2ref[0:area_depth, :, :] / layer_info[i, 0] #[s]
        # where deeper than last boundary
        elif i == num_layers-1:
            area_depth = int(layer_info[i-1, 1]) # intをつけないとなぜか動かない
            tau_ref2rx[area_depth:, :, :] = L_ref2rx[area_depth:, :, :] / layer_info[i, 0] # [s]
            tau_src2ref[area_depth:, :, :] = L_src2ref[area_depth:, :, :] / layer_info[i, 0] #[s]
        # others, between each boundaries
        else:
            area_depth_start = int(np.sum(layer_info[i-1, 1])) # intをつけないとなぜか動かない
            area_depth_end = int(np.sum(layer_info[i, 1])) # intをつけないとなぜか動かない
            tau_ref2rx[area_depth_start:area_depth_end, :, :] \
                = L_ref2rx[area_depth_start:area_depth_end, :, :] / layer_info[i, 0] #[s]
            tau_src2ref[area_depth_start:area_depth_end, :, :] \
                = L_src2ref[area_depth_start:area_depth_end, :, :] / layer_info[i, 0] #[s]
    #tau_ref2rx[layer_info[num_layers]:, :, :] = L_ref2rx[layer_info[num_layers]:, :, :] / layer_info[num_layers, 0] #[s]
    #tau_src2ref[layer_info[num_layers]:, :, :] = L_src2ref[layer_info[num_layers]:, :, :] / layer_info[num_layers, 0] #[s]
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
    """

else:
    print('error, input y or n')


#* calculate cross-correlation
cross_corr = np.zeros((domain_z, domain_x))

def calc_Amp(z, x, i): # i: 0~antenna_num-1を入力する
    # i番目のrxにおける遅れ時間配列（1D）を作成
    # 簡単のため，i番目のsrcで送信してi番目のrxで受診するのもOKとする
    tau = 8.73e-9 + tau_src2ref[z, x, :] + tau_ref2rx[z, x, i] # i番目のrxで受信する場合の遅れ時間[s]
    tau_index = (tau / dt).astype(int) # convert tau[s] to index, 'index' means index of A-scan data

    # i番目のrxにおける振幅配列（1D）を作成
    Amp_array = np.zeros(antenna_num)# １次元配列, 行：rxインデックス, 列：srcインデックス

    for src in range(antenna_num):
        # i番目のrxにおける，src番目のsrcで送信したときの振幅を取得
        if tau_index[src] < len(data_list[i-1]):
            Amp_array[src] = data_list[i-1][tau_index[src], src]
        elif tau_index[src] >= len(data_list[i-1]):
            Amp_array[src] = 0

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

#* make output dir and save data as txt file
output_dir_path = data_dir_path + '/imaging'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)


#* calculate and save as txt file
if args.plot == False:

    corr = calc_corr()
    corr = corr / np.amax(corr) # normalize


    #* save as txt file
    if args.velocity_structure == 'n':
        np.savetxt(output_dir_path + '/imaging_result_n'+str(constant_Vrms)+'.csv', corr, delimiter=',')
    elif args.velocity_structure == 'y':
        np.savetxt(output_dir_path + '/imaging_result_y.csv', corr, delimiter=',')

#* don't calculate, only plot
elif args.plot == True:
    corr = np.loadtxt(params['imaging_result_csv'], delimiter=',')


#* plot
fig = plt.figure(figsize=(5, 5*corr.shape[0]/corr.shape[1]) ,facecolor='w', edgecolor='w')
ax = fig.add_subplot(111)
plt.imshow(corr, 
        extent=[0, corr.shape[1] * imaging_resolution, 
                corr.shape[0]*imaging_resolution-antenna_height, -antenna_height], 
        aspect=1,
        cmap='jet', # recomended: 'jet', 'binary'
        norm=colors.LogNorm(vmin=1e-5, vmax=np.amax(corr)))

ax.set_yticks(np.arange(0, corr.shape[0]*imaging_resolution, 10))

ax.set_xlabel('x [m]', fontsize=14)
ax.set_ylabel('z [m]', fontsize=14)
ax.set_title('Imaging result', fontsize=14)

#ax.grid(which='major', color='white', linestyle='--')

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Cross-correlation')

if args.velocity_structure == 'n':
    plt.savefig(output_dir_path + '/imaging_result_n'+str(constant_Vrms)+'.png')
elif args.velocity_structure == 'y':
    plt.savefig(output_dir_path + '/imaging_result_y.png')
plt.show()