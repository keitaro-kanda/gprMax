import argparse
import json  # jsonの取り扱いに必要
import os

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from tqdm import tqdm  # プログレスバーに必要

from tools.outputfiles_merge import get_output_data

# ======load files=====
# Parse command line arguments
parser = argparse.ArgumentParser(description='Processing migration', 
                                 usage='cd gprMax; python -m tools.migration jsonfile file_type')
parser.add_argument('jsonfile', help='json file name')
parser.add_argument('file_type', choices=['out', 'txt'], help='file type')
parser.add_argument('-select_rx', help='select specific rx number from array', default=False, action='store_true')
args = parser.parse_args()

# load json file
with open (args.jsonfile) as f:
    params = json.load(f)

# Open output file and read number of outputs (receivers)
file_name_out = params['input_data_out']
file_name_txt = params['input_data_txt']
output_data_out = h5py.File(file_name_out, 'r')
nrx = output_data_out.attrs['nrx']
output_data_out.close()
input_dir_path = os.path.dirname(file_name_out)
# =====load files=====



# 定数の設定
c = 299792458 # [m/s], 光速
epsilon_0 = 1 # 真空の誘電率
epsilon_ground_1 = params['epsilon_ground_1'] 


tx_step = params['tx_step'] # [m]
rx_step = params['rx_step'] # [m]
x_resolution = params['x_resolution'] # [m]
z_resolution = params['z_resolution'] # [m]
antenna_zpoint = params['antenna_zpoint'] # [m]
h = params['antenna_hight'] # [m], アンテナの高さ
antenna_distance = params["monostatic_antenna_distance"]# [m], アンテナ間隔



xgrid_num = int(params['geometry_matrix_axis1'] / x_resolution) # x
zgrid_num = int(params['geometry_matrix_axis0'] / z_resolution) # z

outputdata_mig = np.zeros([zgrid_num, xgrid_num]) # grid数で定義、[m]じゃないよ！！



# migration処理関数の作成
def migration(rx, tx_step, rx_step, spatial_step, x_index, z_index):
    recieve_power_array = np.zeros(xgrid_num) # rxの数だけ0を並べた配列を作成
    total_trace_num =  params["total_trace_num"] # rxの数
    tx_start = params["tx_start"] # txの初期位置
    rx_start = params["rx_start"] # rxの初期位置

    for k in range(total_trace_num): 
        if params['monostatic'] == "yes" and params['bistatic'] == "no" and params['array'] == "no":
            x_rx = k * rx_step + rx_start # rxの位置
            x_tx = x_rx + antenna_distance # txの位置
        elif params['bistatic'] == "yes" and params['monostatic'] == "no" and params['array'] == "no":
            x_rx = rx_start + k * rx_step
            x_tx = tx_start + k * tx_step
        elif params['array'] == "yes" and params['monostatic'] == "no" and params['bistatic'] == "no":
            x_rx = rx_start + (rx-1) * x_resolution
            x_tx = tx_start + k * tx_step
        else:
            print("input correct antenna type")
            break

        x = x_index * x_resolution # [m]
        z = z_index * spatial_step # [m]

        # trace k
        Lr = np.sqrt(np.abs(x_rx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]
        if x == x_rx and z == antenna_zpoint:
            recieved_time_k = 0
        elif z <= antenna_zpoint:
            Lt_k = np.sqrt(np.abs(x_tx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]
            delta_t_k = (Lr + Lt_k) / c # [s]
            recieved_time_k = delta_t_k + params["wave_start_time"] # [s]
        else:
            Lt_k = np.sqrt(np.abs(x_tx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]

            L_vacuum_k = np.sqrt(epsilon_0)*(Lt_k + Lr) * h / np.abs(antenna_zpoint - z)
            L_ground_k = np.sqrt(epsilon_ground_1)*(Lt_k + Lr) * np.abs(antenna_zpoint - z - h) / np.abs(antenna_zpoint - z)

            delta_t_k = (L_vacuum_k + L_ground_k) / c # [s]
            recieved_time_k = delta_t_k + params["wave_start_time"] # [s]
        
        if recieved_time_k/dt <= outputdata.shape[0]:
            recieve_power_array[k] = outputdata[int(recieved_time_k / dt), k]
        else:
            recieve_power_array[k] = 0

    
    # recieve_power_arrayの要素の和をとる
    outputdata_mig[z_index, x_index] = np.sum(recieve_power_array)
    return outputdata_mig


# migration処理関数の実行しまくって地下構造を推定する
def calc_subsurface_structure(rx, tx_step, rx_step, spatial_step):
    for i in tqdm(range(xgrid_num), desc="rx" + str(rx)): # x
        for j in range(zgrid_num): # z

            migration(rx, tx_step, rx_step, spatial_step, i, j)
    
    return outputdata_mig



# =====rxの指定=====
rx_num_start =  1
rx_num_end =  nrx + 1

# -select_rx用の用の手動設定
if args.select_rx == True:
    rx_num_start = 25
    rx_num_end = rx_num_start + 1
# ==================

txt_dir_path = os.path.dirname(file_name_txt)

# =====make output directory=====
output_dir_path_out = os.path.join(input_dir_path, 'migration_out')
if not os.path.exists(output_dir_path_out):
    os.mkdir(output_dir_path_out)
output_dir_path_txt = os.path.join(input_dir_path, 'migration_txt')
if not os.path.exists(output_dir_path_txt):
    os.mkdir(output_dir_path_txt)
# ==============================


for rx in range(rx_num_start, rx_num_end):
    # from .out file
    if args.file_type == 'out':
        outputdata, dt = get_output_data(file_name_out, rx, 'Ez') 

        migration_result = calc_subsurface_structure(rx, tx_step, rx_step, z_resolution)
        migration_result_standardize = migration_result / np.amax(migration_result) * 100
        np.savetxt(output_dir_path_out+'/migration_result_rx' + str(rx) + '.txt', migration_result_standardize)

    # from .txt file
    elif args.file_type == 'txt':
        load_txt_name = os.path.join(txt_dir_path, 'corr_data_rx' + str(rx) + '.txt')
        outputdata = np.loadtxt(load_txt_name)

        no_use, dt = get_output_data(file_name_out, rx, 'Ez') # dtを取り出すため必要
        migration_result = calc_subsurface_structure(rx, tx_step, rx_step, z_resolution)
        
        migration_result_standardize = np.zeros_like(migration_result)
        for i in range(xgrid_num):
            for j in range(zgrid_num):
                if migration_result[j, i] == 0:
                    migration_result_standardize[j, i] = 10 * np.log10(1e-10 / np.amax(migration_result))
                else:
                    migration_result_standardize[j, i] = 10 * \
                        np.log10(np.abs(migration_result[j, i]) / np.abs(np.amax(migration_result)))

        np.savetxt(output_dir_path_txt+'/migration_result_rx' + str(rx) + '.txt', migration_result_standardize)
    
    
    # =====plot=====
    fig = plt.figure(figsize=(13, 14), facecolor='w', edgecolor='w')
    ax = fig.add_subplot(211)
    if args.file_type == 'out':
        plt.imshow(migration_result_standardize,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                aspect=z_resolution/x_resolution, cmap='seismic', vmin=-10, vmax=10)
    elif args.file_type == 'txt':
        plt.imshow(migration_result_standardize,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                cmap='rainbow', vmin=-40, vmax=0)
    
    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax)

    ax.set_xlabel('Horizontal distance [m]', size=20)
    ax.set_ylabel('Depth form surface [m]', size=20)
    #ax.set_xticks(np.linspace(0, xgrid_num, 10), np.linspace(0, xgrid_num*x_resolution, 10))
    #ax.set_yticks(np.linspace(0, zgrid_num, 10), np.linspace(0, zgrid_num*spatial_step, 11))
    ax.set_title('Migration result rx' + str(rx), size=20)


    ax = fig.add_subplot(212)
    if args.file_type == 'out':
        plt.imshow(migration_result_standardize,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                aspect=z_resolution/x_resolution, cmap='seismic', vmin=-10, vmax=10)
    elif args.file_type == 'txt':
        plt.imshow(migration_result_standardize,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                cmap='rainbow', vmin=-40, vmax=0)
    
    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax)

    ax.set_xlabel('Horizontal distance [m]', size=20)
    ax.set_ylabel('Depth form surface [m]', size=20)
    #ax.set_xticks(np.linspace(0, xgrid_num, 10), np.linspace(0, xgrid_num*x_resolution, 10))
    #ax.set_yticks(np.linspace(0, zgrid_num, 10), np.linspace(0, zgrid_num*spatial_step, 11))

    if args.file_type == 'out':
        edge_color = 'gray'
    elif args.file_type == 'txt':
        edge_color = 'white'

    ax.set_title('Migration result rx' + str(rx), size=20)
    # 地形のプロット
    rille_apex_list = [(0, 10), (25, 10), 
                    (125, 260), (425, 260),
                    (525, 10), (550, 10)]
    rille = patches.Polygon(rille_apex_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
    ax.add_patch(rille)

    surface_hole_tube_list = [(35, 35), (250, 35),
                            (250, 60), (175, 60),
                            (175, 77), (375, 77),
                            (375, 60), (300, 60),
                            (300, 35), (515, 35)]
    tube = patches.Polygon(surface_hole_tube_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
    ax.add_patch(tube)


    # plotの保存
    if args.file_type == 'out':
        plt.savefig(output_dir_path_out+'/migration_result' + str(rx) + '.png', bbox_inches='tight', dpi=300)
    elif args.file_type == 'txt':
        plt.savefig(output_dir_path_txt+'/migration_result' + str(rx) + '.png', bbox_inches='tight', dpi=300)

    if params['monostatic'] == "yes" or params['bistatic'] == "yes":
        plt.show()