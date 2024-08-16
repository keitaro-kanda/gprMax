import argparse
import json
import os
from cProfile import label

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
from tqdm import tqdm  # プログレスバーに必要

from tools.outputfiles_merge import get_output_data

# ======load files=====
# Parse command line arguments
parser = argparse.ArgumentParser(
    prog = 'k_migration.py',
    description='Processing migration',
    epilog='End of help message',
    usage='python tools/k_migration.py [jsonfile] [file_type] [epsilon_map]')
parser.add_argument('jsonfile', help='json file name')
parser.add_argument('file_type', choices=['raw', 'pulse_comp'], help='file type')
parser.add_argument('epsilon_map', choices=['y', 'n'], help='whether consider about epsilon distribution or not')
parser.add_argument('-all_rx', help='migrate all rx for array configuration', default=False, action='store_true')
parser.add_argument('-closeup', help='closeup option', default=False, action='store_true')
args = parser.parse_args()


# load json file
with open (args.jsonfile) as f:
    params = json.load(f)


# Open output file and read number of outputs (receivers)
file_name_out = params['data']
#file_name_txt = params['input_data_txt']
output_data_out = h5py.File(file_name_out, 'r')
nrx = output_data_out.attrs['nrx']
output_data_out.close()
input_dir_path = os.path.dirname(file_name_out)
# =====load files=====


# =====load epsilon map======
#epsilon_map_path = params['epsilon_map']
#epsilon_map = np.loadtxt(epsilon_map_path)
# =====load epsilon map======


# 定数の設定
c = 299792458 # [m/s], 光速
epsilon_0 = 1 # 真空の誘電率
epsilon_ground_1 = 3 # 地面の誘電率

# =====load jason settings=====
radar_type = 'monostatic' # radar type

#* Load antenna settings
tx_step = params['antenna_settings']['src_step']
rx_step = params['antenna_settings']['rx_step']
tx_start = params['antenna_settings']['src_start']
rx_start = params['antenna_settings']['rx_start']
antenna_height = params['antenna_settings']['antenna_height']
antenna_distance = np.abs(tx_start - rx_start)

x_resolution = 0.005 # [m]
z_resolution = 0.005 # [m]
#antenna_distance = int(params["monostatic_antenna_distance"]) # [m], アンテナ間隔
#array_interval = params['array_interval'] # [m] array antenna distance
#total_trace_num =  params["total_trace_num"] # rxの数

wave_duration_half = params['pulse_info']['pulse_length'] / 2 # [s]
# =====load jason settings=====



#* Geometry settings
geometry_json = params['geometry_settings']['geometry_json']
with open(geometry_json) as f:
    geometry_params = json.load(f)
h5_file_path = geometry_params['geometry_settings']['h5_file']

xgrid_num = int(geometry_params['geometry_settings']['domain_x'] / x_resolution) # x
zgrid_num = int(geometry_params['geometry_settings']['ground_depth'] / z_resolution) # z

antenna_zpoint = antenna_height + geometry_params['geometry_settings']['ground_depth'] # [m]

outputdata_mig = np.zeros([zgrid_num, xgrid_num]) # grid数で定義、[m]じゃないよ！！
delay_time = np.zeros([zgrid_num, xgrid_num]) # grid数で定義、[m]じゃないよ！！


#txt_dir_path = os.path.dirname(file_name_txt)


# migration処理関数の作成
def migration(rx, x_index, z_index, x, z):
    recieve_power_array = np.zeros(xgrid_num) # rxの数だけ0を並べた配列を作成


    #　using epsilon_map version
    def migration_epsilon_map():
    # =====make pass array: reflector -> rx=====
        if radar_type =='bistatic' or 'array':
            x_rx = rx_start + (rx-1) * array_interval
        
        diff_x_ref2rx = np.int(x_rx - x)
        diff_z_ref2rx = np.int(-(antenna_zpoint - z))
        
        if diff_x_ref2rx == 0 and diff_z_ref2rx == 0:
            pass_ref2rx_z = (z * np.ones(1)).astype(int)
            pass_ref2rx_x = (x * np.ones(1)).astype(int)
        elif diff_z_ref2rx == 0:
            pass_ref2rx_z = (z * np.ones(np.abs(diff_x_ref2rx))).astype(int)
            pass_ref2rx_x = (np.arange(x, x_rx, np.sign(diff_x_ref2rx))).astype(int)
        else:
            grad_ref2rx = np.abs(diff_x_ref2rx / diff_z_ref2rx)
            pass_ref2rx_z = z * np.ones(np.abs(diff_z_ref2rx)) \
                - np.sign(diff_z_ref2rx) * np.arange(np.abs(diff_z_ref2rx))
            pass_ref2rx_z = pass_ref2rx_z.astype(int)
            pass_ref2rx_x = x * np.ones(np.abs(diff_z_ref2rx)) \
                + np.sign(diff_x_ref2rx) * grad_ref2rx * np.arange(np.abs(diff_z_ref2rx))
            pass_ref2rx_x = pass_ref2rx_x.astype(int)



        # =====calculate recieved time=====
        L_ref2rx = np.sqrt(np.abs(diff_x_ref2rx)**2 + np.abs(diff_z_ref2rx)**2)
        if diff_z_ref2rx == 0 and diff_x_ref2rx == 0:
            pass_len_ref2rx = 0
        elif diff_z_ref2rx == 0:
            pass_len_ref2rx = L_ref2rx / np.abs(diff_x_ref2rx) * np.sqrt(epsilon_map[pass_ref2rx_z, pass_ref2rx_x])
            pass_len_ref2rx = pass_len_ref2rx.astype(int)
        else:
            pass_len_ref2rx = L_ref2rx / np.abs(diff_z_ref2rx) * np.sqrt(epsilon_map[pass_ref2rx_z, pass_ref2rx_x])
            pass_len_ref2rx = pass_len_ref2rx.astype(int)
        t_ref2rx = np.sum(pass_len_ref2rx) / c


        # for tx
        diff_z_tx2ref = np.int(z - antenna_zpoint)
        pass_tx2ref_z = antenna_zpoint * np.ones(np.abs(diff_z_tx2ref)) \
                - np.sign(diff_z_tx2ref) * np.arange(np.abs(diff_z_tx2ref))
        pass_tx2ref_z = pass_tx2ref_z.astype(int)

        for k in range(total_trace_num):
            # ATTENTION!! この中にはkが絡む計算しか記述しない！！（計算時間削減のため）
            if radar_type == 'monostatic':
                x_rx = k * rx_step + rx_start # rxの位置
                x_tx = x_rx + antenna_distance # txの位置
            elif radar_type =='bistatic' or 'array':
                x_tx = tx_start + k * tx_step
            else:
                print("input correct antenna type")
                break


            # =====make pass array: tx -> reflector=====
            diff_x_tx2ref = np.int(x - x_tx)
            if diff_x_ref2rx == 0 and diff_z_ref2rx == 0:
                pass_tx2ref_z = (z * np.ones(1)).astype(int)
                pass_tx2ref_x = (x * np.ones(1)).astype(int)
            elif diff_z_tx2ref == 0:
                pass_tx2ref_z = (z * np.ones(np.abs(diff_x_tx2ref))).astype(int)
                pass_tx2ref_x = (np.arange(x_tx, x, -1 if np.sign(diff_x_tx2ref) < 0 else 1)).astype(int)
            else:
                grad_tx2ref = np.abs(diff_x_tx2ref / diff_z_tx2ref)
                pass_tx2ref_z = z * np.ones(np.abs(diff_z_tx2ref)) \
                    - np.sign(diff_z_tx2ref) * np.arange(np.abs(diff_z_tx2ref))
                pass_tx2ref_z = pass_tx2ref_z.astype(int)
                pass_tx2ref_x = x_tx * np.ones(np.abs(diff_z_tx2ref)) \
                    + np.sign(diff_x_tx2ref) * grad_tx2ref * np.arange(np.abs(diff_z_tx2ref))
                pass_tx2ref_x = pass_tx2ref_x.astype(int)



            # =====calculate recieved time=====
            L_tx2ref = np.sqrt(np.abs(diff_x_tx2ref)**2 + np.abs(diff_z_tx2ref)**2)
            if diff_z_tx2ref == 0 and diff_x_tx2ref == 0:
                pass_len_tx2ref = 0
            elif diff_z_tx2ref == 0:
                pass_len_tx2ref = L_tx2ref / np.abs(diff_x_tx2ref) * np.sqrt(epsilon_map[pass_tx2ref_z, pass_tx2ref_x])
            else:
                pass_len_tx2ref = L_tx2ref / np.abs(diff_z_tx2ref) * np.sqrt(epsilon_map[pass_tx2ref_z, pass_tx2ref_x])
            t_tx2ref = np.sum(pass_len_tx2ref) / c


            # =====calculate recieved power=====
            recieved_time = t_tx2ref + t_ref2rx + params["wave_start_time"] # [s]
            delay_time[z_index, x_index] = recieved_time

            if recieved_time/dt <= outputdata.shape[0]:
                recieve_power_array[k] = outputdata[int(recieved_time / dt), k]
            else:
                recieve_power_array[k] = 0
        
        return recieve_power_array, delay_time
    
    if args.epsilon_map == 'y':
        migration_epsilon_map()
    

    # old　version
    def migration_mapN():
        if radar_type == 'array':
            x_rx = rx_start + (rx-1) * array_interval
            pass_len_ref2rx = np.sqrt(np.abs(x_rx - x)**2 + np.abs(antenna_zpoint - z)**2 )
        
        for k in range(total_trace_num):
            # ATTENTION!! この中にはkが絡む計算しか記述しない！！（計算時間削減のため）
            if radar_type == 'monostatic':
                x_rx = k * rx_step + rx_start # rxの位置
                x_tx = k * tx_step + tx_start # txの位置
                pass_len_ref2rx = np.sqrt(np.abs(x_rx - x)**2 + np.abs(antenna_zpoint - z)**2 )
            elif radar_type =='bistatic':
                x_rx = k * rx_step + rx_start # rxの位置
                x_tx = k * tx_step + tx_start # txの位置
                pass_len_ref2rx = np.sqrt(np.abs(x_rx - x)**2 + np.abs(antenna_zpoint - z)**2 )
            elif radar_type =='array':
                x_tx = tx_start + k * tx_step
            else:
                print("WARNING!! input correct antenna type")
                break


            if x == x_rx and z == antenna_zpoint:
                recieved_time_k = 0
            
            elif z <= antenna_zpoint: # assume that epsiron_r = 1
                pass_len_tx2ref = np.sqrt(np.abs(x_tx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]
                delay_time = (pass_len_ref2rx + pass_len_tx2ref) / c # [s]
                recieved_time_k = delay_time + params['pulse_info']['transmitting_delay'] # [s]
            

            else: # assume that epsiron_r is that of ground
                pass_len_tx2ref = np.sqrt(np.abs(x_tx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]

                """
                L_vacuum_k = np.sqrt(epsilon_0)*(pass_len_tx2ref + pass_len_ref2rx) * h / np.abs(antenna_zpoint - z)
                L_ground_k = np.sqrt(epsilon_ground_1)*(pass_len_tx2ref + pass_len_ref2rx) * np.abs(antenna_zpoint - z - h) / np.abs(antenna_zpoint - z)
                recieved_time_k = (L_vacuum_k + L_ground_k) / c # [s]
                """

                delta_t = np.sqrt(epsilon_ground_1) * (pass_len_tx2ref + pass_len_ref2rx) / c
                recieved_time_k = delta_t + params['pulse_info']['transmitting_delay'] # [s]
                
            
            t_index_start = int((recieved_time_k - wave_duration_half) / dt)
            t_index_end = int((recieved_time_k + wave_duration_half) / dt)

            if recieved_time_k/dt <= outputdata.shape[0]:
                recieve_power_array[k] = outputdata[int(recieved_time_k / dt), k]
            else:
                recieve_power_array[k] = 0
            """"
            if 0 <= t_index_start and t_index_end <= outputdata.shape[0]:
                recieved_power_list = outputdata[t_index_start:t_index_end, k]
                max_power_index = np.argmax(np.abs(recieved_power_list))
                recieve_power_array[k] = recieved_power_list[max_power_index]
            """

            
        return recieve_power_array
    
    if args.epsilon_map == 'n':
        migration_mapN()
    
    
    # recieve_power_arrayの要素の和をとる
    outputdata_mig[z_index, x_index] = np.sum(recieve_power_array)
    return outputdata_mig, delay_time


# migration処理関数の実行しまくって地下構造を推定する
def calc_subsurface_structure(rx):
    
    for i in tqdm(range(xgrid_num), desc="rx" + str(rx)): # x
        ref_x = i * x_resolution # [m]
        for j in range(zgrid_num): # z
            ref_z = j * z_resolution # [m]
            migration_result , delay_time_array = migration(rx, i, j, ref_x, ref_z)

    return migration_result, delay_time_array



# =====rxの指定=====
if radar_type == 'monostatic':
    rx_num_start = 1
elif radar_type == 'bistatic':
    rx_num_start = 1
elif radar_type == 'array':
    rx_num_start = 25
else:
    print('input correct radar type')
    exit()


print('rx_num_start:', rx_num_start)
rx_num_end =  rx_num_start + 1

# -select_rx用の用の手動設定
if args.all_rx == True:
    rx_num_start = 1
    rx_num_end = nrx + 1
# ==================


# make output directory path
if args.file_type == 'raw':
    if args.epsilon_map == 'y':
        output_dir_path = os.path.join(input_dir_path, 'migration_raw_mapY')
    elif args.epsilon_map == 'n':
        output_dir_path = os.path.join(input_dir_path, 'migration_raw_mapN')
elif args.file_type == 'pulse_comp':
    if args.epsilon_map == 'y':
        output_dir_path = os.path.join(input_dir_path, 'migration_plscomp_mapY')
    elif args.epsilon_map == 'n':
        output_dir_path = os.path.join(input_dir_path, 'migration_plscomp_mapN')

if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)

for rx in range(rx_num_start, rx_num_end):
    # from raw file
    if args.file_type == 'raw':
        outputdata, dt = get_output_data(file_name_out, rx, 'Ez') 
        total_trace_num = outputdata.shape[1]

        migration_calc, time_calc = calc_subsurface_structure(rx)
        migration_result_standardize = migration_calc / np.amax(migration_calc) * 100


    # from pulse_comp file
    elif args.file_type == 'pulse_comp':
        load_txt_name = os.path.join(txt_dir_path, 'corr_data_rx' + str(rx) + '.txt')
        outputdata = np.loadtxt(load_txt_name)

        no_use, dt = get_output_data(file_name_out, rx, 'Ez') # dtを取り出すため必要
        migration_calc, time_calc = calc_subsurface_structure(rx)
        
        migration_result_standardize = np.zeros_like(migration_calc)
        for i in tqdm(range(xgrid_num), desc = 'calculate power'):
            for j in range(zgrid_num):
                if migration_calc[j, i] == 0:
                    migration_result_standardize[j, i] = 10 * np.log10(1e-10 / np.amax(migration_calc))
                else:
                    migration_result_standardize[j, i] = 10 * \
                        np.log10(np.abs(migration_calc[j, i]) / np.abs(np.amax(migration_calc)))


    # =====save migration_result txt file=====
    np.savetxt(output_dir_path + '/migration_result_rx' + str(rx) + '.txt', migration_result_standardize)
    
    


    # =====plot=====
    fig = plt.figure(figsize=(10, 7), facecolor='w', edgecolor='w')
    ax = fig.add_subplot(211)
    
    if args.file_type == 'raw':
        plt.imshow(migration_result_standardize,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                aspect=z_resolution/x_resolution, cmap='seismic', vmin=-5, vmax=5)
        
        delvider = axgrid1.make_axes_locatable(ax)
        cax = delvider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(cax=cax, label = 'Ex intensity [%]')


    elif args.file_type == 'pulse_comp':
        plt.imshow(migration_result_standardize,
                extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
                cmap='rainbow', vmin=-50, vmax=0)
    
        delvider = axgrid1.make_axes_locatable(ax)
        cax = delvider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(cax=cax, label = 'power [dB]')

    ax.set_xlabel('Horizontal distance [m]', size=14)
    ax.set_ylabel('Depth form surface [m]', size=14)

    ax.set_title('Migration result rx' + str(rx), size=18)


    # closeup option
    if args.closeup == True:
        ax.set_xlim(200, 350)
        ax.set_ylim(300, 150)


    if args.file_type == 'raw':
        edge_color = 'gray'
    elif args.file_type == 'pulse_comp':
        edge_color = 'white'


    """"
    ax = fig.add_subplot(212)
    plt.imshow(time_calc,
            extent=[0, xgrid_num*x_resolution, zgrid_num*z_resolution, 0],
            aspect=z_resolution/x_resolution, cmap='rainbow', vmin=0, vmax=4e-6)
    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(cax=cax)


    ax.set_xlabel('Horizontal distance [m]', size=20)
    ax.set_ylabel('Depth form surface [m]', size=20)
    ax.set_title('Delay time', size=20)
    """


    # =====seve plot=====
    if args.closeup == False:
        plt.savefig(output_dir_path + '/migration_result' + str(rx) + '.png', bbox_inches='tight', dpi=300)
    if args.closeup == True:
        plt.savefig(output_dir_path + '/migration_result' + str(rx) + '_closeup.png', bbox_inches='tight', dpi=300)

    
    plt.show()