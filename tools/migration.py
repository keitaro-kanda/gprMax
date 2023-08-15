import argparse
import json  # jsonの取り扱いに必要
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # プログレスバーに必要

from tools.outputfiles_merge import get_output_data

# ======load files=====
# Parse command line arguments
parser = argparse.ArgumentParser(description='Processing migration', 
                                 usage='cd gprMax; python -m tools.migration jsonfile')
parser.add_argument('jsonfile', help='json file name')
args = parser.parse_args()

# load json file
with open (args.jsonfile) as f:
    params = json.load(f)

# Open output file and read number of outputs (receivers)
file_name = params['input_data']
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()
input_dir_path = os.path.dirname(file_name)
# =====load files=====

# make output directory
output_dir_path = os.path.join(input_dir_path, 'migration')
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)



# 定数の設定
c = 299792458 # [m/s], 光速
epsilon_0 = 1 # 空気
epsilon_ground_1 = params['epsilon_ground_1'] # レゴリス


tx_step = params['tx_step'] # [m]
rx_step = params['rx_step'] # [m]
x_resolution = params['x_resolution'] # [m]
spatial_step = params['spatial_step'] # [m]
antenna_zpoint = params['antenna_zpoint'] # [m]
h = params['antenna_hight'] # [m], アンテナの高さ
antenna_distance = params["monostatic_antenna_distance"]# [m], アンテナ間隔

outputdata_mig = np.zeros([params['geometry_matrix_axis0'], 
                           params['geometry_matrix_axis1']]) # grid数で定義、[m]じゃないよ！！

xgrid_num = outputdata_mig.shape[1] # x
zgrid_num = outputdata_mig.shape[0] # z



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
        Lt_k = np.sqrt(np.abs(x_tx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]
        Lr = np.sqrt(np.abs(x_rx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]

        L_vacuum_k = np.sqrt(epsilon_0)*(Lt_k + Lr) * h / np.abs(antenna_zpoint - z)
        L_ground_k = np.sqrt(epsilon_ground_1)*(Lt_k + Lr) * np.abs(antenna_zpoint - z - h) / np.abs(antenna_zpoint - z)

        delta_t_k = (L_vacuum_k + L_ground_k) / c # [s]
        recieved_time_k = delta_t_k + params["wave_start_time"] # [s]
        recieve_power_array[k] = outputdata[int(recieved_time_k / dt), k]

        """ correration
        for l in range(k+1, total_trace_num):
            # trace l
            x_tx_l = x_tx + (l-k) * x_resolution
            Lt_l = np.sqrt(np.abs(x_tx_l - x)**2 + (h + z)**2 ) # [m]

            L_vacuum_l = np.sqrt(epsilon_1)*(Lt_l + Lr) * h / (z + h)   
            L_ground_l = np.sqrt(epsilon_2)*(Lt_l + Lr) * z / (z + h) 

            delta_t_l = (L_vacuum_l + L_ground_l) / c # [s]
            recieved_time_l = delta_t_l + params["wave_start_time"] # [s]

            #recieve_power_array[k] = outputdata[int(recieved_time / dt), k]
            l_array = np.zeros(total_trace_num-k)
            l_array[l-k-1] = (Lt_k + Lr) * outputdata[int(recieved_time_k / dt), k] \
                * (Lt_l + Lr) * outputdata[int(recieved_time_l / dt), l]
            recieve_power_array[k] = np.sum(l_array)
        """
    
    # recieve_power_arrayの要素の和をとる
    outputdata_mig[z_index, x_index] = np.sum(recieve_power_array)
    return outputdata_mig

"""
        # ===Xiao et al.,(2019)の式(5)===

        # d_Rを求める、d_R: 電波の地中侵入地点とrxの水平距離
        d_R_array = np.arange(0, xgrid_num*x_resolution, spatial_step) # 間隔は空間ステップにしたがう

        # (x, z)が地表面にいる場合、ゼロ徐算を避ける
        if z == 0:
            #d_R = np.sqrt((x_rx - x)**2 + h**2)
            d_R = np.sqrt(((x_tx - x_rx)/2)**2 + h**2)
        else:
            d_R_left = np.sqrt(epsilon_2) * d_R_array / np.sqrt(h**2 + d_R_array**2)
            d_R_right = np.sqrt(epsilon_1) \
                * (np.abs(x - x_rx) - d_R_array) \
                / np.sqrt(z**2 + (np.abs(x_rx - x) - d_R_array)**2)

            # d_R_leftとd_R_rightの差が最小になるd_Rarrayをd_Rとする
            d_R = np.argmin(np.abs(d_R_left - d_R_right)) / 100 # [m] 
            #print(d_R) 



        # d_Tを求める、d_T：電波の地中侵入地点とtxの水平距離
        d_T_array = np.arange(0, xgrid_num*x_resolution, spatial_step)

        # (x, z)が地表面にいる場合、ゼロ徐算を避ける
        if z == 0:
            #d_T = np.sqrt((x_tx - x)**2 + h**2)
            d_T = d_R
        else:
            d_T_left = np.sqrt(epsilon_2) * d_T_array / np.sqrt(h**2 + d_T_array**2)
            d_T_right = np.sqrt(epsilon_1) \
                * (np.abs(x - x_tx) - d_T_array) \
                / np.sqrt(z**2 + (np.abs(x_tx - x) - d_T_array)**2)

            # d_T_leftとd_T_rightの差が最小になるd_T_arrayをd_Tとする
            d_T = np.argmin(np.abs(d_T_left - d_T_right)) / 100 # [m]
            #print(d_T)

        
        # ===Xiao et al.,(2019)の式(4)===
        if z == 0:
            R_1 = d_T # 送信点から地面までの距離
            R_2 = 0 # 地面から(x, z)までの距離
            R_3 = 0 # (x, z)から地面までの距離
            R_4 = d_R # 地面から受信点までの距離

        else:
            R_1 = np.sqrt(h**2 + d_T**2) # 送信点から地面までの距離
            R_2 = np.sqrt(z**2 + (np.abs(x_tx - x) - d_T)**2) # 地面から(x, z)までの距離
            R_3 = np.sqrt(z**2 + (np.abs(x_rx - x) - d_R)**2) # (x, z)から地面までの距離
            R_4 = np.sqrt(h**2 + d_R**2) # 地面から受信点までの距離


        # ===伝搬時間、到来時間の計算===
        total_propagating_distance = (np.sqrt(epsilon_1)*(R_1 + R_4) + np.sqrt(epsilon_2) * (R_2 + R_3))
        delta_t = total_propagating_distance / c # 伝搬時間
        recieve_time = 0.1e-8 + delta_t # 伝搬時間


        # ===それぞれのアンテナ位置rxに対し、位置(x, z)における反射強度を保存===
        recieve_power_array[k] = outputdata[int(recieve_time / dt), k] \
        #* (4 * np.pi)**2 * (R_1 + epsilon_2*R_2)**2 * (R_4 + epsilon_2*R_3)**2 # 受信点の電力を配列に格納
        
    # recieve_power_arrayの要素の和をとる
    outputdata_mig[z_index, x_index] = np.sum(recieve_power_array)
"""




# migration処理関数の実行しまくって地下構造を推定する
def calc_subsurface_structure(rx, tx_step, rx_step, spatial_step):
    for i in tqdm(range(xgrid_num), desc="rx" + str(rx)): # x
        for j in range(zgrid_num): # z

            migration(rx, tx_step, rx_step, spatial_step, i, j)
    
    return outputdata_mig


# 関数の実行
#for rx in range(1, nrx + 1):
for rx in range(1, 32):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')
    migration_result = calc_subsurface_structure(rx, tx_step, rx_step, spatial_step)
    # migration_resultをtxtファイルに保存
    np.savetxt(output_dir_path+'/migration_result_rx' + str(rx) + '.txt', migration_result)


    # プロット
    migration_result_percent = migration_result / np.amax(migration_result) * 100
    plt.figure(figsize=(18, 15), facecolor='w', edgecolor='w')
    plt.imshow(migration_result_percent,
            aspect='auto', cmap='seismic', vmin=-10, vmax=10)
    plt.colorbar()
    plt.xlabel('Horizontal distance [m]', size=20)
    plt.ylabel('Depth form surface [m]', size=20)
    plt.xticks(np.arange(0, xgrid_num, 5), np.arange(0, xgrid_num*0.2, 1))
    plt.yticks(np.arange(0, zgrid_num, 100), np.arange(0, zgrid_num*0.01, 1))
    plt.title('Migration result rx' + str(rx), size=20)

    # plotの保存
    plt.savefig(output_dir_path+'/migration_result' + str(rx) + '.png', bbox_inches='tight', dpi=300)

    if params['monostatic'] == "yes" or params['bistatic'] == "yes":
        plt.show()