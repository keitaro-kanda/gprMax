import json  # jsonの取り扱いに必要
import os
from calendar import c

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm  # プログレスバーに必要

from tools.outputfiles_merge import get_output_data

# jsonファイルの読み込み
with open ('kanda/domain_10x10/test/B-scan/smooth_2_bi/smooth_2_bi.json') as f:
    params = json.load(f)


# .outファイルの読み込み
file_name = params['input_data']
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()



# 定数の設定
c = 299792458 # [m/s], 光速
epsilon_1 = 1 # 空気
epsilon_2 = 4 # レゴリス


tx_step = params['tx_step'] # [m]
rx_step = params['rx_step'] # [m]
x_resolution = params['x_resolution'] # [m]
spatial_step = params['spatial_step'] # [m]
#antenna_zindex = params['antenna_zindex'] / spatial_step # [m]
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
            x_rx = rx_start + rx * rx_step
            x_tx = tx_start + k * tx_step
        else:
            print("input correct antenna type")
            break

        x = x_index * x_resolution # [m]
        z = z_index * spatial_step # [m]


        # ===Xiao et al.,(2019)の式(5)===

        # d_Rを求める、d_R：電波の地中侵入地点とrxの水平距離
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

    return outputdata_mig




# migration処理関数の実行しまくって地下構造を推定する
def calc_subsurface_structure(rx, tx_step, rx_step, spatial_step):
    for i in tqdm(range(xgrid_num)): # x
        for j in range(zgrid_num): # z

            migration(rx, tx_step, rx_step, spatial_step, i, j)
    
    return outputdata_mig


# 関数の実行
for rx in range(1, nrx + 1):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')
    migration_result = calc_subsurface_structure(rx, tx_step, rx_step, spatial_step)


    # プロット
    plt.figure(figsize=(18, 15), facecolor='w', edgecolor='w')
    plt.imshow(migration_result,
            aspect='auto', cmap='seismic', vmin=-np.amax(outputdata_mig), vmax=np.amax(outputdata_mig))
    plt.colorbar()
    plt.xlabel('Horizontal distance [m]', size=20)
    plt.ylabel('Depth form surface [m]', size=20)
    plt.xticks(np.arange(0, xgrid_num, 5), np.arange(0, xgrid_num*0.2, 1))
    plt.yticks(np.arange(0, zgrid_num, 100), np.arange(0, zgrid_num*0.01, 1))
    plt.title('Migration result rx: ' + str(rx), size=20)

    # plotの保存
    path = os.path.dirname(file_name)
    plt.savefig(path+'/migration_result' + str(rx) + '.png', bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()
