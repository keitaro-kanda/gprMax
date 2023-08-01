import re
from calendar import c

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

# 読み込みファイル名
file_name = 'kanda/domain_10x10/test/B-scan/smooth/test_B_merged.out'

# .outファイルの読み込み
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()

for rx in range(1, nrx + 1):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')


c = 299792458 # [m/s], 光速
epsilon_1 = 1 # 空気
epsilon_2 = 4 # レゴリス

h = 1.5 # [m], アンテナ高さ
antenna_distance = 0.5 # [m], アンテナ間隔

outputdata_mig = np.zeros(outputdata.shape)

xgrid_num = outputdata.shape[1] # x
zgrid_num = outputdata.shape[0] # z


# migration処理関数の作成
def migration(src_step, x, z):
    recieve_power_array = np.zeros(xgrid_num) # rxの数だけ0を並べた配列を作成

    for k in range(outputdata.shape[1]): # rx

        #x = i * src_step # xの位置
        #z = j * dt * c / 2 # zの位置
        x_rx = k * src_step + 1.75 # rxの位置、1.75 mは初期位置
        x_tx = x_rx + antenna_distance # txの位置


        # ===Xiao et al.,(2019)の式(5)===
        # d_Rを求める、d_R：電波の地中侵入地点とrxの水平距離
        d_R_array = np.arange(0.01, 10.01, 0.01) # 間隔は空間ステップにしたがう

        # (x, z)が地表面にいる場合、ゼロ徐算を避ける
        if z == 0:
            d_R = np.sqrt((x_rx - x)**2 + h**2)
        else:
            d_R_left = np.sqrt(epsilon_2) * d_R_array / np.sqrt(h**2 + d_R_array**2)
            d_R_right = np.sqrt(epsilon_1) \
                * (np.abs(x - x_rx) - d_R_array) \
                / np.sqrt(z**2 + (np.abs(x_rx - x) - d_R_array)**2)

            # d_R_leftとd_R_rightの差が最小になるd_Rarrayをd_Rとする
            d_R = np.argmin(np.abs(d_R_left - d_R_right)) / 100 # [m]
            #print(d_R) 



        # d_Tを求める、d_T：電波の地中侵入地点とtxの水平距離
        d_T_array = np.arange(0.01, 10.01, 0.01)

        # (x, z)が地表面にいる場合、ゼロ徐算を避ける
        if z == 0:
            d_T = np.sqrt((x_tx - x)**2 + h**2)
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
            R_1 = np.sqrt(h**2 + d_T**2) # 送信点から地面までの距離
            R_2 = 0 # 地面から(x, z)までの距離
            R_3 = 0 # (x, z)から地面までの距離
            R_4 = np.sqrt(h**2 + d_R**2) # 地面から受信点までの距離

        else:
            R_1 = np.sqrt(h**2 + d_T**2) # 送信点から地面までの距離
            R_2 = np.sqrt(z**2 + (np.abs(x_rx - x) - d_T)**2) # 地面から(x, z)までの距離
            R_3 = np.sqrt(z**2 + (np.abs(x_rx - x) - d_R)**2) # (x, z)から地面までの距離
            R_4 = np.sqrt(h**2 + d_R**2) # 地面から受信点までの距離


        # ===伝搬時間、到来時間の計算===
        delta_t = (np.sqrt(epsilon_1)*(R_1 + R_4) + np.sqrt(epsilon_2) * (R_2 + R_3)) / c # 伝搬時間
        recieve_time = 0.1e-8 + delta_t # 伝搬時間


        # ===それぞれのアンテナ位置rxに対し、位置(x, z)における反射強度を保存===
        recieve_power_array[k] = outputdata[int(recieve_time / dt), k] # 受信点の電力を配列に格納
        
    # recieve_power_arrayの要素の和をとる
    outputdata_mig[x, z] = np.sum(recieve_power_array)

    return outputdata_mig



# migration処理関数の実行しまくって地下構造を推定する
def calc_subsurface_structure(src_step):
    for i in range(xgrid_num): # x
        for j in range(zgrid_num): # z
            migration(src_step, i, j)

    return migration(src_step, i, j)


migration_result = calc_subsurface_structure(0.2)
print(migration_result.shape)
print(migration_result)

plt.imshow(migration_result, aspect='auto', cmap='seismic', vmin=-0.1, vmax=0.1)
plt.show( )

