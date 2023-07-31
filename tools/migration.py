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

print(outputdata.shape)

c = 299792458 # [m/s], 光速
epsilon_1 = 1 # 空気
epsilon_2 = 4 # レゴリス

h = 1.5 # [m], アンテナ高さ
antenna_distance = 0.5 # [m], アンテナ間隔

outputdata_mig = np.zeros(outputdata.shape)

print(outputdata.shape[1]) # x
print(outputdata.shape[0]) # z
# migrate
def migrate(src_step):
    for i in range(outputdata.shape[1]+1): # x
        for j in range(outputdata.shape[0]+1): # z
            for k in range(outputdata.shape[1]): # rx

                x = i * src_step # xの位置
                z = j * dt * c / 2 # zの位置
                x_rx = k * src_step # rxの位置
                x_tx = x_rx + antenna_distance # txの位置



                # d_Rを求める
                d_R_array = np.arange(0.01, 10.01, 0.01)

                d_R_left = np.sqrt(epsilon_2) * d_R_array / np.sqrt(h**2 + d_R_array**2)
                d_R_right = np.sqrt(epsilon_1) \
                    * (np.abs(x - x_rx) - d_R_array) \
                    / np.sqrt(z**2 + (np.abs(x_rx - x) - d_R_array)**2)

                # d_R_leftとd_R_rightの差が最小になるd_Rarrayをd_Rとする
                d_R = np.argmin(np.abs(d_R_left - d_R_right)) / 100 # [m]
                print(d_R)



                # d_Tを求める
                d_T_array = np.arange(0.01, 10.01, 0.01)

                d_T_left = np.sqrt(epsilon_2) * d_T_array / np.sqrt(h**2 + d_T_array**2)
                d_T_right = np.sqrt(epsilon_1) \
                    * (np.abs(x - x_tx) - d_T_array) \
                    / np.sqrt(z**2 + (np.abs(x_tx - x) - d_T_array)**2)

                # d_T_leftとd_T_rightの差が最小になるd_T_arrayをd_Tとする
                d_T = np.argmin(np.abs(d_T_left - d_T_right)) / 100 # [m]
                print(d_T)

                R_1 = np.sqrt(h**2 + d_T**2) # 送信点から地面までの距離
                R_2 = np.sqrt(z**2 + (np.abs(x_rx - x) - d_T)**2) # 地面から(x, z)までの距離
                R_3 = np.sqrt(z**2 + (np.abs(x_rx - x) - d_R)**2) # (x, z)から地面までの距離
                R_4 = np.sqrt(h**2 + d_R**2) # 地面から受信点までの距離

                delta_t = (np.sqrt(epsilon_1)*(R_1 + R_4) + np.sqrt(epsilon_2) * (R_2 + R_3)) / c # 伝搬時間
                print(delta_t)



                recieve_time = 0.1e-8 + delta_t # 伝搬時間
                outputdata_mig[x, z] = np.sum(outputdata[recieve_time, x]) # マイグレーション後のデータ
                # ↑ほんまか？ループのどのタイミングで値を入れるのかわからん

migrate(0.2)