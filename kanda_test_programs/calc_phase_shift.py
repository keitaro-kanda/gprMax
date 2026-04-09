"""
全反射における位相シフトの量を計算するプログラム。
参考：Balanis (2012), Advanced Engineering Electromagnetics Second Edition
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy import signal


# パラメータ設定
epsilon_1 = 9.0 # 入射・反射側の媒質
epsilon_2 = 3.0 # 透過側の媒質
mu = 1.25663706127e-6 # 真空の透磁率、両媒質で同じと仮定
theta_i = np.linspace(0, 90, 1000) # 入射角 [度]
theta_i_rad = np.radians(theta_i) # 入射角 [ラジアン]


# 臨界角の計算
theta_c_rad = np.arcsin(np.sqrt(epsilon_2 / epsilon_1))
theta_c_deg = np.degrees(theta_c_rad)
print(f"Critical Angle: {theta_c_deg:.2f} degrees")

# 位相シフトの計算
X = np.sqrt(mu / epsilon_1) * np.sqrt(mu * epsilon_1 / mu / epsilon_2 * np.sin(theta_i_rad)**2 - 1)
R = np.sqrt(mu / epsilon_2) * np.cos(theta_i_rad)
pahse_shift = np.arctan2(X, R) # 位相シフト [ラジアン]



# グラフの描画
plt.figure(figsize=(8, 6))
plt.plot(theta_i, np.degrees(pahse_shift), label='Phase Shift')
plt.axvline(theta_c_deg, color='red', linestyle='--', label=r'Critical angle $\theta_c$: ' + f'{theta_c_deg:.2f}°')

plt.xlabel(r'Incident angle $\theta_i$', fontsize=16)
plt.ylabel(r'Phase shift $2 \psi_{\perp}$', fontsize=16)
plt.tick_params(labelsize=14)
plt.grid()
plt.legend(fontsize=14)

output_dir = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/phase_shift'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(output_dir + '/phase_shift.png', dpi=120)
plt.savefig(output_dir + '/phase_shift.pdf', dpi=300)
plt.show()


# FDTDで生成した波形に位相シフトの効果を加える
signal_data_path = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/waveform_test/gaussiandot_500MHz/A-scan/direct.out'

# Open output file and read some attributes
f = h5py.File(signal_data_path, 'r')
nrx = f.attrs['nrx']
dt = f.attrs['dt']

path = '/rxs/rx' + str(1) + '/'
availableoutputs = list(f[path].keys())

outputdata = f[path + '/Ez']
outputdata = np.array(outputdata)
outputdata_norm = outputdata / np.amax(np.abs(outputdata))

# データの切り出し
threshold = 0.01
indices = np.where(np.abs(outputdata_norm) >= threshold)[0]
if len(indices) > 0:
    start_index = indices[0]
    end_index = indices[-1] + 1
    outputdata = outputdata_norm[start_index:end_index]
else:
    print("Warning: No data points above the threshold. Using the entire signal.")

# 元信号で最大振幅の時刻を取得
original_max_index = np.argmax(np.abs(outputdata))
original_max_time = original_max_index * dt * 1e9 # [ns]

# 解析信号の作成
analytic_signal = signal.hilbert(outputdata)

# 位相シフトを加える
incident_angles = [40, 45, 60, 75, 90] # 入射角の例
shifted_signals = []
for angle in incident_angles:
    theta_i_rad = np.radians(angle)
    
    # 臨界角の判定用（ルートの中身）
    sin_theta_t_sq = (epsilon_1 / epsilon_2) * np.sin(theta_i_rad)**2
    
    if sin_theta_t_sq < 1.0:
        # 臨界角未満の場合（全反射せず、位相シフトは生じない）
        # ※インピーダンスの大小による π の反転は考慮せず、そのままの波形とする場合
        shifted_signals.append(outputdata)
        continue

    X = np.sqrt(mu / epsilon_1) * np.sqrt(sin_theta_t_sq - 1)
    R = np.sqrt(mu / epsilon_2) * np.cos(theta_i_rad)
    
    # 2. 位相シフト量の計算 (Balanisの psi を2倍にする)
    psi = np.arctan2(X, R)
    total_phase_shift = 2 * psi

    # 3. 解析信号に位相シフトを適用し、実部をとる
    shifted_wave = np.real(analytic_signal * np.exp(1j * total_phase_shift))
    shifted_signals.append(shifted_wave)

shifted_signals.insert(0, outputdata) # 元の信号も追加
plot_names = ['Original'] + [f'Incident Angle: {angle}°' for angle in incident_angles]

# グラフの描画
fig, ax = plt.subplots(len(incident_angles) + 1, 1, figsize=(6, 12))
for i in range(len(plot_names)):
    if i == 0:
        ax[i].plot(np.arange(len(outputdata)) * dt * 1e9, shifted_signals[i], color = 'k')
    else:
        ax[i].plot(np.arange(len(outputdata)) * dt * 1e9, np.real(shifted_signals[i]), color = 'cyan')
        ax[i].plot(np.arange(len(outputdata)) * dt * 1e9, np.real(shifted_signals[i]) + np.real(shifted_signals[0]), color = 'magenta')
        ax[i].plot(np.arange(len(outputdata)) * dt * 1e9, shifted_signals[0], label='Original Signal', color = 'gray', linestyle='--')
    ax[i].axvline(original_max_time, color='r', linestyle='--')

    ax[i].set_title(plot_names[i], fontsize=16)
    ax[i].tick_params(labelsize=12)
    ax[i].grid()
    ax[i].set_xlabel('Time (ns)', fontsize=14)
    ax[i].set_ylabel('Amplitude', fontsize=14)
    ax[i].set_ylim(-2.1, 2.1)
#plt.ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig(output_dir + '/shifted_signals.png', dpi=120)
plt.savefig(output_dir + '/shifted_signals.pdf', dpi=300)
plt.show()