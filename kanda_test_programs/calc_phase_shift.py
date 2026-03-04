"""
全反射における位相シフトの量を計算するプログラム。
参考：Balanis (2012), Advanced Engineering Electromagnetics Second Edition
"""

import numpy as np
import matplotlib.pyplot as plt
import os


epsilon_1 = 9.0 # 入射・反射側の媒質
epsilon_2 = 3.0 # 透過側の媒質
mu = 1.0 # 真空の透磁率、両媒質で同じと仮定
theta_i = np.linspace(0, 90, 1000) # 入射角 [度]
theta_i_rad = np.radians(theta_i) # 入射角 [ラジアン]

X = np.sqrt(mu / epsilon_1) * np.sqrt(mu * epsilon_1 / mu / epsilon_2 * np.sin(theta_i_rad)**2 - 1)
R = np.sqrt(mu / epsilon_2) * np.cos(theta_i_rad)
pahse_shift = np.arctan2(X, R) # 位相シフト [ラジアン]



# グラフの描画
plt.figure(figsize=(8, 6))
plt.plot(theta_i, np.degrees(pahse_shift))

plt.xlabel('Incident Angle (degrees)', fontsize=16)
plt.ylabel('Phase Shift (degrees)', fontsize=16)
plt.tick_params(labelsize=14)
plt.grid()

output_dir = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/phase_shift'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(output_dir + '/phase_shift.png', dpi=120)
plt.savefig(output_dir + '/phase_shift.pdf', dpi=300)
plt.show()