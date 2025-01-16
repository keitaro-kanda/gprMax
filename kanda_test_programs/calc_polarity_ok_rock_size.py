import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from tqdm.contrib import tenumerate


# Physical constants
c = 299792458  # Speed of light in vacuum [m/s]


#* parameters
antenna_height = 0.3  # [m]
rock_depth = 2.0  # [m]
rock_heights = np.arange(0.0, 6.01, 0.05)  # [m]
rock_widths = np.arange(0.0, 6.01, 0.05)  # [m]
thetas = np.arange(0, np.pi * 1/2, np.pi / 1440)  # [rad], 0 ~ pi/2
er_regolith = 3.0
er_rock = 9.0
FWHM = 1.56e-9  # [s], 1.56 ns
FWHM_2GHz = 0.4e-9  # [s], 0.4 ns




#* Calculation
def calc(h_index, h, fwhm):
    w_theta_matrix = np.zeros((len(rock_widths), len(thetas)))
    w_theta_TorF = np.zeros((len(rock_widths), len(thetas)))

    #* Bottom component
    L_bottom = 2 * (antenna_height + rock_depth * np.sqrt(er_regolith) + h * np.sqrt(er_rock))  # [m]
    T_bottom = L_bottom / c  # [s]

    for i, theta in enumerate(thetas):

        #* Criteria for the side component
        side_criteria_1 = antenna_height * np.tan(theta) + rock_depth * (np.sin(theta)) / (np.sqrt(er_regolith - np.sin(theta)**2))  # [m]
        side_criteria_2 = side_criteria_1 + h * (np.sin(theta)) / (np.sqrt(er_rock - np.sin(theta)**2))  # [m]
        for j, w in enumerate(rock_widths):
            if side_criteria_1 < w / 2 < side_criteria_2:
                #* Side component
                L_side = w * np.sin(theta) + 2 * antenna_height * np.cos(theta) + 2 * rock_depth * np.sqrt(er_regolith - np.sin(theta)**2)\
                            + 2 * h * np.sqrt(er_rock - np.sin(theta)**2)  # [m]
                T_side = L_side / c  # [s]
                w_theta_matrix[j, i] = T_side - T_bottom  # [s]


                #* 不等式のやつ
                left_hand = w * np.sin(theta) + 2 * h * (np.sqrt(er_rock - np.sin(theta)**2) - np.sqrt(er_rock)) # [m]
                right_hand = c * fwhm - 2 * antenna_height * (np.cos(theta) - 1) - 2 * rock_depth * (np.sqrt(er_regolith - np.sin(theta)**2) - np.sqrt(er_regolith))  # [m]
                if left_hand >= right_hand:
                    w_theta_TorF[j, i] = 1

            else:
                w_theta_matrix[j, i] = np.nan
                w_theta_TorF[j, i] = np.nan

    #* Find the minimum time difference for each width
    for i in range(len(rock_widths)):
        if np.all(np.isnan(w_theta_matrix[i])):
            w_h_delta_T[h_index, i] = np.nan
        else:
            min_delta_T = np.nanmin(w_theta_matrix[i])  # [s]
            w_h_delta_T[h_index, i] = min_delta_T # [s]

        if np.all(np.isnan(w_theta_TorF[i])):
            w_h_TorF[h_index, i] = np.nan
        elif np.nanmean(w_theta_TorF[i]) == 1: # nan以外の要素が全て1の場合
            w_h_TorF[h_index, i] = 1



w_h_delta_T_156ns = np.zeros((len(rock_heights), len(rock_widths)))
w_h_TorF_156ns = np.zeros((len(rock_heights), len(rock_widths)))
w_h_delta_T_04ns = np.zeros((len(rock_heights), len(rock_widths)))
w_h_TorF_04ns = np.zeros((len(rock_heights), len(rock_widths)))

# FWHM = 1.56nsの場合の計算
for i, height in tenumerate(rock_heights):
    w_h_delta_T = w_h_delta_T_156ns
    w_h_TorF = w_h_TorF_156ns
    calc(i, height, FWHM)

# FWHM = 0.4nsの場合の計算
for i, height in tenumerate(rock_heights):
    w_h_delta_T = w_h_delta_T_04ns
    w_h_TorF = w_h_TorF_04ns
    calc(i, height, FWHM_2GHz)



#* Plot
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_delta_T_156ns / 1e-9, cmap='jet',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                origin='lower'
                )

ax.set_xlabel('Width [m]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(which='both', axis='both', linestyle='-.')

#* x, y軸のメモリを0.3刻みにする
#ax.set_xticks(np.arange(0, 2.1, 0.3))
#ax.set_yticks(np.arange(0, 2.1, 0.3))

# --- ここから等高線の追加 ---
# imshow と同じ座標系に対応する x, y 軸配列を作成
x = np.linspace(rock_widths[0],  rock_widths[-1],  w_h_delta_T_156ns.shape[1])
y = np.linspace(rock_heights[0], rock_heights[-1], w_h_delta_T_156ns.shape[0])
X, Y = np.meshgrid(x, y)

# w_h_matrix = 1.56 の等高線(1本だけ)を描画
contour_level = FWHM / 1e-9  # [ns]
cs = ax.contour(X, Y, w_h_delta_T_156ns / 1e-9, levels=[contour_level], colors='w')  # [ns]

# 等高線にラベルを付ける場合
ax.clabel(cs, inline=True, fontsize=20, fmt=f"{contour_level:.2f}")

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Time difference [ns]', fontsize=24)
cbar.ax.tick_params(labelsize=20)

plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/w_h_deltaT.png')
plt.show()


#* Plot True or False
fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_TorF_156ns, cmap='coolwarm',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                origin='lower'
                )

ax.set_xlabel('Width [m]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(which='both', axis='both', linestyle='-.')

#* x, y軸のメモリを0.3刻みにする
#ax.set_xticks(np.arange(0, 2.1, 0.3))
#ax.set_yticks(np.arange(0, 2.1, 0.3))

plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/w_h_TorF.png')
plt.show()


#* Plot True or False with FDTD results
polarity_ok_size = [[1.8, 0.3], [2.1, 0.3], [2.4, 0.3], [2.7, 0.3], [3.0, 0.3],
                                [1.8, 0.6], [2.4, 0.6], [3.0, 0.6], [3.6, 0.6], [4.2, 0.6], [4.8, 0.6],
                                [5.4, 0.6], [6.0, 0.6],
                                [2.0, 1.5], [2.5, 1.5], [3.0, 1.5],
                                [2.5, 2.5], [3.0, 2.5],
                                [1.8, 1.8], [2.1, 2.1]
                                ]
polarity_not_ok_size = [[0.15, 0.15], [0.3, 0.3], [0.6, 0.3], [0.9, 0.3], [1.2, 0.3], [1.5, 0.3],
                            [0.6, 0.6], [1.2, 0.6], [0.9, 0.9], [1.2, 1.2], [1.5, 1.5], [2.0, 2.5]
                            ]

fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_TorF_156ns, cmap='coolwarm',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                origin='lower'
                )
for i in range(len(polarity_ok_size)):
        ax.plot(polarity_ok_size[i][0], polarity_ok_size[i][1], 'o', markersize=5, color='w')
for i in range(len(polarity_not_ok_size)):
        ax.plot(polarity_not_ok_size[i][0], polarity_not_ok_size[i][1], 'o', markersize=5, color='gray')

ax.set_xlabel('Width [m]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)
ax.set_xlim(0, 3.15)
ax.set_ylim(0, 3.15)
ax.tick_params(labelsize=20)
ax.grid(which='both', axis='both', linestyle='-.')

#* x, y軸のメモリを0.3刻みにする
#ax.set_xticks(np.arange(0, 2.1, 0.3))
#ax.set_yticks(np.arange(0, 2.1, 0.3))

plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/w_h_TorF_compare.png')
plt.show()



#* Plot the True of False with linear approximation
a = (3.0 - 0.15)/ (2.6 - 1.7)
b = - 1.7 * a
y_approx = a * rock_widths + b

fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_TorF_156ns, cmap='coolwarm',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                origin='lower'
                )
ax.plot(rock_widths, y_approx, color='w', linewidth=2)

ax.set_xlabel('Width [m]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)
ax.set_xlim(0, 3.15)
ax.set_ylim(0, 3.15)
ax.tick_params(labelsize=20)
ax.grid(which='both', axis='both', linestyle='-.')

plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/w_h_TorF_linear_approximation.png')
plt.show()



#* Plot the height-to-diameter ratio
max_diam_Di = 1.53 # [m]
min_diam_Di = 0.05 # [m]
diam_range_Di = np.arange(min_diam_Di, max_diam_Di, 0.01)
max_height_Di = 0.36 # [m]
min_height_Di = 0.03 # [m]

max_diam_Li_Wu = 2.52 # [m]
min_diam_Li_Wu = 0.05 # [m]
diam_range_Li_Wu = np.arange(min_diam_Li_Wu, max_diam_Li_Wu, 0.01)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='w', edgecolor='w', tight_layout=True)

# FWHM = 1.56nsの結果をプロット
im1 = ax1.imshow(w_h_TorF_156ns, cmap='coolwarm',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                origin='lower')

ax1.fill_between(diam_range_Di, min_height_Di, max_height_Di, color='y', alpha=0.5)
ax1.fill_between(diam_range_Li_Wu, min_diam_Li_Wu, max_diam_Li_Wu, color='w', alpha=0.5)
# Diのデータ用の四角形
ax1.plot([min_diam_Di, max_diam_Di, max_diam_Di, min_diam_Di, min_diam_Di],
         [min_height_Di, min_height_Di, max_height_Di, max_height_Di, min_height_Di],
         'k-', linewidth=2)

# Li & Wuのデータ用の四角形
ax1.plot([min_diam_Li_Wu, max_diam_Li_Wu, max_diam_Li_Wu, min_diam_Li_Wu, min_diam_Li_Wu],
         [min_diam_Li_Wu, min_diam_Li_Wu, max_diam_Li_Wu, max_diam_Li_Wu, min_diam_Li_Wu],
         'k-', linewidth=2)

ax1.set_xlabel('Width [m]', fontsize=24)
ax1.set_ylabel('Height [m]', fontsize=24)
ax1.set_xlim(0, 3.15)
ax1.set_ylim(0, 3.15)
ax1.tick_params(labelsize=20)
ax1.set_title('FWHM = 1.56 ns', fontsize=24)

# FWHM = 0.4nsの結果をプロット
im2 = ax2.imshow(w_h_TorF_04ns, cmap='coolwarm',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                origin='lower')

ax2.fill_between(diam_range_Di, min_height_Di, max_height_Di, color='y', alpha=0.5)
ax2.fill_between(diam_range_Li_Wu, min_diam_Li_Wu, max_diam_Li_Wu, color='w', alpha=0.5)
# Diのデータ用の四角形
ax2.plot([min_diam_Di, max_diam_Di, max_diam_Di, min_diam_Di, min_diam_Di],
         [min_height_Di, min_height_Di, max_height_Di, max_height_Di, min_height_Di],
         'k-', linewidth=2)
# Li & Wuのデータ用の四角形
ax2.plot([min_diam_Li_Wu, max_diam_Li_Wu, max_diam_Li_Wu, min_diam_Li_Wu, min_diam_Li_Wu],
         [min_diam_Li_Wu, min_diam_Li_Wu, max_diam_Li_Wu, max_diam_Li_Wu, min_diam_Li_Wu],
         'k-', linewidth=2)

ax2.set_xlabel('Width [m]', fontsize=24)
ax2.set_ylabel('Height [m]', fontsize=24)
ax2.set_xlim(0, 3.15)
ax2.set_ylim(0, 3.15)
ax2.tick_params(labelsize=20)
ax2.set_title('FWHM = 0.4 ns', fontsize=24)

plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/w_h_TorF_HD_comparison.png')
plt.show()