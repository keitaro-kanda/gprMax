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
rock_heights = np.arange(0.15, 6.01, 0.05)  # [m]
rock_widths = np.arange(0.0, 6.01, 0.05)  # [m]
thetas = np.arange(0, np.pi * 1/2, np.pi / 1440)  # [rad], 0 ~ pi/2
er_regolith = 3.0
er_rock = 9.0
FWHM = 1.56e-9  # [s], 1.56 ns




#* Calculation
def calc(h_index, h):
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
                right_hand = c * FWHM - 2 * antenna_height * (np.cos(theta) - 1) - 2 * rock_depth * (np.sqrt(er_regolith - np.sin(theta)**2) - np.sqrt(er_regolith))  # [m]
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
        elif np.any(w_theta_TorF[i] == 1):
            w_h_TorF[h_index, i] = 1



w_h_delta_T = np.zeros((len(rock_heights), len(rock_widths)))
w_h_TorF = np.zeros((len(rock_heights), len(rock_widths)))
for i, height in tenumerate(rock_heights):
    calc(i, height)



#* Plot
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_delta_T / 1e-9, cmap='jet',
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
x = np.linspace(rock_widths[0],  rock_widths[-1],  w_h_delta_T.shape[1])
y = np.linspace(rock_heights[0], rock_heights[-1], w_h_delta_T.shape[0])
X, Y = np.meshgrid(x, y)

# w_h_matrix = 1.56 の等高線(1本だけ)を描画
contour_level = FWHM / 1e-9  # [ns]
cs = ax.contour(X, Y, w_h_delta_T / 1e-9, levels=[contour_level], colors='w')  # [ns]

# 等高線にラベルを付ける場合
ax.clabel(cs, inline=True, fontsize=20, fmt=f"{contour_level:.2f}")

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Time difference [ns]', fontsize=24)
cbar.ax.tick_params(labelsize=20)

plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/w_h_deltaT.png')
plt.show()


#* Plot with FDTD results
fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_TorF, cmap='coolwarm',
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
