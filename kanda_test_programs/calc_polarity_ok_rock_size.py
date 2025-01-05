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
rock_heights = np.arange(0.0, 2.11, 0.01)  # [m]
rock_widths = np.arange(0.0, 2.11, 0.01)  # [m]
thetas = np.arange(0, np.pi * 1/2, np.pi / 1440)  # [rad], 0 ~ pi/2
er_regolith = 3.0
er_rock = 9.0
FWHM = 1.56e-9  # [s], 1.56 ns




#* Calculation
def calc(w, h):
    side_criteria_1 = antenna_height * np.tan(thetas) + rock_depth * (np.sin(thetas)) / (np.sqrt(er_regolith - np.sin(thetas)**2))  # [m]
    side_criteria_2 = side_criteria_1 + h * (np.sin(thetas)) / (np.sqrt(er_rock - np.sin(thetas)**2))  # [m]
    # マスクを作る:  side_criteria_1 < w/2 < side_criteria_2 を満たす要素だけ取り出す
    mask = (side_criteria_1 < w / 2) & (w / 2 < side_criteria_2)
    #mask = side_criteria_1 < w / 2
    #mask = w / 2 < side_criteria_2

    L_bottom = 2 * (antenna_height + rock_depth * np.sqrt(er_regolith) + h * np.sqrt(er_rock))  # [m]
    T_bottom = L_bottom / c  # [s]
    #if w == 0:
    #    print(h, T_bottom)

    L_side = w + 2 * antenna_height * np.cos(thetas) + 2 * rock_depth * np.sqrt(er_regolith - np.sin(thetas)**2)\
                 + 2 * h * np.sqrt(er_rock - np.sin(thetas)**2)  # [m]
    L_side = L_side[mask]  # [m]
    T_side = L_side / c  # [s]
    #T_side = T_side[mask] # [s]
    if len(T_side) == 0:
        return 'nan'
    else:
        T_side_min = np.amin(T_side)  # [s]
        #if h == 2.1:
        #    print(w, T_side_min)

        #delta_T = (T_side_min - T_bottom)  # [s]
        delta_T = np.amin(T_side - T_bottom)  # [s]

    #if delta_T > FWHM:
    #if delta_T > 0:
    return delta_T / 1e-9 # [ns]


    """
    # マスクを満たす θ だけ left_hand, right_hand を計算
    left_hand_array  = w + 2 * h * (np.sqrt(er_rock - np.sin(thetas)**2) - np.sqrt(er_rock))
    right_hand_array = c * FWHM - 2 * rock_depth * (np.sqrt(er_regolith - np.sin(thetas)**2) - np.sqrt(er_regolith))\
                            - 2 * antenna_height * (np.cos(thetas) - 1)

    # マスクを満たす要素を抽出
    left_sub  = left_hand_array[mask]
    right_sub = right_hand_array[mask]
    #left_sub = left_hand_array
    #right_sub = right_hand_array

    # マスクが空なら (条件を満たす theta が皆無なら)、元コードでは「np.all([] > []) → True」の扱いなので True にする。
    if mask.sum() == 0:
        return False

    # すべての要素について left_sub > right_sub なら True, さもなくば False
    return np.all(left_sub > right_sub)
    """


w_h_matrix = np.zeros((len(rock_heights), len(rock_widths)))
#for i, width in tenumerate(rock_widths):
    #for j, height in enumerate(rock_heights):
for i, height in tenumerate(rock_heights):
    for j, width in enumerate(rock_widths):
        #if calc(width, height)==True:
        #    w_h_matrix[j, i] = 1
        #else:
        #    w_h_matrix[j, i] = 0
        w_h_matrix[i, j] = calc(width, height)

#* Plot
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w', edgecolor='w', tight_layout=True)
im = ax.imshow(w_h_matrix, cmap='jet',
                extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='auto',
                origin='lower'
                )

ax.set_xlabel('Width [m]', fontsize=24)
ax.set_ylabel('Height [m]', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(which='both', axis='both', linestyle='-.')

#* x, y軸のメモリを0.3刻みにする
ax.set_xticks(np.arange(0, 2.1, 0.3))
ax.set_yticks(np.arange(0, 2.1, 0.3))

# --- ここから等高線の追加 ---
# imshow と同じ座標系に対応する x, y 軸配列を作成
x = np.linspace(rock_widths[0],  rock_widths[-1],  w_h_matrix.shape[1])
y = np.linspace(rock_heights[0], rock_heights[-1], w_h_matrix.shape[0])
X, Y = np.meshgrid(x, y)

# w_h_matrix = 1.56 の等高線(1本だけ)を描画
contour_level = FWHM / 1e-9  # [ns]
cs = ax.contour(X, Y, w_h_matrix, levels=[contour_level], colors='k')  # colors='k'で黒線

# 等高線にラベルを付ける場合
ax.clabel(cs, inline=True, fontsize=20, fmt=f"{contour_level:.2f}")

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Time difference [ns]', fontsize=24)
cbar.ax.tick_params(labelsize=20)

plt.show()
