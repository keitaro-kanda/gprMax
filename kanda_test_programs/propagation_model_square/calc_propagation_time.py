import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from tqdm.contrib import tenumerate

# Physical constants
c0 = 299792458  # Speed of light in vacuum [m/s]



#* Constants
antenna_height = 0.3  # [m]
rock_depth = 2.0  # [m]

er_regolith = 3.0
er_rock = 9.0

h_dash = antenna_height


#* Parameters
rock_heights = np.arange(0, 2.101, 0.001)  # [m]
rock_widths = np.arange(0.3, 2.11, 0.3)  # [m]
thetas = np.arange(0, np.pi * 1/2, np.pi / 1440)  # [rad], 0 ~ pi/2



#* Bottom component
Lb = 2 * (antenna_height + rock_depth * np.sqrt(er_regolith) + rock_heights * np.sqrt(er_rock))  # [m]
Tb = Lb / c0  # [s]

Tb = np.vstack([Tb for _ in range(len(thetas))])
print('Tb:', Tb.shape)



#* Define function to calculate the side component
def calc_side_component(rock_width):
    Ls = np.zeros((len(thetas), len(rock_heights)))  # [m]
    #print('Ls:', Ls.shape)
    for i, theta in tenumerate(thetas, desc=f'Rock width: {rock_width:.1f} m'):
        for j, h in enumerate(rock_heights):
            # Criteriaの計算
            side_criteria_1 = antenna_height * np.tan(theta) + rock_depth * (np.sin(theta)) / (np.sqrt(3 - np.sin(theta)**2))  # [m]
            side_criteria_2 = side_criteria_1 + h * (np.sin(theta)) / (np.sqrt(9 - np.sin(theta)**2))  # [m]

            # 条件を判定
            if side_criteria_1 < rock_width / 2 < side_criteria_2:
                h_dash = (2 * np.cos(theta)**2 - 1) * antenna_height + \
                            (rock_width - 2 * np.sin(theta) * (rock_depth / np.sqrt(3 - np.sin(theta)**2) + h / np.sqrt(9 - np.sin(theta)**2))) * \
                            np.sin(theta) * np.cos(theta) # [m]
                Ls[i, j] = (antenna_height + h_dash) / np.cos(theta) \
                    + (6 * rock_depth) / (np.sqrt(3 - np.sin(theta)**2)) \
                    + (18 * h) / (np.sqrt(9 - np.sin(theta)**2))  # [m]
            else:
                Ls[i, j] = 'nan'

    Ts = Ls / c0  # [s]
    #print('Ts:', Ts.shape)

    #* Calculate the time difference
    delta_T = Ts - Tb  # [s]

    return Ts, delta_T


#* Define function to calculate minimum time difference
def calc_min_time_difference(delta_T):
    min_delta_T = np.zeros(len(rock_heights))  # 結果を格納する配列
    for i in range(len(rock_heights)):
        # delta_T[:, i] に NaN 以外の値があるかチェック
        if np.all(np.isnan(delta_T[:, i])):
            min_delta_T[i] = np.nan  # 全てがNaNの場合は NaN を設定
        else:
            min_delta_T[i] = np.nanmin(delta_T[:, i])  # NaN以外の最小値を取得
    return min_delta_T



def plot(Ts, delta_T, min_delta_T):
    #* Create a GridSpec layout
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])  # Equal-sized panels

    #* Bottom component
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(rock_heights, Tb[0] / 1e-9, linewidth=2)
    ax0.set_title('Bottom component', fontsize=24)
    ax0.set_ylabel('Two-way travel time [ns]', fontsize=20)

    #* Side component
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        Ts / 1e-9,
        extent=(rock_heights[0], rock_heights[-1], thetas[0], thetas[-1]),
        origin='lower',
        cmap='jet',
    )
    ax1.set_title('Side component', fontsize=24)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('bottom', size='10%', pad=1)
    cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar1.set_label('Two-way travel time [ns]', fontsize=20)
    cbar1.ax.tick_params(labelsize=18)

    #* Time difference
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(
        delta_T / 1e-9,
        extent=(rock_heights[0], rock_heights[-1], thetas[0], thetas[-1]),
        origin='lower',
        cmap='jet',
    )
    ax2.set_title('Time difference of two components', fontsize=24)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='10%', pad=1)
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
    cbar2.set_label('Time difference [ns]', fontsize=20)
    cbar2.ax.tick_params(labelsize=18)

    #* y軸のメモリをpiに変換
    y_ticks = np.arange(0, np.pi * 5 / 8, np.pi / 8)
    y_labels = ['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']
    for ax in [ax1, ax2]:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Initial transmission angle', fontsize=20)
        ax.set_ylim(0, np.pi / 4)
        #ax.set_xlim(1.0, np.max(rock_heights))

    #* Add common elements to all axes
    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel('Rock height [m]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(axis='both')
        ax.set_aspect('auto')

    fig.suptitle(f'Rock width: {w:.1f} m', fontsize=24)
    plt.tight_layout()
    plt.savefig(f'/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/propagation_time_w{w:.1f}.png')
    #plt.show()
    plt.close()



    #* Plot the minimum time difference
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.plot(rock_heights, min_delta_T / 1e-9, linewidth=2)

    plt.title(f'Rock width: {w:.1f} m', fontsize=24)
    plt.xlabel('Rock height [m]', fontsize=20)
    plt.ylabel('Minimum time difference [ns]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid()

    plt.savefig(f'/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/min_time_difference_w{w:.1f}.png')
    plt.close()



#* main
if __name__ == '__main__':
    for w in tqdm(rock_widths):
        Ts, delta_T = calc_side_component(w)
        min_delta_T = calc_min_time_difference(delta_T)
        plot(Ts, delta_T, min_delta_T)