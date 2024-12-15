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
#rock_heights = np.arange(0, 2.101, 0.001)  # [m]
#rock_widths = np.arange(0.3, 2.11, 0.3)  # [m]
rock_heights = np.arange(0.30, 2.11, 0.30)  # [m]
rock_widths = np.arange(0, 6.01, 0.01)  # [m]
thetas = np.arange(0, np.pi * 1/2, np.pi / 1440)  # [rad], 0 ~ pi/2



#* Bottom component
Lb = 2 * (antenna_height + rock_depth * np.sqrt(er_regolith) + rock_heights * np.sqrt(er_rock))  # [m]
Tb = Lb / c0  # [s], number of component is the same as the number of rock_heights



#* Define function to calculate the side component
def calc_side_component(height_index, Tb_at_height):
    Ls = np.zeros((len(thetas), len(rock_widths)))  # [m]

    height = rock_heights[height_index]
    for i, theta in tenumerate(thetas, desc=f'Rock height: {height:.1f} m'):
        for j, width in enumerate(rock_widths):
            # Criteriaの計算
            side_criteria_1 = antenna_height * np.tan(theta) + rock_depth * (np.sin(theta)) / (np.sqrt(3 - np.sin(theta)**2))  # [m]
            side_criteria_2 = side_criteria_1 + height * (np.sin(theta)) / (np.sqrt(9 - np.sin(theta)**2))  # [m]

            # 条件を判定
            if side_criteria_1 < width / 2 < side_criteria_2:
                h_dash = (2 * np.cos(theta)**2 - 1) * antenna_height + \
                            (width - 2 * np.sin(theta) * (rock_depth / np.sqrt(3 - np.sin(theta)**2) + height / np.sqrt(9 - np.sin(theta)**2))) * \
                            np.sin(theta) * np.cos(theta) # [m]
                Ls[i, j] = (antenna_height + h_dash) / np.cos(theta) \
                    + (6 * rock_depth) / (np.sqrt(3 - np.sin(theta)**2)) \
                    + (18 * height) / (np.sqrt(9 - np.sin(theta)**2))  # [m]
            else:
                Ls[i, j] = 'nan'

    Ts = Ls / c0  # [s]

    #* Calculate the time difference
    delta_T = Ts - Tb_at_height  # [s]

    return Ts, delta_T


#* Define function to calculate minimum time difference
def calc_min_time_difference(delta_T):
    min_delta_T = np.zeros(len(rock_widths))  # 結果を格納する配列
    mean_delta_T = np.zeros(len(rock_widths))  # 結果を格納する配列
    for i in range(len(rock_widths)):
        # delta_T[:, i] に NaN 以外の値があるかチェック
        if np.all(np.isnan(delta_T[:, i])):
            min_delta_T[i] = np.nan  # 全てがNaNの場合は NaN を設定
        else:
            min_delta_T[i] = np.nanmin(delta_T[:, i])  # NaN以外の最小値を取得
            mean_delta_T[i] = np.nanmean(delta_T[:, i])  # NaN以外の平均値を取得
    return min_delta_T, mean_delta_T


#* Define function to calculate mean and standard deviation of time difference
def calc_time_difference(delta_T):
    delta_T_mean = np.zeros(len(rock_widths))  # 結果を格納する配列
    delta_T_std = np.zeros(len(rock_widths))  # 結果を格納する配列
    for i in range(len(rock_widths)):
        # delta_T[:, i] に NaN 以外の値があるかチェック
        if np.all(np.isnan(delta_T[:, i])):
            delta_T_mean[i] = np.nan
            delta_T_std[i] = np.nan
        else:
            delta_T_mean[i] = np.nanmean(delta_T[:, i])
            delta_T_std[i] = np.nanstd(delta_T[:, i])
    return delta_T_mean, delta_T_std



def plot(height, Ts, delta_T, min_delta_T, mean_delta_T):
    #* Create a GridSpec layout
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])  # Equal-sized panels

    #* Bottom component
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(rock_heights, Tb / 1e-9, linewidth=2)
    ax0.axvline(height, color='red', linestyle='--', linewidth=3)

    ax0.set_xlabel('Rock height [m]', fontsize=20)
    ax0.set_title('Bottom component', fontsize=24)
    ax0.set_ylabel('Two-way travel time [ns]', fontsize=20)

    #* Side component
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        Ts / 1e-9,
        extent=(rock_widths[0], rock_widths[-1], thetas[0], thetas[-1]),
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
        extent=(rock_widths[0], rock_widths[-1], thetas[0], thetas[-1]),
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
        ax.set_xlabel('Rock width [m]', fontsize=20)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Initial transmission angle', fontsize=20)
        #ax.set_ylim(0, np.pi / 4)
        #ax.set_xlim(1.0, np.max(rock_heights))

    #* Add common elements to all axes
    for ax in [ax0, ax1, ax2]:
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(axis='both')
        ax.set_aspect('auto')

    fig.suptitle(f'Rock height: {height:.1f} m', fontsize=24)
    plt.tight_layout()
    plt.savefig(f'/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/propagation_time_h{height:.1f}.png')
    #plt.show()
    plt.close()



    #* Plot the minimum time difference
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.plot(rock_widths, min_delta_T / 1e-9, linewidth=2, label='Min')
    #plt.plot(rock_widths, mean_delta_T / 1e-9, linewidth=2, label='Mean', linestyle='-.')

    plt.title(f'Rock height: {height:.1f} m', fontsize=24)
    plt.xlabel('Rock width [m]', fontsize=20)
    plt.ylabel('Minimum time difference [ns]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid()
    plt.legend(fontsize=18)

    plt.savefig(f'/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/min_time_difference_h{height:.1f}.png')
    plt.close()

    """
    #* Plot the mean and standard deviation of time difference
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.errorbar(rock_widths, delta_T_mean / 1e-9, yerr=delta_T_std / 1e-9, fmt='o', markersize=5, color='gray', alpha=0.5)
    plt.plot(rock_widths, delta_T_mean / 1e-9, linewidth=2, color='orange')

    plt.title(f'Rock height: {height:.1f} m', fontsize=24)
    plt.xlabel('Rock width [m]', fontsize=20)
    plt.ylabel('Minimum time difference [ns]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid()

    plt.savefig(f'/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/mean_std_time_difference_h{height:.1f}.png')
    #plt.show()
    plt.close()
    """


def compare_with_FDTD(height, min_delta_T):
    #* 現状計算している高さ30 cm, 60 cmの場合のFDTD結果のみ実装
    if height == 0.3:
        time_difference_FDTD = [[1.8, 1.82], [2.4, 3.04], [2.7, 3.74], [3.0, 4.53]]
    elif height == 0.6:
        time_difference_FDTD = [[1.8, 1.66], [2.4, 2.83], [3.0, 4.26], [3.6, 5.91], [4.2, 7.71], [4.8, 9.62],
                                [5.4, 11.59], [6.0, 13.55]]
    #* Plot the minimum time difference
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.plot(rock_widths, min_delta_T / 1e-9, linewidth=2, label='Model')

    #* Plt the FDTD results
    for i in range(len(time_difference_FDTD)):
        plt.plot(time_difference_FDTD[i][0], time_difference_FDTD[i][1], 'o', markersize=5, color='r', label='FDTD' if i == 0 else None)

    plt.title(f'Rock height: {height:.1f} m', fontsize=24)
    plt.xlabel('Rock width [m]', fontsize=20)
    plt.ylabel('Minimum time difference [ns]', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.grid()
    plt.legend(fontsize=18)

    plt.savefig(f'/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/min_time_difference_h{height:.1f}_compare.png')
    plt.close()




#* main
if __name__ == '__main__':
    for i, h in tenumerate(rock_heights, desc='Rock height'):
        Tb_i = np.tile(Tb[i], (len(thetas), len(rock_widths)))  # [s]
        Ts, delta_T = calc_side_component(i, Tb_i)
        min_delta_T, mean_delta_T = calc_min_time_difference(delta_T)
        delta_T_mean, delta_T_std = calc_time_difference(delta_T)
        plot(h, Ts, delta_T, min_delta_T, mean_delta_T)
        if h == 0.3 or h == 0.6:
            compare_with_FDTD(h, min_delta_T)