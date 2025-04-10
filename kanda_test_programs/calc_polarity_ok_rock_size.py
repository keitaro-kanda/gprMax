import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from tqdm.contrib import tenumerate
import argparse
import os

# Physical constants
c = 299792458  # Speed of light in vacuum [m/s]

#* Global parameters
rock_heights = np.arange(0.0, 3.15, 0.05)  # [m]
rock_widths = np.arange(0.0, 3.15, 0.05)  # [m]
thetas = np.arange(0, np.pi * 1/2, np.pi / 2880)  # [rad], 0 ~ pi/2

#* Initialize arrays
w_h_delta_T = np.zeros((len(rock_heights), len(rock_widths)))
w_h_delta_T_max = np.zeros((len(rock_heights), len(rock_widths)))
w_h_TorF = np.zeros((len(rock_heights), len(rock_widths)))

#* FDTD comparison data
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

#* Height-to-diameter ratio parameters
max_diam_Di = 1.53 # [m]
min_diam_Di = 0.05 # [m]
diam_range_Di = np.arange(min_diam_Di, max_diam_Di, 0.01)
max_height_Di = 0.36 # [m]
min_height_Di = 0.03 # [m]

max_diam_Li_Wu = 2.52 # [m]
min_diam_Li_Wu = 0.05 # [m]
diam_range_Li_Wu = np.arange(min_diam_Li_Wu, max_diam_Li_Wu, 0.01)



#* Calculation function
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
            max_delta_T = np.nanmax(w_theta_matrix[i])  # [s]
            w_h_delta_T[h_index, i] = min_delta_T # [s]
            w_h_delta_T_max[h_index, i] = max_delta_T # [s]

        if np.all(np.isnan(w_theta_TorF[i])):
            w_h_TorF[h_index, i] = np.nan
        elif np.nanmean(w_theta_TorF[i]) == 1: # nan以外の要素が全て1の場合
            w_h_TorF[h_index, i] = 1




if __name__ == '__main__':
    #* Get parameters from user input
    print("\n=== Parameter Settings ===")
    antenna_height = float(input("Enter GPR antenna height [m] (default=0.3): ") or "0.3")
    rock_depth = float(input("Enter rock depth [m] (default=2.0): ") or "2.0")
    er_regolith = float(input("Enter relative permittivity of regolith (default=3.0): ") or "3.0")
    er_rock = float(input("Enter relative permittivity of rock (default=9.0): ") or "9.0")
    FWHM = float(input("Enter FWHM [s] (default=1.56e-9): ") or "1.56e-9")

    #* Set output directory
    output_dir = '/Volumes/SSD_Kanda_BUFFALO/gprMax/propagation_path_model/polarity_size'
    param_dir = os.path.join(output_dir, f'H{antenna_height}_d{rock_depth}_er-reg{er_regolith}_er-rock{er_rock}_FWHM{FWHM:.2e}ns')
    os.makedirs(param_dir, exist_ok=True)

    print(f"\nResults will be saved in: {param_dir}")
    print("\nStarting calculations...\n")

    #* Calculate for given FWHM
    for i, height in tenumerate(rock_heights):
        calc(i, height, FWHM)

    #* Plot the minimum time difference
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='w', edgecolor='w', tight_layout=True)
    im = ax.imshow(w_h_delta_T / 1e-9, cmap='jet',
                    extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                    origin='lower'
                    )

    ax.set_xlabel('Width [m]', fontsize=24)
    ax.set_ylabel('Height [m]', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.grid(which='both', axis='both', linestyle='-.')
    ax.set_xlim(0, 3.15)
    ax.set_ylim(0, 3.15)

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

    plt.savefig(os.path.join(param_dir, 'w_h_deltaT.png'))
    plt.show()


    #* Plot the maximum time difference
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='w', edgecolor='w', tight_layout=True)
    im = ax.imshow(w_h_delta_T_max / 1e-9, cmap='jet',
                    extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                    origin='lower'
                    )
    ax.set_xlabel('Width [m]', fontsize=24)
    ax.set_ylabel('Height [m]', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.grid(which='both', axis='both', linestyle='-.')
    ax.set_xlim(0, 3.15)
    ax.set_ylim(0, 3.15)

    #* x, y軸のメモリを0.3刻みにする
    #ax.set_xticks(np.arange(0, 2.1, 0.3))
    #ax.set_yticks(np.arange(0, 2.1, 0.3))

    # --- ここから等高線の追加 ---
    # imshow と同じ座標系に対応する x, y 軸配列を作成
    x = np.linspace(rock_widths[0],  rock_widths[-1],  w_h_delta_T_max.shape[1])
    y = np.linspace(rock_heights[0], rock_heights[-1], w_h_delta_T_max.shape[0])
    X, Y = np.meshgrid(x, y)

    # w_h_matrix = 1.56 の等高線(1本だけ)を描画
    contour_level = FWHM / 1e-9  # [ns]
    cs = ax.contour(X, Y, w_h_delta_T_max / 1e-9, levels=[contour_level], colors='w')  # [ns]

    # 等高線にラベルを付ける場合
    ax.clabel(cs, inline=True, fontsize=20, fmt=f"{contour_level:.2f}")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Time difference [ns]', fontsize=24)
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(os.path.join(param_dir, 'w_h_deltaT_max.png'))
    plt.show()

    #* Plot True or False
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
    im = ax.imshow(w_h_TorF, cmap='coolwarm',
                    extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                    origin='lower'
                    )

    ax.set_xlabel('Width [m]', fontsize=24)
    ax.set_ylabel('Height [m]', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.grid(which='both', axis='both', linestyle='-.')
    ax.set_xlim(0, 3.15)
    ax.set_ylim(0, 3.15)

    #* x, y軸のメモリを0.3刻みにする
    #ax.set_xticks(np.arange(0, 2.1, 0.3))
    #ax.set_yticks(np.arange(0, 2.1, 0.3))

    plt.savefig(os.path.join(param_dir, 'w_h_TorF.png'))
    plt.show()


    #* Plot True or False with FDTD results (only for default parameters)
    if (rock_depth == 2.0 and er_regolith == 3.0 and er_rock == 9.0 and FWHM == 1.56e-9):
        
        print("\nCreating FDTD comparison plots...")
        
        #* Plot True or False with FDTD results
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
        im = ax.imshow(w_h_TorF, cmap='coolwarm',
                      extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], 
                      aspect='equal', origin='lower')
        
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

        plt.savefig(os.path.join(param_dir, 'w_h_TorF_compare.png'))
        plt.show()

        #* Plot the True of False with linear approximation
        a = (3.0 - 0.15)/ (2.6 - 1.7)
        b = - 1.7 * a
        y_approx = a * rock_widths + b

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)
        im = ax.imshow(w_h_TorF, cmap='coolwarm',
                        extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], 
                        aspect='equal', origin='lower')
        ax.plot(rock_widths, y_approx, color='w', linewidth=2)

        ax.set_xlabel('Width [m]', fontsize=24)
        ax.set_ylabel('Height [m]', fontsize=24)
        ax.set_xlim(0, 3.15)
        ax.set_ylim(0, 3.15)
        ax.tick_params(labelsize=20)
        ax.grid(which='both', axis='both', linestyle='-.')

        plt.savefig(os.path.join(param_dir, 'w_h_TorF_linear_approximation.png'))
        plt.show()



    #* Plot with the height-to-diameter ratio obtained by CE-3
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='w', tight_layout=True)

    im = ax.imshow(w_h_TorF, cmap='coolwarm',
                    extent=[rock_widths[0], rock_widths[-1], rock_heights[0], rock_heights[-1]], aspect='equal',
                    origin='lower')

    ax.fill_between(diam_range_Di, min_height_Di, max_height_Di, color='y', alpha=0.5)
    ax.fill_between(diam_range_Li_Wu, min_diam_Li_Wu, max_diam_Li_Wu, color='w', alpha=0.5)
    
    # Diのデータ用の四角形
    ax.plot([min_diam_Di, max_diam_Di, max_diam_Di, min_diam_Di, min_diam_Di],
            [min_height_Di, min_height_Di, max_height_Di, max_height_Di, min_height_Di],
            'k-', linewidth=2)

    # Li & Wuのデータ用の四角形
    ax.plot([min_diam_Li_Wu, max_diam_Li_Wu, max_diam_Li_Wu, min_diam_Li_Wu, min_diam_Li_Wu],
            [min_diam_Li_Wu, min_diam_Li_Wu, max_diam_Li_Wu, max_diam_Li_Wu, min_diam_Li_Wu],
            'k-', linewidth=2)

    ax.set_xlabel('Width [m]', fontsize=24)
    ax.set_ylabel('Height [m]', fontsize=24)
    ax.set_xlim(0, 3.15)
    ax.set_ylim(0, 3.15)
    ax.tick_params(labelsize=20)
    ax.grid(which='both', axis='both', linestyle='-.')

    plt.savefig(os.path.join(param_dir, 'w_h_TorF_HD_comparison.png'))
    plt.show()
