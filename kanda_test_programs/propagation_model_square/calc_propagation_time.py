import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1

# Physical constants
c0 = 299792458  # Speed of light in vacuum [m/s]



#* Constants
antenna_height = 0.3  # [m]
rock_depth = 2.0  # [m]

er_regolith = 3.0
er_rock = 9.0

h_dash = antenna_height


#* Parameters
rock_size_array = np.arange(0, 2.11, 0.01)  # [m]
theta_array = np.arange(0, np.pi * 181 / 360, np.pi / 360)  # [rad]
print('theta:', len(theta_array))


#* Bottom component
Lb = 2 * (antenna_height + rock_depth * np.sqrt(er_regolith) + rock_size_array * np.sqrt(er_rock))  # [m]
Tb = Lb / c0  # [s]

Tb = np.vstack([Tb for _ in range(len(theta_array))])
print('Tb:', Tb.shape)



#* Side component
Ls = np.zeros((len(theta_array), len(rock_size_array)))  # [m]
print('Ls:', Ls.shape)
for i, theta in enumerate(theta_array):
    for j, rock_size in enumerate(rock_size_array):
        side_criteria_1 = antenna_height * np.tan(theta) + rock_depth * (np.sin(theta)) / (np.sqrt(3 - np.sin(theta)**2))  # [m]
        side_criteria_2 = antenna_height * np.tan(theta) + rock_depth * (3 * np.sin(theta)) / (np.sqrt(9 - np.sin(theta)**2))\
                + rock_size * (3 * np.sin(theta)) / (np.sqrt(9 - np.sin(theta)**2))  # [m]
        if side_criteria_1 < rock_size/2 and rock_size/2 < side_criteria_2:
            Ls[i, j] = (antenna_height + h_dash) / np.cos(theta) \
                + (6 * rock_depth) / (np.sqrt(3 - np.sin(theta)**2)) \
                + (18 * rock_size) / (np.sqrt(9 - np.sin(theta)**2))  # [m]
        else:
            Ls[i, j] = 'nan'

Ts = Ls / c0  # [s]
print('Ts:', Ts.shape)


delta_T = Ts - Tb  # [s]



#* Plot
fig, ax = plt.subplots(1, 3, figsize=(20, 7), tight_layout=True)

#* Plot colormap
color = 'jet'
colorbar_size = '10%'
colorbar_pad = 1

#* Bottom component
im0 = ax[0].imshow(Tb / 1e-9,
            extent=(rock_size_array[0], rock_size_array[-1], theta_array[0], theta_array[-1]),
            cmap = color)
ax[0].set_title('Bottom component', fontsize=24)
ax[0].set_ylabel('Initial transmission angle', fontsize=20)

delvider0 = axgrid1.make_axes_locatable(ax[0])
cax0 = delvider0.append_axes('bottom', size=colorbar_size, pad=colorbar_pad)
cbar0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
cbar0.set_label('Time [ns]', fontsize=20)
cbar0.ax.tick_params(labelsize=18)

#* Side component
im1 = ax[1].imshow(Ts / 1e-9,
            extent=(rock_size_array[0], rock_size_array[-1], theta_array[0], theta_array[-1]),
            cmap = color)
ax[1].set_title('Side component', fontsize=24)

delvider1 = axgrid1.make_axes_locatable(ax[1])
cax1 = delvider1.append_axes('bottom', size=colorbar_size, pad=colorbar_pad)
cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
cbar1.set_label('Time [ns]', fontsize=20)
cbar1.ax.tick_params(labelsize=18)

#* Time difference
im2 = ax[2].imshow(delta_T / 1e-9,
            extent=(rock_size_array[0], rock_size_array[-1], theta_array[0], theta_array[-1]),
            cmap = color)
ax[2].set_title('Time difference', fontsize=24)

delvider2 = axgrid1.make_axes_locatable(ax[2])
cax2 = delvider2.append_axes('bottom', size=colorbar_size, pad=colorbar_pad)
cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
cbar2.set_label('Time [ns]', fontsize=20)
cbar2.ax.tick_params(labelsize=18)


#* y軸のメモリをpiに変換
y_ticks = np.arange(0, np.pi * 5 / 8, np.pi / 8)
y_labels = ['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']
for i in range(3):
    ax[i].set_xlabel('Rock size [m]', fontsize=20)

    ax[i].set_yticks(y_ticks)
    ax[i].set_yticklabels(y_labels)
    ax[i].tick_params(axis='both', labelsize=18)


plt.savefig('kanda_test_programs/propagation_model_square/propagation_time.png')
plt.show()


"""
plt.figure(figsize=(12, 10), tight_layout=True)

# 第1縦軸，Tb, Ts vs rock_sizeのプロット
plt.plot(rock_size, Tb / 1e-9, label='Bottom', linewidth=2, color='red')
plt.plot(rock_size, Ts / 1e-9, label='Side', linewidth=2, color='blue')

plt.xlabel('Rock size [m]', fontsize=24)
plt.ylabel('Time [ns]', fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20)
plt.grid(axis='both')

# 第2縦軸，delta_T vs rock_sizeのプロット
plt.twinx()
plt.plot(rock_size, delta_T / 1e-9, color='green')


plt.ylabel('Time difference [ns]', color='green', fontsize=24)
plt.tick_params(axis='y', labelcolor='green', labelsize=20)
plt.ylim(2.5, 4.5)


plt.show()
"""
