import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Physical constants
c0 = 299792458  # Speed of light in vacuum [m/s]



#* Constants
antenna_height = 0.3  # [m]
rock_depth = 2.0  # [m]

er_regolith = 3.0
er_rock = 9.0

h_dash = antenna_height


#* Parameters
rock_size_array = np.arange(0, 2.101, 0.001)  # [m]
theta_array = np.arange(0, np.pi * 1/2, np.pi / 720)  # [rad], 0 ~ pi/2
print('theta:', len(theta_array))


#* Bottom component
Lb = 2 * (antenna_height + rock_depth * np.sqrt(er_regolith) + rock_size_array * np.sqrt(er_rock))  # [m]
Tb = Lb / c0  # [s]

Tb = np.vstack([Tb for _ in range(len(theta_array))])
print('Tb:', Tb.shape)



#* Side component
Ls = np.zeros((len(theta_array), len(rock_size_array)))  # [m]
print('Ls:', Ls.shape)
for i, theta in tqdm(enumerate(theta_array), desc='Calculating side component...'):
    for j, rock_size in enumerate(rock_size_array):
        side_criteria_1 = antenna_height * np.tan(theta) + rock_depth * (np.sin(theta)) / (np.sqrt(3 - np.sin(theta)**2))  # [m]
        side_criteria_2 = antenna_height * np.tan(theta) + rock_depth * (np.sin(theta)) / (np.sqrt(9 - np.sin(theta)**2))\
                + rock_size * (np.sin(theta)) / (np.sqrt(9 - np.sin(theta)**2))  # [m]
        if side_criteria_1 < rock_size/2 and rock_size/2 < side_criteria_2:
        #if side_criteria_1 < rock_size / 2:
            Ls[i, j] = (antenna_height + h_dash) / np.cos(theta) \
                + (6 * rock_depth) / (np.sqrt(3 - np.sin(theta)**2)) \
                + (18 * rock_size) / (np.sqrt(9 - np.sin(theta)**2))  # [m]
        else:
            Ls[i, j] = 'nan'

Ts = Ls / c0  # [s]
print('Ts:', Ts.shape)


#* Calculate the time difference
delta_T = Ts - Tb  # [s]


#* Survey the smallest time difference in each rock size
min_delta_T = np.zeros(len(rock_size_array))
# 'nan'の場合は、最小値を取らないようにする
for i in range(len(rock_size_array)):
    if Ts[:, i].all() == 'nan':
        min_delta_T[i] = 'nan'
    else:
        min_delta_T[i] = np.nanmin(delta_T[:, i])

print('min_delta_T:', min_delta_T.shape)
print('min_delta_T:', min_delta_T)



#* Create a GridSpec layout
fig = plt.figure(figsize=(20, 7))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])  # Equal-sized panels

#* Bottom component
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(rock_size_array, Tb[0] / 1e-9, label='Bottom', linewidth=2)
ax0.set_title('Bottom component', fontsize=24)
ax0.set_ylabel('Two-way travel ime [ns]', fontsize=20)
ax0.set_aspect('auto')

#* Side component
ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(
    Ts / 1e-9,
    extent=(rock_size_array[0], rock_size_array[-1], theta_array[0], theta_array[-1]),
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
    extent=(rock_size_array[0], rock_size_array[-1], theta_array[0], theta_array[-1]),
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
    ax.set_xlim(1.0, np.max(rock_size_array))

#* Add common elements to all axes
for ax in [ax0, ax1, ax2]:
    ax.set_xlabel('Rock size [m]', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(axis='both')

plt.tight_layout()
plt.savefig('kanda_test_programs/propagation_model_square/propagation_time.png')
plt.show()



#* Plot the minimum time difference
plt.figure(figsize=(10, 8))
plt.plot(rock_size_array, min_delta_T / 1e-9, linewidth=2)

plt.xlabel('Rock size [m]', fontsize=20)
plt.ylabel('Minimum time difference [ns]', fontsize=20)
plt.tick_params(labelsize=18)
plt.grid()

plt.savefig('kanda_test_programs/propagation_model_square/min_time_difference.png')
plt.show()