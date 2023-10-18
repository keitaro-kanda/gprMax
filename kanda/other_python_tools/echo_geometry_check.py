import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1
import matplotlib.patches as patches


x_list = np.arange(0, 550, 1)
z_list = np.arange(0, 300, 1)
recieved_time_array = np.zeros([len(z_list), len(x_list)])

x_tx = 250
x_rx = 275
antenna_zpoint = 77
h = 1
wave_start_time = 0.5e-8
c = 3e8


epsilon_0 = 1.0
epsilon_ground_1 = 6.0


for x in tqdm(x_list, desc='x'):
    for z in z_list:
        pass_len_ref2rx = np.sqrt(np.abs(x_rx - x)**2 + np.abs(antenna_zpoint - z)**2 )
        pass_len_tx2ref = np.sqrt(np.abs(x_tx - x)**2 + np.abs(antenna_zpoint - z)**2 ) # [m]

        L_vacuum_k = np.sqrt(epsilon_0)*(pass_len_tx2ref + pass_len_ref2rx) * h / np.abs(antenna_zpoint - z)
        L_ground_k = np.sqrt(epsilon_ground_1)*(pass_len_tx2ref + pass_len_ref2rx) * np.abs(antenna_zpoint - z - h) / np.abs(antenna_zpoint - z)


        delta_t = (L_vacuum_k + L_ground_k) / c # [s]

        
        recieved_time_array[z, x] = delta_t + wave_start_time # [s]


fig = plt.figure(figsize=(10, 7), facecolor='w', edgecolor='w')
ax = fig.add_subplot(211)

plt.imshow(recieved_time_array, vmin=0, vmax=4e-6, cmap='rainbow')
delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)

ax.set_ylim(275, 70)
ax.set_xlim(100, 450)

plt.colorbar(cax=cax, label = 'delay time')

ax.set_title('rx:x='+str(x_rx) + ', tx:x='+str(x_tx))
ax.set_xlabel('Horizontal distance [m]', size=14)
ax.set_ylabel('Depth form surface [m]', size=14)


# 地形のプロット
edge_color = 'white'
rille_apex_list = [(0, 10), (25, 10), 
            (175, 260), (375, 260),
            (525, 10), (550, 10)]
rille = patches.Polygon(rille_apex_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(rille)

surface_hole_tube_list = [(40, 35), (250, 35),
                    (250, 60), (200, 60),
                    (200, 77), (350, 77),
                    (350, 60), (300, 60),
                    (300, 35), (515, 35)]
tube = patches.Polygon(surface_hole_tube_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(tube)


#layer2_apex_list = [(100, 135), (450, 135)]
layer2_apex_list = [(139, 200), (411, 200)]
layer2 = patches.Polygon(layer2_apex_list, ec=edge_color, linestyle='--', fill=False, linewidth=1, closed=False)
ax.add_patch(layer2)


plt.savefig('kanda/domain_550x270_v2/rille_rough?/map_fig/delay_time.png', bbox_inches='tight', dpi=300)
plt.show()
