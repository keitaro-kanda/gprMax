import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axes_grid1 as axgrid1

from tools.core.outputfiles_merge import get_output_data

# Input
file_name = input("Enter the path to the .out file: ").strip()
rx_steps = input("Enter receiver step size (m) [default: 0.2]: ").strip()
if rx_steps == '':
    rx_step = 0.2
else:
    rx_step = float(rx_steps)

output_basename = 'background'
output_dir = os.path.join(os.path.dirname(file_name), output_basename)
os.makedirs(output_dir, exist_ok=True)


# .outファイルの読み込み
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()

for rx in range(1, nrx + 1):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')
print(f"Loaded data from {file_name} with shape {outputdata.shape} and dt={dt:.2e} s")


# 全A-scanの平均を取る
outputdata_ave = np.mean(outputdata[:, 1:], axis=1) # 1列目はpml内なので除外
print(f"Calculated average background trace with shape {outputdata_ave.shape}")


# 強度をdBに変換
outputdata_ave_db = 20 * np.log10(np.abs(outputdata_ave) / np.amax(np.abs(outputdata_ave)) + 1e-12) # avoid log(0)

# 移動平均の計算
window_size = len(outputdata_ave_db)//20
outputdata_ave_db_moving = np.convolve(outputdata_ave_db, np.ones(window_size)/window_size, 'same') # 入力サイズと同じサイズの配列を返す


# plot background only
plt.figure(figsize=(4, 8), facecolor='w', edgecolor='w')
plt.plot(outputdata_ave_db, np.arange(outputdata_ave.shape[0]) * dt * 1e9, color='k', linestyle='-') # time in ns
plt.plot(outputdata_ave_db_moving, np.arange(outputdata_ave.shape[0]) * dt * 1e9, color='r', linestyle='--', alpha=0.7) # time in ns
plt.xlabel('Amplitude (dB)', size=14)
plt.ylabel('Time (ns)', size=14)
plt.ylim(outputdata_ave.shape[0] * dt * 1e9, 0) # time in ns
plt.tick_params(labelsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'background_trace.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'background_trace.pdf'), format='pdf', dpi=300)
plt.show()


# plot B-scan with background
# B-scanプロットの幅とbackgroundプロットの幅を2:1にする
fig, ax = plt.subplots(
    nrows=1, # 縦
    ncols=2, # 横
    width_ratios=[3, 1],
    height_ratios=[1],
    figsize=(12, 8)
)

im = ax[0].imshow(outputdata,
             extent=[0, outputdata.shape[1] * rx_step, outputdata.shape[0] * dt * 1e9, 0],
            interpolation='nearest', aspect='auto', cmap='seismic',
            vmin=-np.amax(np.abs(outputdata[:, 1:]))/1e3, vmax=np.amax(np.abs(outputdata[:, 1:]))/1e3)
ax[0].set_xlabel('Trace number', size=14)
ax[0].set_ylabel('Time (ns)', size=14)
ax[0].tick_params(labelsize=12)
ax[0].grid()
# coloarbar
delvider = axgrid1.make_axes_locatable(ax[0])
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Amplitude', size=14)
cbar.ax.tick_params(labelsize=12)

ax[1].plot(outputdata_ave_db, np.arange(outputdata_ave.shape[0]) * dt * 1e9, color='k', linestyle='-') # time in ns
ax[1].plot(outputdata_ave_db_moving, np.arange(outputdata_ave.shape[0]) * dt * 1e9, color='r', linestyle='--', alpha=0.7) # time in ns
ax[1].set_xlabel('Amplitude (dB)', size=14)
ax[1].set_ylabel('Time (ns)', size=14)
ax[1].set_ylim(outputdata_ave.shape[0] * dt * 1e9, 0) # time in ns
ax[1].tick_params(labelsize=12)
ax[1].grid()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'background_comparison.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'background_comparison.pdf'), format='pdf', dpi=300)
plt.show()