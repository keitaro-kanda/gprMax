import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import signal
from matplotlib.animation import FuncAnimation, FFMpegWriter

import sys
sys.path.append('tools')
from outputfiles_merge import get_output_data

# データの読み込み
data_path = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/20240824_echo_shape_test/direct_wave/direct.out'
output_dir = os.path.join(os.path.dirname(data_path), 'phase_shift')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    sig, dt = get_output_data(data_path, (rx+1), 'Ez')
print('data shape: ', sig.shape)

t = np.arange(len(sig)) * dt / 1e-9  # 時間をナノ秒に変換
sig = sig / np.max(np.abs(sig))  # 正規化

def shift(shift_time):
    shift_samples = int(shift_time * 1e-9 / dt)
    shifted_sig = -np.roll(sig, shift_samples) * 0.7
    return shifted_sig

def env(data):
    return np.abs(signal.hilbert(data))

# 時間遅延のリスト
time_lag_list = np.arange(0, 5.05, 0.05)  # [ns]

# プロットの準備
fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
line1, = ax.plot([], [], label='original')
line2, = ax.plot([], [], label='shifted')
line3, = ax.plot([], [], label='overlapped')

ax.set_xlim(0, 25)  # [ns]
ax.set_ylim(-7.5, 2.5)
ax.grid()
ax.legend(fontsize=16, loc='upper right')
ax.tick_params(labelsize=16)
fig.supxlabel('Time [ns]', fontsize=20)
fig.supylabel('Normalized amplitude', fontsize=20)

# アニメーション関数
def animate(i):
    time_lag = time_lag_list[i]
    shifted = shift(time_lag)
    overlapped = sig + shifted

    plot_list = [sig, shifted, overlapped]
    for idx, line in enumerate([line1, line2, line3]):
        line.set_data(t, plot_list[idx] - 2.5 * idx)
    ax.set_title(f'Time lag: {time_lag:.2f} ns', fontsize=16)
    title_artist = ax.title
    return line1, line2, line3, title_artist

# アニメーションの作成
ani = FuncAnimation(fig, animate, frames=len(time_lag_list), blit=True, repeat=False)

# 動画の保存（FFmpegが必要）
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save(os.path.join(output_dir, 'wave_overlap_animation.mp4'), writer=writer)

plt.show()
