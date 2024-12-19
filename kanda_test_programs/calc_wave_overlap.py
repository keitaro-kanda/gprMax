import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import signal
from matplotlib.animation import FuncAnimation, FFMpegWriter
import sys
sys.path.append('tools')
from outputfiles_merge import get_output_data
from tqdm import tqdm
from tqdm.contrib import tenumerate





# データの読み込み
data_path = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x6/20241111_polarity_v2/direct/A-scan/direct.out' # 送信波形データを読み込む

f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    original_sig, dt = get_output_data(data_path, (rx+1), 'Ez')


#* Define output directory
output_parent_dir = '/Volumes/SSD_Kanda_BUFFALO/gprMax/wave_overlap'



def shift(shift_time, original_sig, amplitude):
    shift_idx= int(shift_time * 1e-9 / dt)
    shifted_sig = np.roll(original_sig, shift_idx) * amplitude

    return shifted_sig

def env(data):
    return np.abs(signal.hilbert(data))


#* Define the function to plot the wave overlap
def plot(original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, time, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    ax.plot(time, original_sig, label='Original', linewidth=2)
    ax.plot(time, shifted_sig - 3.0, label=f'Shifted {shift_time:.1f} ns', linewidth=2)
    ax.plot(time, overlapped_sig - 6.0, label='Overlapped', linewidth=2)

    ax.set_xlim(0, 20)
    ax.set_ylim(-8, 1)
    ax.set_title(f'Amplitude: {amplitude:.1f}', fontsize=24)
    ax.set_xlabel('Time [ns]', fontsize=24)
    ax.set_ylabel('Amplitude', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20, loc='lower right')
    ax.grid()

    plt.savefig(f'{output_dir}/wave_overlap_{shift_time:.1f}ns_{amplitude:.1f}.png')
    plt.close()


#* Define the function to plot the wave overlap for animation
def plot_for_animation(ax, original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, time):
    # アニメーション用のplot関数。既存のaxにプロットするのみ。保存やcloseはしない。
    ax.clear()
    ax.plot(time, original_sig, label='Original', linewidth=2)
    ax.plot(time, shifted_sig - 3.0, label=f'Shifted {shift_time:.1f} ns', linewidth=2)
    ax.plot(time, overlapped_sig - 6.0, label='Overlapped', linewidth=2)

    ax.set_xlim(0, 20)
    ax.set_ylim(-8, 1)
    ax.set_title(f'Amplitude: {amplitude:.1f}', fontsize=24)
    ax.set_xlabel('Time [ns]', fontsize=24)
    ax.set_ylabel('Amplitude', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20, loc='lower right')
    ax.grid()


class Animation:
    def __init__(self, ax, original_sig, shifted_sigs, overlapped_sigs, shift_times, amplitude, time, output_dir):
        self.ax = ax
        self.original_sig = original_sig
        self.shifted_sigs = shifted_sigs
        self.overlapped_sigs = overlapped_sigs
        self.shift_times = shift_times
        self.amplitude = amplitude
        self.time = time
        self.output_dir = output_dir

    def __call__(self, i):
        self.ax.clear()  # アニメーション用にクリア
        shift_time = self.shift_times[i]
        amplitude = self.amplitude
        shifted_sig = self.shifted_sigs[i, :]
        overlapped_sig = self.overlapped_sigs[i, :]

        # axを渡すplot_for_animationを想定
        plot_for_animation(self.ax, self.original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, self.time)
        # 保存はアニメーションループ内では不要（必要なら外で実行）




#* Main part
t = np.arange(len(original_sig)) * dt / 1e-9  # 時間をナノ秒に変換
sig = original_sig / np.max(np.abs(original_sig))  # 正規化


shift_times = np.arange(0, 5.02, 0.2) # [ns]
amplitudes = np.arange(-1.0, 1.2, 0.2)


for amplitude in amplitudes:
    output_dir = os.path.join(output_parent_dir, f'{amplitude:.1f}')
    os.makedirs(output_dir, exist_ok=True)

    shifted_sigs = np.zeros((len(shift_times), len(sig)))
    overlapped_sigs = np.zeros((len(shift_times), len(sig)))

    for i, shift_time in tenumerate(shift_times, desc=f'Amplitude: {amplitude:.1f}'):
        shifted_sig = shift(shift_time, sig, amplitude)
        overlapped_sig = sig + shifted_sig

        shifted_sigs[i, :] = shifted_sig
        overlapped_sigs[i, :] = overlapped_sig

        plot(sig, shifted_sig, overlapped_sig, shift_time, amplitude, t, output_dir)

    #* Make animation
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    anim_func = Animation(ax, sig, shifted_sigs, overlapped_sigs, shift_times, amplitude, t, output_dir)
    ani = FuncAnimation(fig, anim_func, frames=len(shift_times), interval=1000)
    writer = FFMpegWriter(fps=5)
    ani.save(f'{output_dir}/animation.mp4', writer=writer)
    plt.close(fig)




