import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import signal
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import hilbert
import sys
import os

# gprMaxのルートディレクトリをパスに追加
gprmax_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, gprmax_root)

from tools.core.outputfiles_merge import get_output_data
from tqdm import tqdm
from tqdm.contrib import tenumerate


def shift(shift_time, original_sig, amplitude):
    shift_idx= int(shift_time * 1e-9 / dt)
    shifted_sig = np.roll(original_sig, shift_idx) * amplitude

    return shifted_sig

def env(data):
    return np.abs(signal.hilbert(data))


#* Define the function to plot the wave overlap
def plot(original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, time, envelope, peak_info, output_dir, original_peak_time, save_idx):
    time_for_plot = time * 1e9  # [ns]
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    ax.plot(time_for_plot, original_sig, label='Bottom component', linewidth=2)
    ax.plot(time_for_plot, shifted_sig - 3.0, label='Side component', linewidth=2)
    ax.plot(time_for_plot, overlapped_sig - 6.0, label='Overlapped', linewidth=2)
    ax.axvline(x=original_peak_time/1e-9, color='gray', linestyle='--', label='Original peak time')

    #* Plot the envelope and peaks
    ax.plot(time_for_plot, envelope - 6.0, label='Envelope', color='k', linestyle='-.')
    for i, info in enumerate(peak_info):
        if info['distinguishable'] == 'True':
            plt.plot(info['max_time']/1e-9, info['max_amplitude'] - 6.0, 'ro', label='Peak' if i == 0 else "")
        """
        plt.hlines(envelope[info['peak_idx']] / 2 - 6.0,
                            info['left_half_time'],
                            info['right_half_time'],
                            color='green', linestyle='--', label='FWHM' if i == 0 else "")
        """

    ax.set_xlim(-10, 10)
    ax.set_ylim(-8, 2)
    ax.set_title(r'$A = $' + f'{amplitude:.1f} ' + r'$\Delta t = ' + f'{shift_time:.1f}$ ns', fontsize=28)
    ax.set_xlabel('Time [ns]', fontsize=24)
    ax.set_ylabel('Normalized amplitude', fontsize=24)
    ax.tick_params(labelsize=20)
    # ax.legend(fontsize=20, loc='lower left')
    ax.grid()

    plt.savefig(f'{output_dir}/wave_overlap_{save_idx}.png')
    plt.close()


#* Define the function to plot the wave overlap for animation
def plot_for_animation(ax, original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, time, envelope, peak_info):
    time_for_plot = time * 1e9  # [ns]
    # アニメーション用のplot関数。既存のaxにプロットするのみ。保存やcloseはしない。
    ax.clear()
    ax.plot(time_for_plot, original_sig, label='Original', linewidth=2)
    ax.plot(time_for_plot, shifted_sig - 3.0, label=f'Shifted {shift_time:.1f} ns', linewidth=2)
    ax.plot(time_for_plot, overlapped_sig - 6.0, label='Overlapped', linewidth=2)

    #* Plot the envelope and peaks
    ax.plot(time_for_plot, envelope - 6.0, label='Envelope', color='k', linestyle='-.')
    for i, info in enumerate(peak_info):
        plt.plot(info['max_time'], info['max_amplitude'], 'ro', label='Peak' if i == 0 else "")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-8, 2)
    ax.set_title(f'Amplitude: {amplitude:.1f}', fontsize=24)
    ax.set_xlabel('Time [ns]', fontsize=24)
    ax.set_ylabel('Amplitude', fontsize=24)
    ax.tick_params(labelsize=20)
    #ax.legend(fontsize=20, loc='lower right')
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



#* Define the function to analyze the pulses
def analyze_pulses(data, dt, time):
    time_ns = time / 1e-9 # ns

    #* Calculate the envelope of the signal
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    #* Find the peaks
    peaks = []
    for i in range(1, len(data) - 1):
        if envelope[i - 1] < envelope[i] > envelope[i + 1] and envelope[i] > 0.1:
            peaks.append(i)


    # Calculate the half-width of the pulses
    pulse_info = []
    for i, peak_idx in enumerate(peaks):
        #peak_amplitude = np.abs(data[peak_idx])
        peak_amplitude = envelope[peak_idx]
        half_amplitude = peak_amplitude / 2

        # 左側の半値位置を探索
        left_idx = peak_idx
        while left_idx > 0 and envelope[left_idx] > half_amplitude:
            left_idx -= 1

        if left_idx == 0:
            left_half_time = time_ns[0]
        else:
            # 線形補間で正確な半値位置を求める
            left_slope = (envelope[left_idx + 1] - envelope[left_idx]) / (time_ns[left_idx + 1] - time_ns[left_idx])
            left_half_time = time_ns[left_idx] + (half_amplitude - envelope[left_idx]) / left_slope
        # 右側の半値位置を探索
        right_idx = peak_idx
        while right_idx < len(envelope) - 1 and envelope[right_idx] > half_amplitude:
            right_idx += 1

        if right_idx == len(envelope) - 1:
            right_half_time_ns = time_ns[-1]
        else:
            # 線形補間で正確な半値位置を求める
            right_slope = (envelope[right_idx] - envelope[right_idx - 1]) / (time_ns[right_idx] - time_ns[right_idx - 1])
            right_half_time = time[right_idx - 1] + (half_amplitude - envelope[right_idx - 1]) / right_slope

        # 半値全幅を計算
        hwhm = np.min([np.abs(time_ns[peak_idx] - left_half_time), np.abs(time_ns[peak_idx] - right_half_time)]) # [ns], Half width at half maximum
        fwhm = right_half_time - left_half_time # [ns], Full width at half maximum
        #width = right_half_time - left_half_time
        #width_half = hwhm

        # 次のピークとの時間差と判定
        if len(peaks) == 1:
            separation = None
            distinguishable = 'True'
        elif len(peaks) == 2:
            separation = time[peaks[1]] - time[peaks[0]]
            if separation >= fwhm:
                distinguishable = 'True'
            else:
                if i == 0:
                    if np.abs(envelope[peaks[0]]) >= np.abs(envelope[peaks[1]]): # envelopeの振幅で判定
                        distinguishable = 'True'
                    elif np.abs(envelope[peaks[0]]) < np.abs(envelope[peaks[1]]):
                        distinguishable = 'False'
                elif i == 1:
                    if np.abs(envelope[peaks[0]]) >= np.abs(envelope[peaks[1]]):
                        distinguishable = 'False'
                    elif np.abs(envelope[peaks[0]]) < np.abs(envelope[peaks[1]]):
                        distinguishable = 'True'


        # 範囲内での最大振幅とそのインデックスを取得
        hwhm_idx = int(hwhm / (dt / 1e-9)) # [ns]
        data_segment = data[peak_idx-hwhm_idx:peak_idx+hwhm_idx+1] # 半値全幅のデータ
        if len(data_segment) > 0:
            local_max_idx = np.argmax(np.abs(data_segment))
            #max_idx = left_time_idx + local_max_idx
            max_idx = peak_idx - hwhm_idx + local_max_idx
            #print(max_idx)
            max_time = time[max_idx]
            max_amplitude = data[max_idx]
        else:
            max_idx = peak_idx
            max_time = time[peak_idx]
            max_amplitude = data[peak_idx]

        pulse_info.append({
            'peak_idx': peak_idx,
            'peak_time': time[peak_idx],
            'peak_amplitude': peak_amplitude,
            'fwhm': fwhm,
            'hwhm': hwhm,
            'left_half_time': left_half_time,
            'right_half_time': right_half_time,
            'separation': separation,
            'distinguishable': distinguishable,
            'max_idx': max_idx,
            'max_time': max_time,
            'max_amplitude': max_amplitude
        })

    return envelope, pulse_info



#* Main part
# データの読み込み
data_path = input('データファイルのパスを入力してください（例：/path/to/direct.out）: ')

if not os.path.exists(data_path):
    print('指定されたファイルが存在しません。')
    sys.exit(1)

f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    original_sig, dt = get_output_data(data_path, (rx+1), 'Ez')
print('Data loaded successfully.')
print(f'Data length: {len(original_sig)} samples')
print(' ')

#* Define output directory
output_parent_dir = os.path.join(os.path.dirname(data_path), 'wave_overlap')
os.makedirs(output_parent_dir, exist_ok=True)


#t = np.arange(len(original_sig)) * dt / 1e-9  # 時間をナノ秒に変換
time_range = 10.0 # [ns]
time_array = np.arange(-time_range*1e-9, time_range*1e-9, dt)  # [s]
print('Length of time_array:', len(time_array))

# 作成したtに合わせてoriginal_sigを切り取り、ゼロ埋めする
signal_array = np.zeros(len(time_array))
signal_start_idx = int(len(time_array) / 2)
signal_normalize = original_sig / np.max(np.abs(original_sig))  # 正規化
simulation_time_range = len(original_sig) * dt / 1e-9  # [ns]
if simulation_time_range > time_range:
    signal_normalize_cut = signal_normalize[0:int(((time_range*1e-9 + dt) / dt))]
    signal_array[signal_start_idx:] = signal_normalize_cut
else:
    signal_array[signal_start_idx:signal_start_idx+len(signal_normalize)] = signal_normalize
print('Length of signal_array:', len(signal_array))

# Find peak time of the original signal
envelope_original, peak_info_original = analyze_pulses(signal_array, dt, time_array)
original_peak_time = peak_info_original[0]['max_time']
print(f'Original peak time: {original_peak_time:.3e} s')
# Save original peak info
peak_info_original_save = []
peak_info_original_save.append({
        'Peak time (envelope) [ns]': peak_info_original[0]['peak_time'],
        'Peak amplitude (envelope)': peak_info_original[0]['peak_amplitude'],
        'FWHM': peak_info_original[0]['fwhm'],
        'Distinguishable': peak_info_original[0]['distinguishable'],
        'Max amplitude': peak_info_original[0]['max_amplitude'],
        'Max time [ns]': peak_info_original[0]['max_time']
})
with open(output_parent_dir + f'/original_peak_info.txt', 'w') as f:
    for key, value in peak_info_original_save[0].items():
        f.write(f'{key}: {value}\n')


shift_times = np.arange(-3.0, 3.1, 0.1) # [ns]
amplitudes = np.arange(-2.0, 2.01, 0.2)


for amplitude in amplitudes:
    output_dir = os.path.join(output_parent_dir, f'{amplitude:.1f}')
    os.makedirs(output_dir, exist_ok=True)

    shifted_sigs = np.zeros((len(shift_times), len(signal_array)))
    overlapped_sigs = np.zeros((len(shift_times), len(signal_array)))

    for i, shift_time in tenumerate(shift_times, desc=f'Amplitude: {amplitude:.1f}'):
        shifted_sig = shift(shift_time, signal_array, amplitude)
        overlapped_sig = signal_array + shifted_sig

        shifted_sigs[i, :] = shifted_sig
        overlapped_sigs[i, :] = overlapped_sig

        #* detect peaks of the overlapped signal
        envelope_overlapped, peak_info_overlapped = analyze_pulses(overlapped_sig, dt, time_array)

        peak_info = []
        for info in peak_info_overlapped:
            peak_info.append({
                'Peak time (envelope) [ns]': info['peak_time'],
                'Peak amplitude (envelope)': info['peak_amplitude'],
                'FWHM': info['fwhm'],
                'Distinguishable': info['distinguishable'],
                'Max amplitude': info['max_amplitude'],
                'Max time [ns]': info['max_time']
            })
        output_dir_peak_info = os.path.join(output_dir, 'peak_info')
        os.makedirs(output_dir_peak_info, exist_ok=True)
        with open(output_dir_peak_info + f'/peak_info_{i}.txt', 'w') as f:
            for info_dict in peak_info:
                for key, value in info_dict.items():
                    f.write(f'{key}: {value}\n')
                f.write('\n')

        plot(signal_array, shifted_sig, overlapped_sig, shift_time, amplitude, time_array, envelope_overlapped, peak_info_overlapped, output_dir, original_peak_time, i)




