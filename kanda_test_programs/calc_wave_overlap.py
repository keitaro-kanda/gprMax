import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import signal
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import hilbert
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
def plot(original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, time, envelope, peak_info, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    ax.plot(time, original_sig, label='Bottom component', linewidth=2)
    ax.plot(time, shifted_sig - 3.0, label='Side component', linewidth=2)
    ax.plot(time, overlapped_sig - 6.0, label='Overlapped', linewidth=2)

    #* Plot the envelope and peaks
    ax.plot(time, envelope - 6.0, label='Envelope', color='k', linestyle='-.')
    for i, info in enumerate(peak_info):
        if info['distinguishable'] == 'True':
            plt.plot(info['max_time'], info['max_amplitude'] - 6.0, 'ro', label='Peak' if i == 0 else "")
        """
        plt.hlines(envelope[info['peak_idx']] / 2 - 6.0,
                            info['left_half_time'],
                            info['right_half_time'],
                            color='green', linestyle='--', label='FWHM' if i == 0 else "")
        """

    ax.set_xlim(0, 20)
    ax.set_ylim(-8, 1)
    ax.set_title(r'$A = $' + f'{amplitude:.1f} ' + r'$\Delta t = ' + f'{shift_time:.1f}$ ns', fontsize=28)
    ax.set_xlabel('Time [ns]', fontsize=24)
    ax.set_ylabel('Normalized amplitude', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20, loc='lower right')
    ax.grid()

    plt.savefig(f'{output_dir}/wave_overlap_{shift_time:.1f}ns_{amplitude:.1f}.png')
    plt.close()


#* Define the function to plot the wave overlap for animation
def plot_for_animation(ax, original_sig, shifted_sig, overlapped_sig, shift_time, amplitude, time, envelope, peak_info):
    # アニメーション用のplot関数。既存のaxにプロットするのみ。保存やcloseはしない。
    ax.clear()
    ax.plot(time, original_sig, label='Original', linewidth=2)
    ax.plot(time, shifted_sig - 3.0, label=f'Shifted {shift_time:.1f} ns', linewidth=2)
    ax.plot(time, overlapped_sig - 6.0, label='Overlapped', linewidth=2)

    #* Plot the envelope and peaks
    ax.plot(time, envelope, label='Envelope', color='k', linestyle='-.')
    for i, info in enumerate(peak_info):
        plt.plot(info['max_time'], info['max_amplitude'], 'ro', label='Peak' if i == 0 else "")

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



#* Define the function to analyze the pulses
def analyze_pulses(data, dt):
    time = np.arange(len(data)) * dt  / 1e-9 # ns

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
            left_half_time = time[0]
        else:
            # 線形補間で正確な半値位置を求める
            left_slope = (envelope[left_idx + 1] - envelope[left_idx]) / (time[left_idx + 1] - time[left_idx])
            left_half_time = time[left_idx] + (half_amplitude - envelope[left_idx]) / left_slope

        # 右側の半値位置を探索
        right_idx = peak_idx
        while right_idx < len(envelope) - 1 and envelope[right_idx] > half_amplitude:
            right_idx += 1

        if right_idx == len(envelope) - 1:
            right_half_time = time[-1]
        else:
            # 線形補間で正確な半値位置を求める
            right_slope = (envelope[right_idx] - envelope[right_idx - 1]) / (time[right_idx] - time[right_idx - 1])
            right_half_time = time[right_idx - 1] + (half_amplitude - envelope[right_idx - 1]) / right_slope

        # 半値全幅を計算
        hwhm = np.min([np.abs(time[peak_idx] - left_half_time), np.abs(time[peak_idx] - right_half_time)]) # [ns], Half width at half maximum
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
t = np.arange(len(original_sig)) * dt / 1e-9  # 時間をナノ秒に変換
sig = original_sig / np.max(np.abs(original_sig))  # 正規化


shift_times = np.arange(0, 5.02, 0.1) # [ns]
#amplitudes = np.arange(-2.0, 2.01, 0.2)
amplitudes = (0.8, -0.8)


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

        #* detect peaks of the overlapped signal
        envelope_overlapped, peak_info_overlapped = analyze_pulses(overlapped_sig, dt)

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
        np.savetxt(output_dir_peak_info + f'/{shift_time:.1f}ns_{amplitude:.1f}.txt', peak_info, delimiter=' ', fmt='%s')

        plot(sig, shifted_sig, overlapped_sig, shift_time, amplitude, t, envelope_overlapped, peak_info_overlapped, output_dir)

    #* Make animation
    #fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    #anim_func = Animation(ax, sig, shifted_sigs, overlapped_sigs, shift_times, amplitude, t, output_dir)
    #ani = FuncAnimation(fig, anim_func, frames=len(shift_times), interval=1000)
    #writer = FFMpegWriter(fps=5)
    #ani.save(f'{output_dir}/animation.mp4', writer=writer)
    #plt.close(fig)




