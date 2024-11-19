import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy.signal import hilbert



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_detect_peak.py',
    description='Detect the peak from the B-scan data',
    epilog='End of help message',
    usage='python -m tools.k_detect_peak [out_file] [-closeup]',
)
parser.add_argument('out_file', help='Path to the .out file')
parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
args = parser.parse_args()


#* Load the A-scan data
data_path = args.out_file
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')

#time = np.arange(len(data)) * dt


#* Define the function to analyze the pulses
def analyze_pulses(data, dt, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end):
    time = np.arange(len(data)) * dt

    #* Calculate the envelope of the signal
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    #* Find the peaks
    peaks = []
    for i in tqdm(range(1, len(data) - 1)):
        if envelope[i - 1] < envelope[i] > envelope[i + 1] and np.abs(data[i]) > 1:
            peaks.append(i)

    # Calculate the half-width of the pulses
    pulse_info = []
    for i, peak_idx in enumerate(peaks):
        peak_amplitude = data[peak_idx]
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
        width = right_half_time - left_half_time
        width_half = width / 2

        # 次のピークとの時間差と判定
        if i < len(peaks) - 1:
            next_peak_idx = peaks[i + 1]
            separation = time[next_peak_idx] - time[peak_idx]
            distinguishable = separation >= width_half
        else:
            separation = None
            distinguishable = None

        # 半値全幅内でのA-scanデータの最大値を探す
        # 時間範囲をインデックス範囲に変換
        left_time_idx = np.searchsorted(time, left_half_time)
        right_time_idx = np.searchsorted(time, right_half_time)

        # 範囲内での最大振幅とそのインデックスを取得
        data_segment = data[left_time_idx:right_time_idx+1]
        if len(data_segment) > 0:
            local_max_idx = np.argmax(np.abs(data_segment))
            max_idx = left_time_idx + local_max_idx
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
            'width': width,
            'width_half': width_half,
            'left_half_time': left_half_time,
            'right_half_time': right_half_time,
            'separation': separation,
            'distinguishable': distinguishable,
            'max_idx': max_idx,
            'max_time': max_time,
            'max_amplitude': max_amplitude
        })


    #* Plot A-scan
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'), num='rx' + str(rx),
                            figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
    ax.plot(time, data, label='A-scan', color='black')
    ax.plot(time, envelope, label='Envelope', color='blue', linestyle='-.')

    for i, info in enumerate(pulse_info):
        peak_time = info['peak_time']
        plt.plot(info['max_time'], info['max_amplitude'], 'ro')

        # 半値全幅を描画
        if np.abs(info['peak_amplitude']) > 1:
            plt.hlines(envelope[info['peak_idx']] / 2,
                    info['left_half_time'],
                    info['right_half_time'],
                    color='green', linestyle='--', label='FWHM' if i == 0 else "")


    plt.xlabel('Time [ns]', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    plt.grid(True)

    if args.closeup:
        ax.set_xlim([closeup_x_start * 1e-9, closeup_x_end * 1e-9])
        ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])
    plt.show()

    return pulse_info

# 使用例
if __name__ == "__main__":
    # for closeup option
    closeup_x_start = 0 #[ns]
    closeup_x_end =100 #[ns]
    closeup_y_start = -60
    closeup_y_end = 60


    #* Run the pulse analysis
    pulse_info = analyze_pulses(data, dt, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end)


    # 結果の表示
    for info in pulse_info:
        print(f"Peak at {info['peak_time']/1e-9:.4f} ns: Width={info['width']/1e-9:.4f} ns, Distinguishable={info['distinguishable']}")

