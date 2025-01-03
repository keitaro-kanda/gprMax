import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy.signal import hilbert



#* Define the function to analyze the pulses
def detect_plot_peaks(data, dt, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, FWHM, output_dir, plt_show):
    time = np.arange(len(data)) * dt  / 1e-9 # ns

    #* Calculate the envelope of the signal
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    #* Find the peaks
    peaks = []
    for i in tqdm(range(1, len(data) - 1)):
        if envelope[i - 1] < envelope[i] > envelope[i + 1] and envelope[i] > 1:
            peaks.append(i)
    print(f'Found {len(peaks)} peaks')


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
        fwhm = hwhm * 2 # [ns], Full width at half maximum

        # 次のピークとの時間差と判定
        
        if i < len(peaks) - 1:
            next_peak_idx = peaks[i + 1]
            separation = time[next_peak_idx] - time[peak_idx]
            distinguishable = separation >= hwhm
        elif i == len(peaks) - 1:
            former_peak_idx = peaks[i - 1]
            separation = time[peak_idx] - time[former_peak_idx]
            distinguishable = separation >= hwhm
        else:
            separation = None
            distinguishable = None


        # 範囲内での最大振幅とそのインデックスを取得
        hwhm_idx = int(hwhm / (dt / 1e-9)) # [ns]
        data_segment = data[peak_idx-hwhm_idx:peak_idx+hwhm_idx+1] # 半値全幅のデータ
        if len(data_segment) > 0:
            local_max_idx = np.argmax(np.abs(data_segment))
            #* FWHMの両端をピークとしてしまうことを避ける
            if local_max_idx in [1, len(data_segment) - 2]:
                continue
            if local_max_idx == 0 or local_max_idx == len(data_segment) - 1:
                local_max_idxs = []
                local_max_amps = []
                for j in range(1, len(data_segment) - 1):
                    if data_segment[j - 1] < data_segment[j] > data_segment[j + 1]:
                        local_max_idxs.append(j)
                        local_max_amps.append(data_segment[j])
                if len(local_max_idxs) > 0:
                    local_max_idx = local_max_idxs[np.argmax(local_max_amps)]
            max_idx = peak_idx - hwhm_idx + local_max_idx
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
            'width': fwhm,
            'width_half': hwhm,
            'left_half_time': left_half_time,
            'right_half_time': right_half_time,
            'separation': separation,
            'distinguishable': distinguishable,
            'max_idx': max_idx,
            'max_time': max_time,
            'max_amplitude': max_amplitude
        })


    #* Plot A-scan
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'),
                            figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
    ax.plot(time, data, label='A-scan', color='black')
    ax.plot(time, envelope, label='Envelope', color='blue', linestyle='-.')

    for i, info in enumerate(pulse_info):
        if info['distinguishable']:
            plt.plot(info['max_time'], info['max_amplitude'], 'ro', label='Peak' if i == 0 else "")

        # 半値全幅を描画
        if FWHM:
            if info['distinguishable'] or len(peaks) == 1:
                plt.hlines(envelope[info['peak_idx']] / 2,
                        info['left_half_time'],
                        info['right_half_time'],
                        color='green', linestyle='--', label='FWHM' if i == 0 else "")
        else:
            continue


    plt.xlabel('Time [ns]', fontsize=24)
    plt.ylabel('Amplitude', fontsize=24)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=20, loc='lower right')
    plt.tick_params(labelsize=20)
    plt.grid(True)

    if FWHM:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])

    #* Save the plot
    if closeup:
        plt.xlim([closeup_x_start, closeup_x_end])
        plt.ylim([closeup_y_start, closeup_y_end])

        fig.savefig(output_dir + '/peak_detection' + '_closeup_x' + str(closeup_x_start) \
                + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(output_dir + '/peak_detection' + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    if plt_show:
        plt.show()
    else:
        plt.close()

    return pulse_info

# 使用例
if __name__ == "__main__":
    #* Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='k_detect_peak.py',
        description='Detect the peak from the B-scan data',
        epilog='End of help message',
        usage='python -m tools.k_detect_peak [out_file] [-closeup] [-FWHM]',
    )
    parser.add_argument('out_file', help='Path to the .out file')
    parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
    parser.add_argument('-FWHM', action='store_true', help='Plot the FWHM')
    args = parser.parse_args()


    #* Define path
    data_path = args.out_file
    output_dir = os.path.join(os.path.dirname(data_path), 'peak_detection')
    os.makedirs(output_dir, exist_ok=True)


    #* Load the A-scan data
    f = h5py.File(data_path, 'r')
    nrx = f.attrs['nrx']
    for rx in range(nrx):
        data, dt = get_output_data(data_path, (rx+1), 'Ez')

    time = np.arange(len(data)) * dt


    # for closeup option
    closeup_x_start = 0 #[ns]
    closeup_x_end = 100 #[ns]
    closeup_y_start = -60
    closeup_y_end = 60

    #* Run the pulse analysis
    pulse_info = detect_plot_peaks(data, dt, args.closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, args.FWHM, output_dir, plt_show=True)


    # 結果の表示
    for info in pulse_info:
        print(f"Peak at {info['peak_time']:.4f} ns: Width={info['width']:.4f} ns, Distinguishable={info['distinguishable']}")

    #* Save the pulse information
    filename = os.path.join(output_dir, 'peak_info.txt')
    peak_info = []
    for info in pulse_info:
        peak_info.append({
            'Peak time (envelope) [ns]': info['peak_time'],
            'Peak amplitude (envelope)': info['peak_amplitude'],
            'Distinguishable': info['distinguishable'],
            'Max amplitude': info['max_amplitude'],
            'Max time [ns]': info['max_time']
        })
    np.savetxt(filename, peak_info, delimiter=' ', fmt='%s')

