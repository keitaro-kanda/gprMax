import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from tools.core.outputfiles_merge import get_output_data
from scipy.signal import hilbert

def detect_peaks(data, dt):
    """
    Detects peaks in A-scan data. This is the core logic function.

    Args:
        data (np.ndarray): Amplitude data of the A-scan.
        dt (float): Time step of the simulation.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about a detected peak.
    """
    data_norm = data / np.amax(np.abs(data)) # Normalize the data
    time = np.arange(len(data_norm)) * dt  / 1e-9 # [ns]

    analytic_signal = hilbert(data_norm)
    envelope = np.abs(analytic_signal)
    evnvelope_moving_average = np.convolve(envelope, np.ones(10)/10, mode='same')

    peaks = []
    for i in range(1, len(data_norm) - 1):
        if evnvelope_moving_average[i - 1] < evnvelope_moving_average[i] > evnvelope_moving_average[i + 1] and evnvelope_moving_average[i] > 0.5e-3:
            peaks.append(i)

    pulse_info = []
    for i, peak_idx in enumerate(peaks):
        peak_amplitude = envelope[peak_idx]
        half_amplitude = peak_amplitude / 2

        left_idx = peak_idx
        while left_idx > 0 and envelope[left_idx] > half_amplitude:
            left_idx -= 1

        if left_idx == 0:
            left_half_time = time[0]
        else:
            left_slope = (envelope[left_idx + 1] - envelope[left_idx]) / (time[left_idx + 1] - time[left_idx])
            left_half_time = time[left_idx] + (half_amplitude - envelope[left_idx]) / left_slope

        right_idx = peak_idx
        while right_idx < len(envelope) - 1 and envelope[right_idx] > half_amplitude:
            right_idx += 1

        if right_idx == len(envelope) - 1:
            right_half_time = time[-1]
        else:
            right_slope = (envelope[right_idx] - envelope[right_idx - 1]) / (time[right_idx] - time[right_idx - 1])
            right_half_time = time[right_idx - 1] + (half_amplitude - envelope[right_idx - 1]) / right_slope

        fwhm = right_half_time - left_half_time
        hwhm = np.min([np.abs(time[peak_idx] - left_half_time), np.abs(time[peak_idx] - right_half_time)])

        separation_prev = time[peak_idx] - time[peaks[i - 1]] if i > 0 else None
        separation_next = time[peaks[i + 1]] - time[peak_idx] if i < len(peaks) - 1 else None

        distinguishable = True
        if separation_prev is not None and separation_prev < fwhm:
            distinguishable = False
        if separation_next is not None and separation_next < fwhm:
            distinguishable = False

        data_segment = np.abs(data_norm[int(left_half_time*1e-9/dt):int(right_half_time*1e-9/dt)])
        max_idx = peak_idx
        max_time = time[peak_idx]
        max_amplitude = data_norm[peak_idx]
        if len(data_segment) > 0:
            primary_max_idx = np.argmax(np.abs(data_segment))
            max_idx = int(left_half_time*1e-9/dt) + primary_max_idx
            max_time = time[max_idx]
            max_amplitude = data_norm[max_idx]

        pulse_info.append({
            'peak_idx': peak_idx,
            'peak_time': time[peak_idx],
            'peak_amplitude': peak_amplitude,
            'width': fwhm,
            'width_half': hwhm,
            'left_half_time': left_half_time,
            'right_half_time': right_half_time,
            'separation': min(separation_prev or np.inf, separation_next or np.inf),
            'distinguishable': distinguishable,
            'max_idx': max_idx,
            'max_time': max_time,
            'max_amplitude': max_amplitude,
        })
    return pulse_info

#* Define the function to analyze the pulses
def detect_plot_peaks(data, dt, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, FWHM, output_dir, plt_show):
    pulse_info = detect_peaks(data, dt)
    data_norm = data / np.amax(np.abs(data))
    time = np.arange(len(data_norm)) * dt  / 1e-9 # [ns]
    analytic_signal = hilbert(data_norm)
    envelope = np.abs(analytic_signal)

    #* Plot A-scan
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'),
                            figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
    ax.plot(time, data_norm, label='A-scan', color='black')
    ax.plot(time, envelope, label='Envelope', color='blue', linestyle='-.')

    for i, info in enumerate(pulse_info):
        if info['distinguishable']==True:
            plt.plot(info['max_time'], info['max_amplitude'], 'ro', label='Primary Peak' if i == 0 else "", markersize=10)

        if FWHM:
            if info['distinguishable'] or len(pulse_info) == 1:
                plt.hlines(info['peak_amplitude'] / 2,
                        info['left_half_time'],
                        info['right_half_time'],
                        color='green', linestyle='--', label='FWHM' if i == 0 else "")

    plt.xlabel('Time [ns]', fontsize=24)
    plt.ylabel('Amplitude', fontsize=24)
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid(True)

    if FWHM or closeup:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])

    if output_dir:
        if closeup:
            fig.savefig(os.path.join(output_dir, f'peak_detection_closeup_x{closeup_x_start}_{closeup_x_end}_y{closeup_y_start}_{closeup_y_end}.png'),
                        dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(os.path.join(output_dir, 'peak_detection.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

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
    closeup_x_start = 20 #[ns]
    closeup_x_end = 40 #[ns]
    closeup_y_start = -3e11
    closeup_y_end = 3e11

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

