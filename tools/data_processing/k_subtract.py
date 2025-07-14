import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy.signal import hilbert
import sys
sys.path.append('tools')
from outputfiles_merge import get_output_data
from tqdm import tqdm
from tqdm.contrib import tenumerate
import argparse
import json



#* Detect first peak in transmmit signal
def detect_first_peak(transmmit_signal, dt):
    for i in range(1, len(transmmit_signal) - 1):
        if transmmit_signal[i - 1] < transmmit_signal[i] > transmmit_signal[i + 1] and transmmit_signal[i] > 1:
            first_peak_idx = i
            break
    first_peak_time = first_peak_idx * dt / 1e-9 # [ns]
    first_peak_amplitude = transmmit_signal[first_peak_idx]

    return first_peak_time, first_peak_amplitude



#* Calculate the two-way travel path length
def calc_TWT(boundary_model):
    #* Physical constants
    c0 = 299792458  # Speed of light in vacuum [m/s]


    optical_path_length = []
    boundary_names = []
    boundaries = boundary_model['boundaries']
    for boundary in boundaries:
        boundary_names.append(boundary['name'])
        if optical_path_length == []:
            optical_path_length.append(2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))
        else:
            optical_path_length.append(optical_path_length[-1] +  2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))


    #* Calculate the two-way travel time
    two_way_travel_time = [length / c0 /1e-9 for length in optical_path_length] # [ns]
    #* Add the initial pulse delay
    delay = boundary_model['initial_pulse_delay']# [ns]
    two_way_travel_time = [t + delay for t in two_way_travel_time]

    #print('Two-way travel time [ns]:')
    #print(two_way_travel_time)
    #print(' ')

    return two_way_travel_time



#* Subtract the transmmit signal from the A-scan
def subtract_signal(Ascan_data, transmmit_signal,dt,  TWT, reference_point_time, reference_point_amplitude):
    segment_start = TWT - 1.2 # [ns]
    segment_start_idx = int(segment_start * 1e-9 / dt)
    segment_end = TWT + 1.2 # [ns]
    segment_end_idx = int(segment_end * 1e-9 / dt)
    #print(f'Segment start: {segment_start} ns, {segment_start_idx}')
    #print(f'Segment end: {segment_end} ns, {segment_end_idx}')
    #print(' ')
    data_segment = Ascan_data[segment_start_idx: segment_end_idx]

    #* Detect the peak
    peak_times = []
    data_segment_abs = np.abs(data_segment)
    for i in range(1, len(data_segment) - 1):
        #if (np.abs(data_segment[i - 1]) < np.abs(data_segment[i]) > np.abs(data_segment[i + 1])) and (np.abs(data_segment[i]) > 1):
        if (data_segment_abs[i - 1] < data_segment_abs[i] > data_segment_abs[i + 1]):
            if data_segment_abs[i] > 1:
                peak_time = segment_start + i * dt / 1e-9 # [ns]
                peak_times.append(peak_time) # [ns]
                #print(data_segment[i])

    #print(f'Found {len(peak_times)} peaks')
    #print(f'TWT of the peaks: {peak_times}')
    #print(' ')

    #* Subtract the transmmit signal
    first_peak_time = peak_times[0] # [ns]
    first_peak_idx = int(first_peak_time * 1e-9 / dt) # [index]
    first_peak_amplitude = Ascan_data[first_peak_idx]
    #print(f'First peak time: {first_peak_time} ns')
    #print(f'First peak amplitude: {first_peak_amplitude}')
    #print(' ')

    #* Shift the transmmit signal to match the first peak
    time_shift = first_peak_time - reference_point_time # [ns]
    shift_idx = int(time_shift * 1e-9 / dt)
    shifted_transmmit_signal = np.roll(transmmit_signal, shift_idx)

    amp_ratio = first_peak_amplitude / reference_point_amplitude
    #print(f'Amplitude ratio: {amp_ratio}')
    shifted_transmmit_signal = shifted_transmmit_signal * amp_ratio

    #* Subtract the transmmit signal
    subtracted_data = Ascan_data - shifted_transmmit_signal

    return shifted_transmmit_signal, subtracted_data



def plot(original_data, shifted_data, subtracted_data, time, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, output_dir, TWT, plt_show):
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'),
                                figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

    #* Plot A-scan
    ax.plot(time, original_data, label='Original A-scan', color='k', linestyle='-')
    ax.plot(time, shifted_data, label='Shifted A-scan', color='r', linestyle='-.')
    ax.plot(time, subtracted_data, label='Subtracted A-scan', color='b', linestyle='-')


    plt.xlabel('Time [ns]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=24, loc='lower right')
    plt.tick_params(labelsize=24)
    plt.grid(True)


    #* for closeup option
    if closeup:
            ax.set_xlim([closeup_x_start, closeup_x_end])
            ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])


    #* Save the plot
    if closeup:
            fig.savefig(output_dir + '/subtracted_' + f'{TWT:.1f}' + '_closeup_x' + str(closeup_x_start) \
                    + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                    ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(output_dir + '/subtracted' + f'{TWT:.1f}' + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    if plt_show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    #* Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='k_subtract.py',
        description='subtranct the signal from the A-scan',
        epilog='End of help message',
        usage='python -m tools.k_plot_time_estimation [out_file] [model_json] [-closeup]',
    )
    parser.add_argument('out_file', help='Path to the .out file')
    parser.add_argument('model_json', help='Path to the model json file')
    parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
    args = parser.parse_args()


    #* Load the transmmit signal data
    transmmit_signal_path = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x6/20241111_polarity_v2/direct/A-scan/direct.out' # 送信波形データを読み込む

    f = h5py.File(transmmit_signal_path, 'r')
    nrx = f.attrs['nrx']
    for rx in range(nrx):
        transmmit_signal, dt = get_output_data(transmmit_signal_path, (rx+1), 'Ez')


    #* Load the A-scan data
    data_path = args.out_file

    f = h5py.File(data_path, 'r')
    nrx = f.attrs['nrx']
    for rx in range(nrx):
        data, dt = get_output_data(data_path, (rx+1), 'Ez')
    print(f'data length: {len(data)}')
    print(f'dt: {dt}')
    print(' ')


    #* Zero padding the transmmit signal
    if len(transmmit_signal) < len(data):
        transmmit_signal = np.pad(transmmit_signal, (0, len(data) - len(transmmit_signal)), 'constant')

    #* Define output directory
    output_dir = os.path.join(os.path.dirname(data_path), 'subtracted')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time = np.arange(len(data)) * dt  / 1e-9 # [ns]


    #* Load the model json file
    with open(args.model_json, 'r') as f:
        boundaries = json.load(f)


    #* Detect the first peak in the transmmit signal
    transmit_sig_first_peak_time, transmit_sig_first_peak_amp = detect_first_peak(transmmit_signal, dt)
    print(f'Transmit signal first peak time: {transmit_sig_first_peak_time} ns')
    print(f'Transmit signal first peak amplitude: {transmit_sig_first_peak_amp}')
    print(' ')

    #* Calculate the estimated two-way travel time
    TWTs = calc_TWT(boundaries)

    #* Subtract the transmmit signal from the A-scan
    closeup_y_start = -60
    closeup_y_end = 60

    for TWT in TWTs:
        shifted_data, subtracted_data = subtract_signal(data, transmmit_signal, dt, TWT, transmit_sig_first_peak_time, transmit_sig_first_peak_amp)

        if TWT > 3:
            closeup_x_start = TWT - 3
        else:
            closeup_x_start = 0
        closeup_x_end = TWT + 7

        #* Plot the subtracted signal
        plot(data, shifted_data, subtracted_data, time, args.closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, output_dir, TWT, plt_show=True)