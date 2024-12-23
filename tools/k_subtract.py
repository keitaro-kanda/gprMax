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


#* Physical constants
c0 = 299792458  # Speed of light in vacuum [m/s]



#* Detect first peak in transmmit signal
def detect_first_peak(transmmit_signal):
    for i in range(1, len(transmmit_signal) - 1):
        if transmmit_signal[i - 1] < transmmit_signal[i] > transmmit_signal[i + 1]:
            first_peak_idx = i
            break
    first_peak_time = first_peak_idx * dt # [ns]
    first_peak_amplitude = transmmit_signal[first_peak_idx]

    return first_peak_time, first_peak_amplitude



#* Calculate the two-way travel path length
def calc_TWT(boundary_model):
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

    print('Two-way travel time [ns]:')
    print(two_way_travel_time)
    print(' ')

    return two_way_travel_time



#* Subtract the transmmit signal from the A-scan
def subtract_signal(data, transmmit_signal, TWT, reference_point_time, reference_point_amplitude):
    segment_start = TWT - 5 # [ns]
    segment_end = TWT + 5 # [ns]
    print(f'Segment start: {segment_start} ns')
    print(f'Segment end: {segment_end} ns')
    data_segment = data[int(segment_start / dt): int(segment_end / dt)]
    print(f'Segment length: {len(data_segment)}')

    #* Detect the peak
    peak_times = []
    for i in tqdm(range(1, len(data_segment) - 1)):
        if np.abs(data_segment[i - 1]) < np.abs(data_segment[i]) > np.abs(data_segment[i + 1]) and np.abs(data_segment[i] > 1):
            peak_time = segment_start + i * dt
            peak_times.append(peak_time)

    print(f'Found {len(peak_times)} peaks')
    print(f'TWT of the peaks: {peak_times}')

    #* Subtract the transmmit signal
    first_peak_time = peak_times[0]
    first_peak_idx = int(first_peak_time / dt)
    first_peak_amplitude = data[first_peak_idx]

    #* Shift the transmmit signal to match the first peak
    time_shift = first_peak_time - reference_point_time
    shift_idx = int(time_shift / dt)
    shifted_transmmit_signal = np.roll(transmmit_signal, shift_idx)

    amp_ratio = first_peak_amplitude / reference_point_amplitude
    shifted_transmmit_signal = shifted_transmmit_signal * amp_ratio

    #* Subtract the transmmit signal
    if first_peak_amplitude > 0:
        subtracted_data = data - shifted_transmmit_signal
    else:
        subtracted_data = data + shifted_transmmit_signal

    return subtracted_data



if __name__ == '__main__':
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

    #* Define output directory
    output_dir = os.path.dirname(data_path)

    time = np.arange(len(data)) * dt  / 1e-9 # [ns]


    #* Load the model json file
    with open(args.model_json, 'r') as f:
        boundaries = json.load(f)


    #* Detect the first peak in the transmmit signal
    transmit_sig_first_peak_time, transmit_sig_first_peak_amp = detect_first_peak(transmmit_signal)

    #* Calculate the estimated two-way travel time
    TWTs = calc_TWT(boundaries)

    #* Subtract the transmmit signal from the A-scan
    subtracted_data = subtract_signal(data, transmmit_signal, TWTs[2], transmit_sig_first_peak_time, transmit_sig_first_peak_amp)



    #* Plot
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'), num='rx' + str(rx),
                            figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

    #* Plot A-scan
    ax.plot(time, data, label='A-scan', color='gray', linestyle='--')
    ax.plot(time, subtracted_data, label='Subtracted A-scan', color='black')


    plt.xlabel('Time [ns]', fontsize=24)
    plt.ylabel('Amplitude', fontsize=24)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=20, loc='lower right')
    plt.tick_params(labelsize=20)
    plt.grid(True)


    #* for closeup option
    closeup_x_start = 0 #[ns]
    closeup_x_end =100 #[ns]
    closeup_y_start = -60
    closeup_y_end = 60

    if args.closeup:
            ax.set_xlim([closeup_x_start, closeup_x_end])
            ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])


    #* Save the plot
    if args.closeup:
            fig.savefig(output_dir + '/subtracted' + '_closeup_x' + str(closeup_x_start) \
                    + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                    ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(output_dir + '/subtracted' + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    plt.show()