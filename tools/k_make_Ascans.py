import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy.signal import hilbert
import json


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_make_Ascans.py',
    description='Make normal A-scan, A-scan with peak detection, and A-scan with estimated two-way travel time',
    epilog='End of help message',
    usage='python -m tools.k_detect_peak [json] [-closeup] [-FWHM]',
)
parser.add_argument('json', help='Path to the json file')
parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
parser.add_argument('-FWHM', action='store_true', help='Plot the FWHM')
args = parser.parse_args()



#* Import json
with open(args.json, 'r') as f:
    path_group = json.load(f)



#* Define function to plot the A-scan
def plot_Ascan(filename, data, time, rx, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, outputtext):
    for rx in range(1, nrx + 1):
        fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
        line = ax.plot(time, data, 'k', lw=2)

        if args.closeup:
            ax.set_xlim([closeup_x_start, closeup_x_end])
            ax.set_ylim([closeup_y_start, closeup_y_end])
        else:
            ax.set_xlim([0, np.amax(time)])

        ax.grid(which='both', axis='both', linestyle='-.')
        ax.minorticks_on()
        ax.set_xlabel('Time [ns]', fontsize=28)
        ax.set_ylabel(outputtext + ' field strength [V/m]', fontsize=28)
        ax.tick_params(labelsize=24)
        plt.tight_layout()


        if args.closeup:
            fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '_closeup_x' + str(closeup_x_start) \
                            + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                            ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


#*
def analyze_pulses(data, time, rx, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end):

    #* Calculate the envelope of the signal
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    #* Find the peaks
    peaks = []
    for i in tqdm(range(1, len(data) - 1)):
        if envelope[i - 1] < envelope[i] > envelope[i + 1] and envelope[i] > 1:
            peaks.append(i)
    #print(f'Found {len(peaks)} peaks')


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
        width = right_half_time - left_half_time
        width_half = width / 2

        # 次のピークとの時間差と判定
        if i < len(peaks) - 1:
            next_peak_idx = peaks[i + 1]
            separation = time[next_peak_idx] - time[peak_idx]
            distinguishable = separation >= width_half
            #distinguishable = separation >= right_half_time - time[peak_idx]
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
    ax.plot(time, data, label='A-scan', color='black', lw=2)
    ax.plot(time, envelope, label='Envelope', color='blue', linestyle='-.', lw=2)

    for i, info in enumerate(pulse_info):
        peak_time = info['peak_time']
        plt.plot(info['max_time'], info['max_amplitude'], 'ro', label='Peak' if i == 0 else "")

        # 半値全幅を描画
        if args.FWHM:
            if info['distinguishable']:
                plt.hlines(envelope[info['peak_idx']] / 2,
                        info['left_half_time'],
                        info['right_half_time'],
                        color='green', linestyle='--', label='FWHM' if i == 0 else "")
        else:
            continue


    plt.xlabel('Time [ns]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=24, loc='lower right')
    plt.tick_params(labelsize=24)
    plt.grid(True)

    if args.closeup:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])

    #* Save the plot
    if args.closeup:
            fig.savefig(output_dir + '/peaks_rx' + str(rx+1) + '_closeup_x' + str(closeup_x_start) \
                    + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                    ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(output_dir + '/peaks_rx' + str(rx+1) + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)



#* Define function to plot A-scan with estimated two-way travel time
def plot_Ascan_estimated_time(data, time, model_path, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, outputtext, rx=1):
    #* Physical constants
    c0 = 299792458  # Speed of light in vacuum [m/s]

    #* Load the model json
    with open(model_path, 'r') as f:
        model = json.load(f)

    #* Calculate the two-way travel path length
    optical_path_length = []
    boundary_names = []
    boundaries = model['boundaries']
    for boundary in boundaries:
        boundary_names.append(boundary['name'])
        if optical_path_length == []:
            optical_path_length.append(2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))
        else:
            optical_path_length.append(optical_path_length[-1] +  2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))

    #* Calculate the two-way travel time
    two_way_travel_time = [length / c0 /1e-9 for length in optical_path_length] # [ns]
    #* Add the initial pulse delay
    delay = model['initial_pulse_delay']# [ns]
    two_way_travel_time = [t + delay for t in two_way_travel_time]

    #* Save the two-way travel time as txt
    np.savetxt(output_dir + '/delay_time.txt', two_way_travel_time, fmt='%.6f', delimiter=' ', header='Two-way travel time [ns]')

    #* Plot
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'), num='rx' + str(rx),
                                figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

    #* Plot A-scan
    ax.plot(time, data, label='A-scan', color='black', lw=2)

    #* Plot the estimated two-way travel time
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, t in enumerate(two_way_travel_time):
        label = ['vacuum-regolith', 'regolith-rock top', 'rock bottom-regolith']
        ax.axvline(t, linestyle='--', label=label[i], color=colors[i], lw=3)

    plt.xlabel('Time [ns]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=24, loc='lower right')
    plt.tick_params(labelsize=24)
    plt.grid(True)

    if args.closeup:
            ax.set_xlim([closeup_x_start, closeup_x_end])
            ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])


    #* Save the plot
    if args.closeup:
            fig.savefig(output_dir + '/delay_time' + '_closeup_x' + str(closeup_x_start) \
                    + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                    ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(output_dir + '/delay_time' + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)





#* Main
if __name__ == "__main__":
    # for closeup option
    closeup_x_start = 0 #[ns]
    closeup_x_end =100 #[ns]
    closeup_y_start = -60
    closeup_y_end = 60

    #* Load the output data
    for data_path in tqdm(path_group['path']):
        output_dir = os.path.dirname(data_path)

        f = h5py.File(data_path, 'r')
        nrx = f.attrs['nrx']
        for rx in range(nrx):
            data, dt = get_output_data(data_path, (rx+1), 'Ez')
        time = np.arange(len(data)) * dt  / 1e-9
        #print(time.shape, data.shape)

        #* Plot the A-scan
        plot_Ascan(data_path, data, time, rx, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, 'Ez normalized')

        #* Run the pulse analysis and plot the A-scan with peak detection
        pulse_info = analyze_pulses(data, time, rx, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end)

        #* Plot A-scan with estimated two-way travel time
        model_path = os.path.join(output_dir, 'model.json')
        plot_Ascan_estimated_time(data, time, model_path, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, 'Ez normalized', rx=1)


    print('Alls done')

