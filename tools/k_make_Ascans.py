import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy.signal import hilbert
import json
import shutil
import k_detect_peak # import the function from k_detect_peak.py
import k_plot_TWT_estimation # import the function from k_plot_TWT_estimation.py
import k_subtract # import the function from k_subtract.py



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


#* Main
if __name__ == "__main__":
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

    # for closeup option
    if args.json == '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x6/20241111_polarity_v2/path_under_resolution.json':
        closeup_x_start = 20 #[ns]
        closeup_x_end =40 #[ns]
        closeup_y_start = -25
        closeup_y_end = 25
    else:
        closeup_x_start = 20 #[ns]
        closeup_x_end =80 #[ns]
        closeup_y_start = -60
        closeup_y_end = 60

    #* Load the transmmit signal data for subtraction
    transmmit_signal_path = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x6/20241111_polarity_v2/direct/A-scan/direct.out' # 送信波形データを読み込む

    f = h5py.File(transmmit_signal_path, 'r')
    nrx = f.attrs['nrx']
    for rx in range(nrx):
        transmmit_signal, dt = get_output_data(transmmit_signal_path, (rx+1), 'Ez')

    #* Load the output data
    for data_path in tqdm(path_group['path'], desc='Make A-scans'):
        output_dir = os.path.dirname(data_path)

        f = h5py.File(data_path, 'r')
        nrx = f.attrs['nrx']
        for rx in range(nrx):
            data, dt = get_output_data(data_path, (rx+1), 'Ez')
        time = np.arange(len(data)) * dt  / 1e-9

        #* Plot the A-scan
        plot_Ascan(data_path, data, time, rx, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, 'Ez normalized')

        #* Run the pulse analysis and plot the A-scan with peak detection
        output_dir_peak_detection = os.path.join(os.path.dirname(data_path), 'peak_detection')
        if not os.path.exists(output_dir_peak_detection):
            os.makedirs(output_dir_peak_detection)
        #else:
        #    shutil.rmtree(output_dir_peak_detection)
        #    os.makedirs(output_dir_peak_detection)

        pulse_info = k_detect_peak.detect_plot_peaks(data, dt, args.closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end,
                                                            args.FWHM, output_dir_peak_detection, plt_show=False)

        #* Save the pulse information
        filename = os.path.join(output_dir_peak_detection, 'peak_info.txt')
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


        #* Plot A-scan with estimated two-way travel time
        model_path = os.path.join(output_dir, 'model.json')

        output_dir_TWT_estimation = os.path.join(os.path.dirname(data_path), 'TWT_estimation')
        if not os.path.exists(output_dir_TWT_estimation):
            os.makedirs(output_dir_TWT_estimation)
        #else:
        #    shutil.rmtree(output_dir_TWT_estimation)
        #    os.makedirs(output_dir_TWT_estimation)

        k_plot_TWT_estimation.calc_plot_TWT(data, time, model_path, args.closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end,
                                                output_dir_TWT_estimation, plt_show=False)


        #* Subtract the transmit signal from the A-scan
        #* Zero padding the transmmit signal
        if len(transmmit_signal) < len(data):
            transmmit_signal = np.pad(transmmit_signal, (0, len(data) - len(transmmit_signal)), 'constant')

        #* Define output directory
        output_dir_subtraction = os.path.join(os.path.dirname(data_path), 'subtracted')
        if not os.path.exists(output_dir_subtraction):
            os.makedirs(output_dir_subtraction)
        #else:
        #    shutil.rmtree(output_dir_subtraction)
        #    os.makedirs(output_dir_subtraction)

        time = np.arange(len(data)) * dt  / 1e-9 # [ns]


        #* Load the model json file
        with open(model_path, 'r') as f:
            boundaries = json.load(f)


        #* Detect the first peak in the transmmit signal
        transmit_sig_first_peak_time, transmit_sig_first_peak_amp = k_subtract.detect_first_peak(transmmit_signal, dt)

        #* Calculate the estimated two-way travel time
        TWTs = k_subtract.calc_TWT(boundaries)

        #* Subtract the transmmit signal from the A-scan
        for TWT in TWTs:
            shifted_data, subtracted_data = k_subtract.subtract_signal(data, transmmit_signal, dt, TWT, transmit_sig_first_peak_time, transmit_sig_first_peak_amp)

            #* Plot the subtracted signal
            if TWT > 5:
                closeup_x_start_sub = TWT - 3
            else:
                closeup_x_start_sub = 0
            closeup_x_end_sub = TWT + 7

            k_subtract.plot(data, shifted_data, subtracted_data, time, args.closeup, closeup_x_start_sub, closeup_x_end_sub, closeup_y_start, closeup_y_end,
                                    output_dir_subtraction, TWT, plt_show=False)


    print('Alls done')

