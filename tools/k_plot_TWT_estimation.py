import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
import json
from scipy.signal import hilbert



#* Calculate and plot the two-way travel time
def calc_plot_TWT(data, time, model_path, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, output_dir, plt_show):
    #* Physical constants
    c0 = 299792458  # Speed of light in vacuum [m/s]


    #* Load the model json file
    with open(model_path, 'r') as f:
        model = json.load(f)
    optical_path_length = []
    boundary_names = []
    boundaries = model['boundaries']
    for boundary in boundaries:
        boundary_names.append(boundary['name'])
        if optical_path_length == []:
            optical_path_length.append(2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))
        else:
            optical_path_length.append(optical_path_length[-1] +  2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))

    #print(boundary_names)
    #print('Optical path length:')
    #print(optical_path_length)
    #print(' ')


    #* Calculate the envelope
    envelope = np.abs(hilbert(data))


    #* Calculate the two-way travel time
    two_way_travel_time = [length / c0 /1e-9 for length in optical_path_length] # [ns]
    #* Add the initial pulse delay
    delay = model['initial_pulse_delay']# [ns]
    two_way_travel_time = [t + delay for t in two_way_travel_time]

    #print('Two-way travel time [ns]:')
    #print(two_way_travel_time)
    #print(' ')

    #* Save the two-way travel time as txt
    np.savetxt(output_dir + '/delay_time.txt', two_way_travel_time, fmt='%.6f', delimiter=' ', header='Two-way travel time [ns]')




    #* Plot
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'),
                                figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

    #* Plot A-scan
    ax.plot(time, data, label='A-scan', color='black', linewidth=2)
    ax.plot(time, envelope, label='Envelope', color='gray', linestyle='-.', linewidth=2)
    #* Plot the estimated two-way travel time
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, t in enumerate(two_way_travel_time):
        if i == 0:
            ax.axvline(t, linestyle='--', color=colors[i], linewidth=3)
        else:
            ax.axvline(t, linestyle='--', color=colors[i], linewidth=3, label=boundary_names[i])


    plt.xlabel('Time [ns]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    #plt.title('Pulse Analysis')
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=24)
    plt.grid(True)


    if closeup:
            ax.set_xlim([closeup_x_start, closeup_x_end])
            ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])


    #* Save the plot
    if closeup:
            fig.savefig(output_dir + '/TWT_estimation' + '_closeup_x' + str(closeup_x_start) \
                    + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                    ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(output_dir + '/TWT_estimation' + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    if plt_show:
        plt.show()
    else:
        plt.close()



#* Main
if __name__ == "__main__":
    #* Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='k_plot_time_estimation.py',
        description='Plot the estimated two-way travel time on the A-scan',
        epilog='End of help message',
        usage='python -m tools.k_plot_time_estimation [out_file] [model_json] [-closeup]',
    )
    parser.add_argument('out_file', help='Path to the .out file')
    #parser.add_argument('model_json', help='Path to the model json file')
    parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
    args = parser.parse_args()


    #* Define path
    data_path = args.out_file
    output_dir = os.path.join(os.path.dirname(data_path), 'TWT_estimation')
    os.makedirs(output_dir, exist_ok=True)

    #* Load the A-scan data
    f = h5py.File(data_path, 'r')
    nrx = f.attrs['nrx']
    for rx in range(nrx):
        data, dt = get_output_data(data_path, (rx+1), 'Ez')

    time = np.arange(len(data)) * dt  / 1e-9 # [ns]

    #* Model json path
    model_path = os.path.join(os.path.dirname(data_path), 'model.json')


    #* Load the model json file
    #with open(args.model_json, 'r') as f:
    #    model = json.load(f)


    #* for closeup option
    closeup_x_start = 25 #[ns]
    closeup_x_end =50 #[ns]
    closeup_y_start = -60
    closeup_y_end = 60

    calc_plot_TWT(data, time, model_path, args.closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, output_dir, plt_show=True)
