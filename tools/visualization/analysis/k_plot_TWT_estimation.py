import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
import json
from scipy.signal import hilbert
from tools.core.outputfiles_merge import get_output_data

#* Physical constants
c0 = 299792458  # Speed of light in vacuum [m/s]

def calculate_TWT(model_path):
    """
    Calculates the theoretical two-way travel time based on a model JSON file.

    Args:
        model_path (str): Path to the model.json file.

    Returns:
        tuple: A tuple containing (list_of_TWTs, list_of_boundary_names).
               Returns (None, None) if an error occurs.
    """
    try:
        with open(model_path, 'r') as f:
            model = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading model file {model_path}: {e}")
        return None, None

    optical_path_length = []
    boundary_names = []
    boundaries = model.get('boundaries', [])
    for boundary in boundaries:
        boundary_names.append(boundary.get('name', 'Unknown'))
        length = boundary.get('length', 0)
        epsilon_r = boundary.get('epsilon_r', 1)
        current_opl = 2 * length * np.sqrt(epsilon_r)
        
        if not optical_path_length:
            optical_path_length.append(current_opl)
        else:
            optical_path_length.append(optical_path_length[-1] + current_opl)

    two_way_travel_time = [length / c0 / 1e-9 for length in optical_path_length]  # [ns]
    delay = model.get('initial_pulse_delay', 0)  # [ns]
    two_way_travel_time = [t + delay for t in two_way_travel_time]

    return two_way_travel_time, boundary_names


#* Calculate and plot the two-way travel time
def calc_plot_TWT(data, time, model_path, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, output_dir, plt_show):
    two_way_travel_time, boundary_names = calculate_TWT(model_path)
    if two_way_travel_time is None:
        return

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, 'delay_time.txt'), two_way_travel_time, fmt='%.6f', delimiter=' ', header='Two-way travel time [ns]')

    envelope = np.abs(hilbert(data))

    #* Plot
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'),
                                figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

    ax.plot(time, data, label='A-scan', color='black', linewidth=2)
    ax.plot(time, envelope, label='Envelope', color='gray', linestyle='-.', linewidth=2)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, t in enumerate(two_way_travel_time):
        ax.axvline(t, linestyle='--', color=colors[i % len(colors)], linewidth=3, label=boundary_names[i])

    plt.xlabel('Time [ns]', fontsize=28)
    plt.ylabel('Amplitude', fontsize=28)
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=24)
    plt.grid(True)

    if closeup:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])

    if output_dir:
        save_path = os.path.join(output_dir, 'TWT_estimation.png')
        if closeup:
            save_path = os.path.join(output_dir, f'TWT_estimation_closeup_x{closeup_x_start}_{closeup_x_end}_y{closeup_y_start}_{closeup_y_end}.png')
        fig.savefig(save_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    if plt_show:
        plt.show()
    else:
        plt.close(fig)


#* Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='k_plot_time_estimation.py',
        description='Plot the estimated two-way travel time on the A-scan',
        epilog='End of help message',
        usage='python -m tools.visualization.analysis.k_plot_TWT_estimation [out_file] [-closeup]',
    )
    parser.add_argument('out_file', help='Path to the .out file')
    parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
    args = parser.parse_args()

    data_path = args.out_file
    output_dir = os.path.join(os.path.dirname(data_path), 'TWT_estimation')
    
    data, dt = get_output_data(data_path, 1, 'Ez')
    time = np.arange(len(data)) * dt / 1e-9  # [ns]

    model_path = os.path.join(os.path.dirname(data_path), 'model.json')

    closeup_x_start, closeup_x_end = 45, 60
    closeup_y_start, closeup_y_end = -60, 60

    calc_plot_TWT(data, time, model_path, args.closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, output_dir, plt_show=True)
