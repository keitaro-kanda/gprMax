import json
from tools.core.outputfiles_merge import get_output_data
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import os
from tqdm import tqdm
import argparse
import scipy.signal as signal
import h5py



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_acorr.py',
    description='Calculate the autocorrelation',
    epilog='End of help message',
    usage='python tools/calc_acorr.py [json_path] [input_type] [x_first] [x_last] [y_first] [y_last]',
)
parser.add_argument('json_path', help='Path to the json file')
parser.add_argument('input_type', choices = ['raw', 'migration'], help='Type of input data')
parser.add_argument('x_first', type=int, help='Start position of x-axis [m]')
parser.add_argument('x_last', type=int, help='End position of x-axis [m]')
parser.add_argument('y_first', type=int, help='Start time of y-axis [ns]')
parser.add_argument('y_last', type=int, help='End time of y-axis [ns]')
args = parser.parse_args()



#* Load json file
with open(args.json_path) as f:
    params = json.load(f)
#* Load antenna settings
src_step = params['antenna_settings']['src_step']
rx_step = params['antenna_settings']['rx_step']
src_start = params['antenna_settings']['src_start']
rx_start = params['antenna_settings']['rx_start']
#* Check antenna step
if src_step == rx_step:
    antenna_step = src_step
    antenna_start = (src_start + rx_start) / 2



#* Load input data
data_path = params['data']
output_dir = os.path.join(os.path.dirname(data_path), 'migration')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')
if args.input_type == 'migration':
    data = np.loadtxt(params['processed_data']['migration'], delimiter=',')
    dt = 2 * dt
print('data shape: ', data.shape)
normalized_data = data / np.amax(data)



#* Trim the data
x_first = args.x_first # [m]
x_last = args.x_last # [m]
y_first = args.y_first # [ns]
y_last = args.y_last  # [ns]

x_first_idx = int(x_first / antenna_step)
x_last_idx = int(x_last / antenna_step)
y_first_idx = int(y_first  * 1e-9 / dt)
y_last_idx = int(y_last * 1e-9 / dt)

trimmed_data = normalized_data[y_first_idx:y_last_idx, x_first_idx:x_last_idx]
print('Trimmed data shape:', trimmed_data.shape)
trimmed_data = trimmed_data / np.amax(trimmed_data)  # Normalize the data



#* Calculate the envelope
env_data = np.abs(signal.hilbert(trimmed_data, axis=0))
print('Envelope data shape:', env_data.shape)



"""
#* Define the function to calculate the autocorrelation
N = trimmed_data.shape[0]
def calc_autocorrelation(Ascan): # data: 1D array
    #* Calculate the autocorrelation
    auto_corr = np.zeros(N)
    data_ave = np.mean(Ascan)
    sigma = 1 / N * np.sum((Ascan - data_ave)**2)
    for i in range(1, N+1):
        auto_corr[i-1] = 1 / N *np.sum((Ascan[i:] - data_ave) * (Ascan[:-i] - data_ave)) / sigma

    return auto_corr
"""


#* Define the autocorrelation function
def calc_acorr_column(Ascan):
    # Check if there are any values greater than 0.1
    peaks = np.where(np.abs(Ascan) > 0.1)[0]
    if peaks.size == 0:
        # If no peak found, return an array of zeros with the same length as Ascan
        return np.zeros(len(Ascan))

    else:
        peak_start = peaks[0]
        Ascan_trim = Ascan[peak_start:]

        N = len(Ascan_trim)
        data_mean = np.mean(Ascan_trim)
        data_var = np.var(Ascan_trim)
        acorr_column_trim = np.correlate(Ascan_trim - data_mean, Ascan_trim - data_mean, mode='full')[-N:] / (N * data_var)

        acorr_column = np.zeros(len(Ascan))
        acorr_column[peak_start:] = acorr_column_trim

        return acorr_column



#* Define the function to calculate the autocorrelation
def run_acorr_func(data):
    auto_corr = np.zeros(data.shape)
    for i in tqdm(range(data.shape[1]), desc='Calculating autocorrelation'):
        auto_corr[:, i] = calc_acorr_column(data[:, i])

    return auto_corr



#* Plot
def plot(plot_list):
    font_large = 20
    font_medium = 18
    font_small = 16

    fig, ax = plt.subplots(1, len(plot_list), figsize=(26, 8), tight_layout=True, sharex=True, sharey=True)

    cmap_list = ['viridis', 'jet', 'seismic']
    title_list = ['Normalized B-scan', 'Envelope', 'Autocorrelation']
    vmin_list = [-1, 0, -np.amax(plot_list[2])]
    vmax_list = [1, np.amax(plot_list[1]), np.amax(plot_list[2])]

    for i, data in enumerate(plot_list):
        im = ax[i].imshow(data, cmap=cmap_list[i], aspect='auto',
                        extent = [x_first, x_first + data.shape[1]*antenna_step,
                                    y_last, y_first],
                        vmin = vmin_list[i], vmax = vmax_list[i]
                        )
        ax[i].set_title(title_list[i], fontsize=font_large)
        ax[i].tick_params(labelsize=font_small)
        ax[i].grid(which='both', axis='both', linestyle='-.')

        delvider = axgrid1.make_axes_locatable(ax[i])
        cax = delvider.append_axes('right', '5%', pad='3%')
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(title_list[i], fontsize=font_medium)

    fig.supxlabel('x [m]', fontsize=font_medium)
    fig.supylabel('Time [ns]', fontsize=font_medium)



    #name_area = str(x_first) + '_' + str(x_last) + '_' + str(int(y_first)) + '_' + str(int(y_last))
    #output_dir = os.path.join('/Volumes/SSD_kanda/LPR/LPR_2B/test/autocorrelation', name_area)
    #os.makedirs(output_dir, exist_ok=True)
    #plt.savefig(output_dir + '/acorr.png')
    #plt.savefig(output_dir + '/acorr.pdf')
    plt.show()



auto_corr = run_acorr_func(env_data)
plot([trimmed_data, env_data, auto_corr])