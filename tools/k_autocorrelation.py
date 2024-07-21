import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy import signal
from matplotlib.gridspec import GridSpec



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_autocorrelation.py',
    description='Calculate the autocorrelation of the data',
    epilog='End of help message',
    usage='python -m tools.k_autocorrelation [json_path]',
)
parser.add_argument('json_path', help='Path to the json file')
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



#* Load output file
data_path = params['out_file']
output_dir = os.path.dirname(data_path)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']

for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')

#* Convert into envelope
data = np.abs(signal.hilbert(data, axis=0))

skip_time = 24e-9
data_skipped = data[int(skip_time/dt):]
print(data_skipped.shape)

peak_time_list = []
for traces in range(data_skipped.shape[1]):
    peak_time_list.append(np.where(np.abs(data_skipped[:, traces]) > 0.1)[0][0] * dt)
fastest_peak_time = np.min(peak_time_list)
vartex = np.argmin(peak_time_list)
reference_wave = data_skipped[:, vartex]
reference_wave_mean = np.mean(reference_wave)
print('Reference wave shape:', reference_wave.shape)
print(fastest_peak_time)

#* Define the function to calculate the autocorrelation
"""
def calc_autocorrelation(Ascan): # data: 1D array
    N = len(Ascan)
    data_ave = np.mean(Ascan)
    sigma = np.var(Ascan)

    # Calculate the autocorrelation using numpy.correlate
    #auto_corr = np.correlate(Ascan_centered, Ascan_centered, mode='full') / (N * sigma)
    for lag in range(N-1):
        auto_corr[k] = np.sum(Ascan_centered[] * Ascan_centered[k:]) / (N * sigma)
    return auto_corr[N-1:]
"""
def calc_acorr(data, k):
    """Returns the autocorrelation of the *k*th lag in a time series data.

    Parameters
    ----------
    data : one dimentional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """

    # yの平均
    y_avg = np.mean(data)

    # 分子の計算
    sum_of_covariance = 0
    for i in range(k+1, len(data)):
        covariance = ( data[i] - y_avg ) * ( data[i-(k+1)] - y_avg )
        sum_of_covariance += covariance

    # 分母の計算
    sum_of_denominator = 0
    for u in range(len(data)):
        denominator = ( data[u] - y_avg )**2
        sum_of_denominator += denominator

    return sum_of_covariance / sum_of_denominator

#acorr_list = []
#for i in tqdm(range(data.shape[0])):
#    acorr_list.append(calc_acorr(data[:, 0], i))



# Define the function to calculate the autocorrelation
def calc_autocorrelation(Ascan):
    N = len(Ascan)
    data_ave = np.mean(Ascan)
    data_var = np.var(Ascan)
    auto_corr = np.correlate(reference_wave -reference_wave_mean, Ascan - data_ave, mode='full')[-N:] / (N * data_var)
    return auto_corr

# Calculate autocorrelation
acorr_list = calc_autocorrelation(data[:, 0])

# Plot the autocorrelation
#plt.plot(acorr_list)
#plt.show()



# Calculate the autocorrelation of the data
auto_corr = np.zeros(data_skipped.shape)
for i in tqdm(range(data_skipped.shape[1]), desc='Calculating autocorrelation'):
    auto_corr[:, i] = calc_autocorrelation(data_skipped[:, i])




#* Plot
font_large = 20
font_medium = 18
font_small = 16

plot_list = [reference_wave, data_skipped, auto_corr]
title_list = ['Reference wave', 'B-scan', 'Autocorrelation']

#fig, ax = plt.subplots(1, 3, figsize=(18, 10), tight_layout=True)
figure = plt.figure(figsize=(15, 10), tight_layout=True)
gs = GridSpec(1, 3, width_ratios=[1, 2, 2])
for i, data in enumerate(plot_list):
    if i == 0:
        ax = plt.subplot(gs[0])
        t = np.arange(skip_time, skip_time + len(data) * dt * 1e9, dt * 1e9)
        print(t)
        ax.plot(data/np.amax(np.abs(data)), t, label='Reference wave', color='black')
        ax.invert_yaxis()
        ax.set_xlim([-1, 1])
        ax.set_ylim([np.max(t),  np.min(t)])
        ax.set_title(title_list[i], fontsize=font_large)
        ax.set_xlabel('Amplitude', fontsize=font_medium)
        ax.tick_params(labelsize=font_small)
    else:
        ax = plt.subplot(gs[i])
        im = ax.imshow(data, cmap='seismic', aspect='auto',
                    extent = [antenna_start, antenna_start + data.shape[1] * antenna_step, data.shape[0] * dt * 1e9, skip_time * 1e9],
                    vmin=-1, vmax=1
                    )

        ax.set_title(title_list[i], fontsize=font_large)
        ax.set_xlabel('x [m]', fontsize=font_medium)
        ax.tick_params(labelsize=font_small)

        delvider = axgrid1.make_axes_locatable(ax)
        cax = delvider.append_axes('right', '5%', pad='3%')
        plt.colorbar(im, cax=cax).set_label('Autocorrelation [%]', fontsize=font_medium)

        if i==1:
            ax.vlines(x=antenna_start + vartex * antenna_step,
                        ymin=skip_time * 1e9, ymax=data.shape[0] * dt * 1e9, color='black', linestyle='--')

figure.supylabel('Time [ns]', fontsize=font_medium)

plt.savefig(output_dir + '/autocorrelation.png')
plt.show()
