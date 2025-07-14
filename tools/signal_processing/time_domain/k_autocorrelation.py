import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from tools.core.outputfiles_merge import get_output_data
from scipy import signal
from matplotlib.gridspec import GridSpec
import scipy.signal as signal



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
data_path = params['data']
output_dir = os.path.join(os.path.dirname(data_path), 'autocorrelation')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')
print('data shape: ', data.shape)



#* Convert into envelope
data_env = np.abs(signal.hilbert(data, axis=0))



#* Trim the data
#* x start and end is in [m], y start and end is in [s]
x_start = 0
x_end = 2
y_start = 20e-9
y_end = 70e-9

data_trim = data_env[int(y_start/dt):int(y_end/dt), int(x_start/antenna_step):int(x_end/antenna_step)]
print('data shape after trim: ', data_trim.shape)



#* Define the autocorrelation function
def calc_acorr_column(Ascan):
    peak_start = np.where(Ascan > 0.1)[0][0]
    Ascan_trim = Ascan[peak_start:]

    N = len(Ascan_trim)
    data_mean = np.mean(Ascan_trim)
    data_var = np.var(Ascan_trim)
    acorr_trim = np.correlate(Ascan_trim - data_mean, Ascan_trim - data_mean, mode='full')[-N:] / (N * data_var)

    acorr = np.zeros(Ascan.shape)
    acorr[peak_start:] = acorr_trim

    return acorr


#* Run the autocorrelation function
auto_corr = np.zeros(data.shape)
for i in tqdm(range(data.shape[1]), desc='Calculating autocorrelation'):
    auto_corr[:, i] = calc_acorr_column(data[:, i])



#* Plot
font_large = 20
font_medium = 18
font_small = 16

figure = plt.figure(figsize=(12, 10), tight_layout=True)
ax = plt.subplot(111)

im = ax.imshow(auto_corr, cmap='seismic', aspect='auto',
                extent = [antenna_start + x_start, antenna_start + x_start + auto_corr.shape[1]*antenna_step,
                          y_end * 1e9, y_start * 1e9],
                vmin=-1, vmax=1
                )
ax.set_xlabel('x [m]', fontsize=font_medium)
ax.set_ylabel('Time [ns]', fontsize=font_medium)
ax.tick_params(labelsize=font_small)
ax.grid(which='both', axis='both', linestyle='-.')

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', '5%', pad='3%')
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Autocorrelation [%]', fontsize=font_medium)
cbar.ax.tick_params(labelsize=font_small)

name_area = str(x_start) + '_' + str(x_end) + '_' + str(int(y_start*1e9)) + '_' + str(int(y_end*1e9))
plt.savefig(output_dir + '/acorr_' + name_area + '.png')
plt.savefig(output_dir + '/acorr_' + name_area + '.pdf', format='pdf', dpi=300)
plt.show()