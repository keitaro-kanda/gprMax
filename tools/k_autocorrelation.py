import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data




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
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']

for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')


#data = data**2

"""
#* Define the function to calculate the autocorrelation
def calc_autocorrelation(Ascan): # data: 1D array
    N = len(Ascan)
    data_ave = np.mean(Ascan)
    sigma = 1 /N * np.sum((Ascan - data_ave)**2)

    #* Calculate the autocorrelation
    auto_corr = np.zeros(N)
    for i in range(1, N+1):
        auto_corr[i-1] = np.sum((Ascan[:N-i] - data_ave) * (Ascan[i:] - data_ave)) / sigma

    return auto_corr
"""

#* Define the function to calculate the autocorrelation
def calc_autocorrelation(Ascan): # data: 1D array
    N = len(Ascan)
    data_ave = np.mean(Ascan)
    Ascan_centered = Ascan - data_ave
    sigma = np.var(Ascan)

    # Calculate the autocorrelation using numpy.correlate
    auto_corr = np.correlate(Ascan_centered, Ascan_centered, mode='full') / (N * sigma)
    return auto_corr[N-1:]

# Calculate the autocorrelation of the data
auto_corr = np.zeros(data.shape)
for i in tqdm(range(data.shape[1]), desc='Calculating autocorrelation'):
    auto_corr[:, i] = calc_autocorrelation(data[:, i])



#* Plot
font_large = 20
font_medium = 18
font_small = 16

fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
im = ax.imshow(auto_corr, cmap='seismic', aspect='auto',
            extent = [antenna_start, antenna_start + auto_corr.shape[1] * antenna_step, auto_corr.shape[0] * dt * 1e9, 0],
            vmin=-np.amax(np.abs(auto_corr))/10, vmax=np.amax(np.abs(auto_corr))/10)

ax.set_xlabel('x [m]', fontsize=font_medium)
ax.set_ylabel('Time [ns]', fontsize=font_medium)
ax.tick_params(labelsize=font_small)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', '5%', pad='3%')
plt.colorbar(im, cax=cax).set_label('Autocorrelation', fontsize=font_medium)


plt.show()
