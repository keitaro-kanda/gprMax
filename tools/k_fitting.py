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
from numpy.linalg import svd, eig, inv



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_fitting.py',
    description='Process hyperbola fitting',
    epilog='End of help message',
    usage='python -m tools.k_fk_migration [json_path]',
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
output_dir = os.path.join(os.path.dirname(data_path), 'fitting')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('data shape: ', data.shape)




#* Define function to extract byperbola by detecting the peak
def extract_peak(data, trac_num):
    """
    data: Ascan data
    i: trace number
    """

    skip_time = 20 # [ns]
    data = data[int(skip_time*1e-9/dt):]

    #* Detect the peak in the envelope
    threshold = np.max(np.abs(data)) * 0.1

    envelope = np.abs(signal.hilbert(data))

    i = 0
    while i < len(envelope):
        if envelope[i] > threshold:
            start = i
            while i < len(envelope) and envelope[i] > threshold:
                i += 1
            end = i
            idx_time.append(np.argmax(np.abs(data[start:end])) + start + int(skip_time*1e-9/dt)) # index, not time
            idx_trace.append(trac_num)
            peak_value.append(data[idx_trace[-1]])
        i += 1



#* Extract peak
idx_time = []
idx_trace = []
peak_value = []

for i in range(data.shape[1]):
    extract_peak(data[:, i], i)



#* Plot the peak on the B-scan
plt.figure(figsize=(20, 15))
im = plt.imshow(data, cmap='gray', aspect='auto',
                extent=[antenna_start,  antenna_start + data.shape[1] * antenna_step,
                data.shape[0] * dt / 1e-9, 0],
                vmin=-np.amax(np.abs(data)/100), vmax=np.amax(np.abs(data)/100)
                )
plt.scatter(np.array(idx_trace) * antenna_step + antenna_start, np.array(idx_time) * dt / 1e-9, c='r', s=20, marker='x')

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [ns])', fontsize=20)
plt.tick_params(labelsize=18)
plt.grid(which='both', axis='both', linestyle='-.')

delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.5)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Amplitude', fontsize=20)

plt.savefig(os.path.join(output_dir, 'fitting.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'fitting.pdf'), format='pdf', dpi=600)
plt.show()