import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
from outputfiles_merge import get_output_data
import os
from scipy import signal
import argparse
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_spectrogram.py',
    description='Plot the spectrogram of the data',
    epilog='End of help message',
    usage='python -m tools.k_spectrogram [json_path]',
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
output_dir = os.path.join(os.path.dirname(data_path), 'migration')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')
print('original data shape: ', data.shape)
print('original dt: ', dt)



#* Gain function
t_2D = np.expand_dims(np.linspace(0, data.shape[0]*dt, data.shape[0]), axis=1)
plt.plot(t_2D[:, 0], t_2D[:, 0]**1.7)
gained_data = data * t_2D ** 1.7
gained_data = gained_data / np.amax(np.abs(gained_data))




x = 12
id = int((x - rx_start) / rx_step)
sig = gained_data[:, id]
tm = np.arange(0, len(sig)*dt, dt)

f, t, Sxx = signal.spectrogram(sig, 1/dt, nperseg=256, noverlap=128, nfft=2048)
t = t / 1e-9 # [ns]
f = f / 1e6 # [MHz]
Sxx = 10 * np.log10(Sxx/np.amax(Sxx))




#* Plot the wavelet transform
fig =  plt.figure(figsize=(12, 12))
ax1 = fig.add_axes([0.1, 0.55, 0.78, 0.35])
ax_sc = fig.add_axes([0.1, 0.1, 0.78, 0.35])
ax_cbar = fig.add_axes([0.9, 0.1, 0.02, 0.35])

ax1.plot(tm/1e-9, gained_data[:, id])
ax1.set_xlabel('Time [ns]', fontsize=18)
ax1.set_ylabel('Amplitude', fontsize=18)
ax1.set_xlim(0, data.shape[0]*dt/1e-9)
ax1.set_title('Original data', fontsize=20)
ax1.grid(True, linestyle='-.', linewidth=0.5)
ax1.tick_params(labelsize=16)


im = ax_sc.imshow(Sxx, aspect='auto',
            extent=[t[0], t[-1], f[-1], f[0]],
            cmap='jet',
            vmin=-100, vmax=0
)
#ax_sc.set_ylim(0, 2000)
ax_sc.set_xlabel('Time [ns]', fontsize=18)
ax_sc.set_ylabel('Frequency [MHz]', fontsize=18)
ax_sc.set_title('Wavelet transform', fontsize=20)
ax_sc.grid(True, linestyle='-.', linewidth=0.5)
ax_sc.tick_params(labelsize=16)
ax_sc.invert_yaxis()
ax_sc.set_ylim(0, 2000)


delvider = axgrid1.make_axes_locatable(plt.gca())
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.ax.tick_params(labelsize=18)

plt.show()
