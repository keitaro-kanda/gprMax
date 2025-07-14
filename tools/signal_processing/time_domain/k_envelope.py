import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
from tools.core.outputfiles_merge import get_output_data
import os
from scipy import signal
import argparse
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1
from matplotlib.colors import LogNorm


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_envelope.py',
    description='Plot the spectrogram of the data',
    epilog='End of help message',
    usage='python -m tools.k_envelope.py [json_path]',
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
output_dir = os.path.join(os.path.dirname(data_path), 'Hilbert')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('original data shape: ', data.shape)
print('original dt: ', dt)


#* Gain function
t_2D = np.expand_dims(np.linspace(0, data.shape[0]*dt, data.shape[0]), axis=1)
gained_data = data * t_2D ** 1.7
gained_data = gained_data / np.amax(np.abs(gained_data))


#* Calculate envelope
envelope = np.abs(signal.hilbert(gained_data, axis=0))
print('envelope shape: ', envelope.shape)


#* Calculate the instantaneous frequency
inst_freq = np.zeros((data.shape[0], data.shape[1]))  # Adjust the shape to match the output of np.diff
for i in tqdm(range(data.shape[1])):
    #inst_freq[:, i] = np.diff(np.unwrap(np.angle(signal.hilbert(data[:, i])))) / (2 * np.pi * dt)
    inst_freq[:, i] = np.angle(signal.hilbert(gained_data[:, i]))



#* Mask the inst_freq by the envelope
inst_freq_mask = np.zeros(inst_freq.shape)
for i in range(inst_freq.shape[1]):
    inst_freq_mask[:, i] = inst_freq[:, i] * envelope[:, i]



#* Plot
envelope = 10 * np.log10(envelope/np.amax(envelope))

plot_list = [envelope, inst_freq, inst_freq_mask]
plot_name = ['envelope', 'inst_freq', 'inst_freq_mask']
cmaps = ['jet', 'seismic', 'seismic']
vmin = [np.amin(envelope)/2, -np.pi, -1]
vmax = [0, np.pi, 1]
cbar_label = ['Envelope', 'Instantaneous phase', 'Instantaneous phase']
for i, plot in enumerate(plot_list):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(plot,
                extent=[antenna_start, antenna_start + plot.shape[1] * antenna_step, plot.shape[0] * dt / 1e-9, 0],
                interpolation='nearest', aspect='auto', cmap=cmaps[i],
                vmin=vmin[i], vmax=vmax[i])
    ax.set_xlabel('x [m]', fontsize=20)
    ax.set_ylabel('Time [ns]', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, linestyle='-.', linewidth=0.5)

    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label(cbar_label[i], size=20)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(os.path.join(output_dir, plot_name[i] + '.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, plot_name[i] + '.pdf'), format='pdf', dpi=600)
    plt.show()