import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
from tools.core.outputfiles_merge import get_output_data
import os
import argparse
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_matched_filter.py',
    description='Calculate the matched filter',
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
output_dir = os.path.join(os.path.dirname(data_path), 'Matched_filter')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('original data shape: ', data.shape)
print('original dt: ', dt)



#* Add noise
#data = data + np.random.normal(0, np.amax(data)/100, data.shape)


#* Define the function to calculate the gaussian wave
def gaussian(t_array, mu, sigma):
        a = 1 / (sigma * np.sqrt(2 * np.pi))
        b = np.exp(-0.5 * (((t_array - mu) / sigma) ** 2))
        return a * b


#* Define the function to calculate the matched filter
def maeched_filter(Ascan):
    t = np.arange(0, 10e-9, dt)
    reference_sig = gaussian(t, 2.8e-9, 0.3e-9)
    reference_sig = np.concatenate([np.flip(reference_sig), np.zeros(len(Ascan) - len(reference_sig))])
    fft_refer = np.fft.fft(np.conj(reference_sig))
    fft_data = np.fft.fft(Ascan)
    conv = np.fft.ifft(fft_refer * fft_data)
    return np.real(conv)



#* Apply the matched filter to the data
data_matched = np.zeros(data.shape)
for i in tqdm(range(data.shape[1])):
    data_matched[:, i] = maeched_filter(data[:, i])

data_matched = np.abs(data_matched)
data_matched = 10 * np.log10(data_matched / np.max(data_matched))


#* Plot
plt.figure(figsize=(20, 15))
im = plt.imshow(data_matched, aspect='auto', cmap='jet',
                extent=[antenna_start, antenna_start + antenna_step * data.shape[1], dt * data.shape[0] / 1e-9, 0],
                vmin=-50, vmax=0)
plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [ns]', fontsize=20)
plt.tick_params(labelsize=18)


delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.5)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.savefig(os.path.join(output_dir, 'matched_filter.png'), dpi=120)
plt.savefig(os.path.join(output_dir, 'matched_filter.pdf'), format='pdf', dpi=600)
plt.show()