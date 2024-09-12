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



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_detect_peak.py',
    description='Detect the peak from the B-scan data',
    epilog='End of help message',
    usage='python -m tools.k_detect_peak [json_path]',
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

#* Define output directory
output_dir = os.path.join(os.path.dirname(data_path), 'detect_peak')
os.makedirs(output_dir, exist_ok=True)


#* Load the data
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('data shape: ', data.shape)


#* Calculate the envelope
envelope = np.abs(signal.hilbert(data, axis=0))



#* Detect peak from the B-scan data
scatter_x_idx = []
scatter_time_idx = []
scatter_value = []

data_trim = data[int(15e-9/dt):, :]
envelope_trim = envelope[int(15e-9/dt):, :]
backgrounds = np.mean(np.abs(data_trim), axis=0)
thresholds = backgrounds * 3


#* Detect the peak
for i in tqdm(range(data.shape[1]), desc='Detecting peaks'):
    above_threshold_indices = np.where(envelope_trim[:, i] > thresholds[i])[0]

    if len(above_threshold_indices) > 0:
        # Find the start and end of each group of indices above the threshold
        split_points = np.split(above_threshold_indices, np.where(np.diff(above_threshold_indices) != 1)[0] + 1)

        for group in split_points:
            start, end = group[0], group[-1] + 1
            peak_idx_in_group = np.argmax(np.abs(envelope_trim[start:end, i])) + start

            scatter_x_idx.append(i)
            scatter_time_idx.append(peak_idx_in_group)
            scatter_value.append(data_trim[peak_idx_in_group, i])
            #if data[peak_idx_in_group, i] > 0:
            #    scatter_value.append(1)
            #else:
            #    scatter_value.append(-1)

print('Length of values array: ', len(scatter_value))
print('Length of x index array: ', len(scatter_x_idx))
print('Length of time index array: ', len(scatter_time_idx))

scatter_x_idx = np.array(scatter_x_idx)
scatter_time_idx = np.array(scatter_time_idx)
scatter_value = np.array(scatter_value)

scatter_max = np.amax(np.abs(scatter_value))


#* Calculate dB
envelope[envelope == 0] = 1e-10
envelope_db = 10 * np.log10(envelope/np.amax(np.abs(envelope)))



#* Plot
print(' ')
print('Plotting...')



fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111)

im = ax.imshow(envelope_db, aspect='auto', cmap='viridis',
                extent=[antenna_start, antenna_start + data.shape[1]*antenna_step,
                15 + data.shape[0]*dt*1e9, 0],
                vmin=-35, vmax=0
                )
scatter = ax.scatter(antenna_start + scatter_x_idx*antenna_step, scatter_time_idx*dt*1e9 + 15, # +50 to compensate the trim
                    c=scatter_value, cmap='bwr', s=5,
                    vmin = -30, vmax = 30)


#* Set labels
ax.set_xlabel('Distance [m]', fontsize=20)
ax.set_ylabel('Time [ns]', fontsize=20)
ax.tick_params(labelsize=16)


delvider = axgrid1.make_axes_locatable(ax)
cax_im = delvider.append_axes('right', size='3%', pad=0.1)
cbar_im = plt.colorbar(im, cax=cax_im, orientation = 'vertical')
cbar_im.set_label('Envelope', fontsize=18)
cbar_im.ax.tick_params(labelsize=16)

cax_scatter = delvider.append_axes('right', size='3%', pad=1)
cbar_scatter = plt.colorbar(scatter, cax=cax_scatter, orientation = 'vertical')
cbar_scatter.set_label('Peak amplitude', fontsize=18)
cbar_scatter.ax.tick_params(labelsize=16)


fig.savefig(output_dir + '/detect_peak.png', format='png', dpi=120)
fig.savefig(output_dir + '/detect_peak.pdf', format='pdf', dpi=300)

plt.show()
