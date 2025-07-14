import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
from outputfiles_merge import get_output_data
import os
import argparse
import mpl_toolkits.axes_grid1 as axgrid1
import shutil


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
output_dir = os.path.join(os.path.dirname(data_path), 'Noise_added')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('original data shape: ', data.shape)
print('original dt: ', dt)



#* Add noise
data = data + np.random.normal(0, np.amax(data)/1000, data.shape)


#* Save the data as .out file
shutil.copy(data_path, os.path.join(output_dir, 'noise.out'))
copy_hdf5 = h5py.File(os.path.join(output_dir, 'noise.out'), 'a')
rx_group = copy_hdf5['rxs/rx1']
if 'gain' in rx_group:
    del rx_group['gain']
    rx_group.create_dataset('noise', data=data)
else:
    rx_group.create_dataset('noise', data=data)
print('gain data is successfully saved as the output file')
print(' ')


#* Copy json file and edit its 'data' key
json_copy_name = os.path.join(output_dir, 'noise.json')
shutil.copy(args.json_path, json_copy_name)
with open(json_copy_name, 'r') as f:
    json_data = json.load(f)
json_data['data'] = os.path.join(output_dir, 'noise.out')
json_data['data_name'] = 'noise'
with open(json_copy_name, 'w') as f:
    json.dump(json_data, f, indent=4)
print('json file is copied and edited')



#* Plot
plt.figure(figsize=(20, 15))
im = plt.imshow(data, cmap='seismic', aspect='auto',
                extent=[antenna_start,  antenna_start + data.shape[1] * antenna_step,
                data.shape[0] * dt / 1e-9, 0],
                vmin=-1, vmax=1
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [ns]', fontsize=20)
plt.tick_params(labelsize=18)
plt.grid(which='both', axis='both', linestyle='-.')

delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.5)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.savefig(os.path.join(output_dir, 'Bscan_gain.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'Bscan_gain.pdf'), format='pdf', dpi=600)
plt.show()