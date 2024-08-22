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
    prog='k_gain.py',
    description='Process gain function',
    epilog='End of help message',
    usage='python -m tools.k_gain.py [json_path]',
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
output_dir = os.path.join(os.path.dirname(data_path), 'gain')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('original data shape: ', data.shape)
print('original dt: ', dt)



#* Prameters
c = 299792458 # [m/s]
epsilon_0 = 8.854187817e-12 # [F/m]



#* Gain function
#* Reference: Feng et al. (2023), p.3
def gain(data, er, tan_delta, freq, pulse_delay):
    t_2D = np.expand_dims(np.arange(0, data.shape[0]*dt, dt), axis=1)

    gain_func = t_2D**2 * c**2 / (4 * er) * np.exp(np.pi * t_2D * freq * np.sqrt(er * epsilon_0)* tan_delta)

    #* Plot gain function
    fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
    ax.plot(gain_func, t_2D/1e-9)

    ax.set_xlabel('Gain function', fontsize=20)
    ax.set_xscale('log')
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylabel('2-way travel time [ns]', fontsize=20)
    ax.invert_yaxis()
    ax.tick_params(labelsize=18)
    ax.grid(which='major', axis='both', linestyle='-.')
    ax.text(0.1, 0.1, r'$\varepsilon_r = $' + str(er) + ', tan$\delta = $' + str(round(tan_delta, 3)),
            fontsize=18, transform=ax.transAxes)

    plt.savefig(os.path.join(output_dir, 'gain_function.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, 'gain_function.pdf'), format='pdf', dpi=300)
    plt.close()
    print('Plot of gain function is successfully saved.')
    print(' ')

    delay_idx = int(pulse_delay / dt)
    gained_data = data
    gained_data[delay_idx:] = data[delay_idx:] * gain_func[delay_idx:]
    return gained_data


epsilon_r = 3
sigma = 0.001
freq = params['pulse_info']['center_frequency']
v = c / np.sqrt(epsilon_r)
tan_delta = sigma / (2 * np.pi * freq * epsilon_r * epsilon_0)
pulse_start_time = params['pulse_info']['transmitting_delay']

data_gain = gain(data, epsilon_r, tan_delta, freq, pulse_start_time)
print('data_gain shape: ', data_gain.shape)

#* Save the data as .out file
shutil.copy(data_path, os.path.join(output_dir, 'gain.out'))
copy_hdf5 = h5py.File(os.path.join(output_dir, 'gain.out'), 'a')
rx_group = copy_hdf5['rxs/rx1']
if 'gain' in rx_group:
    del rx_group['gain']
    rx_group.create_dataset('gain', data=data_gain)
else:
    rx_group.create_dataset('gain', data=data_gain)
print('gain data is successfully saved as the output file')
print(' ')


#* Copy json file and edit its 'data' key
json_copy_name = os.path.join(output_dir, 'gain.json')
shutil.copy(args.json_path, json_copy_name)
with open(json_copy_name, 'r') as f:
    json_data = json.load(f)
json_data['data'] = os.path.join(output_dir, 'gain.out')
json_data['data_name'] = 'gain'
with open(json_copy_name, 'w') as f:
    json.dump(json_data, f, indent=4)
print('json file is copied and edited')





#* Plot
plt.figure(figsize=(20, 15))
im = plt.imshow(data_gain, cmap='seismic', aspect='auto',
                extent=[antenna_start,  antenna_start + data_gain.shape[1] * antenna_step,
                data_gain.shape[0] * dt / 1e-9, 0],
                vmin=-np.amax(np.abs(data_gain)), vmax=np.amax(np.abs(data_gain))
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [ns] (assume ' r'$\varepsilon_r = $'+ str(epsilon_r) + ')', fontsize=20)
plt.tick_params(labelsize=18)
plt.grid(which='both', axis='both', linestyle='-.')

delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.5)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.savefig(os.path.join(output_dir, 'Bscan_gain.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'Bscan_gain.pdf'), format='pdf', dpi=600)
plt.show()