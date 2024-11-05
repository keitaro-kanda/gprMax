import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import signal

import sys
sys.path.append('tools')
from outputfiles_merge import get_output_data


data_path = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/20240824_echo_shape_test/direct_wave/direct.out'
output_dir = os.path.join(os.path.dirname(data_path), 'phase_shift')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    sig, dt = get_output_data(data_path, (rx+1), 'Ez')
print('data shape: ', sig.shape)


t = np.arange(len(sig)) * dt / 1e-9 # Time in ns

sig = sig / np.max(np.abs(sig))


def shift(shit_time):
    shifted_sig = - np.roll(sig, int(shit_time * 1e-9 / dt)) * 0.7
    return shifted_sig

def env(data):
    env = np.abs(signal.hilbert(data))
    return env


time_lag = 3.2 # [ns]
shifted = shift(time_lag)
overlaped = sig + shifted

plot_list = [sig, shifted, overlaped]
label_list = ['original', f'{time_lag} ns shifted', 'overlaped']

#* 重ね合わせる
fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True, tight_layout=True)
for i in range(len(plot_list)):
    ax.plot(t, plot_list[i] - 2.5 * i, label=label_list[i])


plt.xlim(0, 25) # [ns]
plt.grid()
plt.legend(fontsize=16, loc='upper right')
ax.tick_params(labelsize=16)
fig.supxlabel('Time [ns]', fontsize=20)
fig.supylabel('Normalized amplitude', fontsize=20)



plt.savefig(os.path.join(output_dir, str(time_lag) + 'ns_shift.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, str(time_lag) + 'ns_shift.pdf'), format='pdf', dpi=300)
plt.show()