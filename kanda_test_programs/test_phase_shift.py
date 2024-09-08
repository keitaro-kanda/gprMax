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

sig = - sig / np.max(np.abs(sig))


def shift(original_sig, shit_time):
    shifted_sig = original_sig + np.roll(sig, int(shit_time * 1e-9 / dt))
    return shifted_sig

def env(data):
    env = np.abs(signal.hilbert(data))
    return env


#* 重ね合わせる
plt.figure(figsize=(12, 8))
plt.plot(t, sig, label='original')
plt.plot(t, env(sig), linestyle = '--', color = 'gray')
for i in range (1, 11, 1):
    plt.plot(t, shift(sig, i * 0.25) - i * 2.5, label = f'shit: {i * 0.25} ns')
    plt.plot(t, env(shift(sig, i * 0.25))- i * 2.5, linestyle = '--', color = 'gray')

plt.xlim(0.25, 25) # [ns]

plt.xlabel('Time [ns]', fontsize=20)
plt.ylabel('Normalized amplitude', fontsize=20)
plt.legend(fontsize=16, loc='upper right')
plt.tick_params(size=16)
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'phase.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'phase.pdf'), format='pdf', dpi=300)
plt.show()