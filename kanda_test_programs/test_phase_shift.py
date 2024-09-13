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


def shift(original_sig, shit_time):
    shifted_sig = original_sig + np.roll(sig, int(shit_time * 1e-9 / dt))
    return shifted_sig

def env(data):
    env = np.abs(signal.hilbert(data))
    return env


multi_overlap = sig + np.roll(sig, int(-0.2e-9/dt)) + np.roll(sig, int(0.1e-9/dt))


#* 重ね合わせる
fig, ax = plt.subplots(10, 1, figsize=(6, 18), sharex=True, sharey=True, tight_layout=True)
ax[0].plot(t, sig, label='original')
ax[0].plot(t, env(sig), linestyle = '--', color = 'gray')
ax[0].legend()
ax[0].grid()
for i in range (1, 10, 1):
    ax[i].plot(t, shift(sig, i * 0.25), label = f'shit: {i * 0.25} ns')
    ax[i].plot(t, env(shift(sig, i * 0.25)), linestyle = '--', color = 'gray')
    ax[i].legend()
    ax[i].grid()

plt.xlim(0.25, 25) # [ns]

fig.supxlabel('Time [ns]', fontsize=20)
fig.supylabel('Normalized amplitude', fontsize=20)



plt.savefig(os.path.join(output_dir, 'phase.png'), format='png', dpi=120)
plt.savefig(os.path.join(output_dir, 'phase.pdf'), format='pdf', dpi=300)
plt.show()