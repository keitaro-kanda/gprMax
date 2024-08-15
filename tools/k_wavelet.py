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
from matplotlib.colors import LogNorm



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_wavelet.py',
    description='Calculate the wavelet transform of the data',
    epilog='End of help message',
    usage='python -m tools.k_wavelet [json_path]',
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
print('data shape: ', data.shape)
print('dt: ', dt)



#* Define function to calculate the wavelet transform
#* sig: A-scan data
#* tm: time array
def calc_wavelet(sig, tm):
    # 連続ウェーブレット変換
    # --- n_cwt をlen(sig)よりも大きな2のべき乗の数になるように設定
    n_cwt = int(2**(np.ceil(np.log2(len(sig)))))
    print('n_cwt: ', n_cwt)

    # --- 後で使うパラメータを定義
    dj = 0.125 # parameter to determine the resolution of frequency
    omega0 = 6.0
    s0 = 2.0*dt # The smallest scale
    J = int(np.log2(n_cwt*dt/s0)/dj) # The largest scale

    #* Scale s
    s = s0*2.0**(dj*np.arange(0, J+1, 1))

    # --- n_cwt個のデータになるようにゼロパディングをして，DC成分を除く
    x = np.zeros(n_cwt)
    x[0:len(sig)] = sig[0:len(sig)] - np.mean(sig)

    # --- omega array
    omega = 2.0*np.pi*np.fft.fftfreq(n_cwt, dt)

    #* Continuous wavelet transform (CWT)
    X = np.fft.fft(x)
    cwt = np.zeros((J+1, n_cwt), dtype=complex) # CWT array

    Hev = np.array(omega > 0.0)
    for j in tqdm(range(J+1)):
        Psi = np.sqrt(2.0*np.pi*s[j]/dt)*np.pi**(-0.25)*np.exp(-(s[j]*omega-omega0)**2/2.0)*Hev # Morlet wavelet
        cwt[j, :] = np.fft.ifft(X*np.conjugate(Psi))

    #* Convert scale s into frequency
    s_to_f = (omega0 + np.sqrt(2 + omega0**2)) / (4.0*np.pi)
    freq_cwt = s_to_f / s
    cwt = cwt[:, 0:len(sig)]

    # --- cone of interference
    COI = np.zeros_like(tm)
    COI[0] = 0.5/dt
    COI[1:len(tm)//2] = np.sqrt(2)*s_to_f/tm[1:len(tm)//2]
    COI[len(tm)//2:-1] = np.sqrt(2)*s_to_f/(tm[-1]-tm[len(tm)//2:-1])
    COI[-1] = 0.5/dt


    return freq_cwt, cwt, COI


time_array = np.arange(0, data.shape[0]*dt, dt) # [s]
freq_cwt, cwt, COI = calc_wavelet(data[:, 400], time_array)
cwt = 10 * np.log10(np.abs(cwt) / np.amax(np.abs(cwt)))
print('shape of freq_cwt: ', cwt.shape)
print(freq_cwt)




#* Plot the wavelet transform
fig =  plt.figure(figsize=(20, 15))
ax1 = fig.add_axes([0.1, 0.55, 0.78, 0.35])
ax_sc = fig.add_axes([0.1, 0.1, 0.78, 0.35])
ax_cbar = fig.add_axes([0.9, 0.1, 0.02, 0.35])

ax1.plot(time_array/1e-9, data[:, 400])
ax1.set_xlabel('Time [ns]', fontsize=18)
ax1.set_ylabel('Amplitude', fontsize=18)
ax1.set_xlim(0, data.shape[0]*dt/1e-9)
ax1.set_title('Original data', fontsize=20)
ax1.grid(True, linestyle='-.', linewidth=0.5)
ax1.tick_params(labelsize=16)


im = ax_sc.imshow(cwt, aspect='auto',
            extent=[0, data.shape[0]*dt/1e-9, np.amin(freq_cwt)/1e6, np.amax(freq_cwt)/1e6],
            cmap='jet',
)
ax_sc.fill_between(time_array/1e-9, 0, COI/1e6, color='k', alpha=0.5)
ax_sc.plot(time_array/1e-9, COI, 'k--')
ax_sc.set_ylim(np.amin(freq_cwt)/1e6, np.amax(freq_cwt)/1e6)
ax_sc.set_xlabel('Time [ns]', fontsize=18)
ax_sc.set_ylabel('Frequency [MHz]', fontsize=18)
ax_sc.set_title('Wavelet transform', fontsize=20)
ax_sc.grid(True, linestyle='-.', linewidth=0.5)
ax_sc.tick_params(labelsize=16)
ax_sc.set_yscale('log')

delvider = axgrid1.make_axes_locatable(plt.gca())
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.ax.tick_params(labelsize=18)

plt.show()
