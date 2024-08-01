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
from matplotlib.gridspec import GridSpec
import scipy.signal as signal
from scipy.interpolate import interp1d, RectBivariateSpline



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_fk_migration.py',
    description='Process f-k migration',
    epilog='End of help message',
    usage='python -m tools.k_fk_migration [json_path]',
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



#* Define the function to calculate the f-k migration
def fk_migration(data, epsilon_r):

    #* Calculate the temporal frequency
    ws = 2 * np.pi * np.fft.fftfreq(data.shape[0], dt)
    ws = ws[ws >= 0]

    #* 2D Fourier transform
    FK = np.fft.fft2(data)
    FK = FK[:len(ws), :]

    #* Calculate the wavenumber in x direction
    kx = 2 * np.pi * np.fft.fftfreq(data.shape[1], antenna_step)

    #* Interpolate from frequency (ws) into wavenumber (kz)
    v = 3e8 / np.sqrt(epsilon_r)
    interp_real = RectBivariateSpline(np.fft.fftshift(kx), ws, np.fft.fftshift(FK.real).T, kx=1, ky=1)
    interp_imag = RectBivariateSpline(np.fft.fftshift(kx), ws, np.fft.fftshift(FK.imag).T, kx=1, ky=1)

    #* interpolation will move from frequency-wavenumber to wavenumber-wavenumber, KK = D(kx,kz,t=0)
    KK = np.zeros_like(FK)

    #* Calculate the wavenumber in z direction
    for zj in tqdm(range(data.shape[0]//2), desc='Calculating wavenumber in z direction'):
        kzj = ws[zj] * 2 / v

        for xi in range(len(kx)):
            kxi = kx[xi]
            wsj = v / 2 * np.sqrt(kzj**2 + kxi**2)

            #* Get the interpolated FFT values, real and imaginary, S(kx, kz, t=0)
            KK[zj, xi] = interp_real(kxi, wsj)[0, 0] + 1j * interp_imag(kxi, wsj)[0, 0]

    #* All vertical waevnumbers
    kz = ws * 2 / v

    #* Calculate the scaling factor
    kX, kZ = np.meshgrid(kx, kz)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = kZ / np.sqrt(kX**2 + kZ**2)
    KK *= scaling
    #* The DC current should be zero
    KK[0, 0] = 0 + 0j



    #* Inverse 2D Fourier transform to get time domain data
    fk_data = np.fft.ifft2(KK)
    print('fk_data shape: ', fk_data.shape)

    return fk_data, v



#* Run the f-k migration function
fk_data, v = fk_migration(data, 3)



#* Plot
plt.figure(figsize=(12, 12), facecolor='w', edgecolor='w')
im = plt.imshow(np.abs(fk_data), cmap='jet', aspect='auto',
                extent=[antenna_start,  antenna_start + fk_data.shape[1] * antenna_step,
                fk_data.shape[0] * dt * v / 2, 0],
                vmin=0, vmax=30
                )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('z [m]', fontsize=20)
plt.tick_params(labelsize=18)

delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Amplitude', fontsize=20)
cbar.ax.tick_params(labelsize=18)

plt.savefig(os.path.join(output_dir, 'fk_migration.png'))
plt.show()