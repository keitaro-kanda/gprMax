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
    usage='python -m tools.k_fk_migration [json_path] [-mask]',
)
parser.add_argument('json_path', help='Path to the json file')
parser.add_argument('-mask', action='store_true', help='Mask the direct wave area', default=False)
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
output_dir = os.path.join(os.path.dirname(data_path), 'migration')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('data shape: ', data.shape)



#* Define the function to calculate the f-k migration
def fk_migration(data, epsilon_r):

    #* Calculate the temporal angular frequency
    omega = 2 * np.pi * np.fft.fftfreq(data.shape[0], dt)
    omega = omega[1: len(omega)//2]
    print('shape of omega: ', omega.shape)

    #* 2D Fourier transform, frequency-wavenaumber domain
    FK = np.fft.fft2(data)
    FK = FK[:len(omega), :]

    #* Calculate the wavenumber in x direction
    kx = 2 * np.pi * np.fft.fftfreq(data.shape[1], antenna_step)
    #kx = kx[1: len(kx)//2]
    #print(kx)
    print('shape of kx: ', kx.shape)

    #* Interpolate from frequency (ws) into wavenumber (kz)
    v = 299792458 / np.sqrt(epsilon_r)
    interp_real = RectBivariateSpline(np.fft.fftshift(kx), omega, np.fft.fftshift(FK.real, axes=1).T, kx=1, ky=1)
    interp_imag = RectBivariateSpline(np.fft.fftshift(kx), omega, np.fft.fftshift(FK.imag, axes=1).T, kx=1, ky=1)

    #* interpolation will move from frequency-wavenumber to wavenumber-wavenumber, KK = D(kx,kz,t=0)
    KK = np.zeros_like(FK)

    #* Calculate the wavenumber in z direction
    for zj in tqdm(range(len(omega)), desc='Calculating wavenumber in z direction'):
        kz_j = omega[zj] * 2 / v

        for xi in range(len(kx)):
            kx_i = kx[xi]
            omega_j = v / 2 * np.sqrt(kz_j**2 + kx_i**2)
            #kz_j = np.sqrt(omega[zj]**2 / v**2 - kx_i**2)
            #omega_j = np.sqrt(kx_i**2 + kz_j**2) * v

            #* Get the interpolated FFT values, real and imaginary, S(kx, kz, t=0)
            KK[zj, xi] = interp_real(kx_i, omega_j)[0, 0] + 1j * interp_imag(kx_i, omega_j)[0, 0]

    #* All vertical waevnumbers
    kz = omega * 2 / v

    #* Calculate the scaling factor
    """
    『地中レーダ』 p. 151
    omega^2 = (kx^2 + kz^2) * v^2
    dw = kz / sqrt(kx^2 + kz^2) * dky
    """
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
er = 3
fk_data, v = fk_migration(data, er)
np.savetxt(os.path.join(output_dir, 'fk_migration.txt'), np.abs(fk_data), delimiter=',')



#* Plot
plt.figure(figsize=(20, 15), facecolor='w', edgecolor='w')
if args.mask:
    mask_first_ns = 5 # [ns]
    mask_last_ns = 200 # [ns]
    mask_data = fk_data[int(mask_first_ns*1e-9/dt):int(mask_last_ns*1e-9/dt), :]

    mask_first_m = mask_first_ns * 1e-9 * v / 2
    mask_last_m = mask_last_ns * 1e-9 * v / 2
    im = plt.imshow(np.abs(mask_data), cmap='jet', aspect='auto',
                extent=[antenna_start,  antenna_start + fk_data.shape[1] * antenna_step,
                mask_first_m + mask_data.shape[0] * dt * v / 2, mask_first_m],
                vmin=0, vmax=np.max(np.abs(mask_data))
                )
else:
    im = plt.imshow(np.abs(fk_data), cmap='jet', aspect='auto',
                    extent=[antenna_start,  antenna_start + fk_data.shape[1] * antenna_step,
                    fk_data.shape[0] * dt * v / 2, 0],
                    vmin=0, vmax=np.max(np.abs(fk_data))
                    )

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('z [m] (assume ' r'$\varepsilon_r = $'+ str(er) + ')', fontsize=20)
plt.tick_params(labelsize=18)


delvider = axgrid1.make_axes_locatable(plt.gca())
cax = delvider.append_axes('right', size='5%', pad=0.5)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

if args.mask:
    plt.savefig(os.path.join(output_dir, 'fk_migration_mask.png'))
    plt.savefig(os.path.join(output_dir, 'fk_migration_mask.pdf'), format='pdf', dpi=300)
else:
    plt.savefig(os.path.join(output_dir, 'fk_migration.png'))
    plt.savefig(os.path.join(output_dir, 'fk_migration.pdf'), format='pdf', dpi=300)
plt.show()