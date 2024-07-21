import json
import numpy as np
import matplotlib.pyplot as plt
from outputfiles_merge import get_output_data
import h5py
import cv2
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from scipy import signal



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_autocorrelation.py',
    description='Calculate the autocorrelation of the data',
    epilog='End of help message',
    usage='python tools/k_gradient.py [json_path] [func_type] [-envelope]',
)
parser.add_argument('json_path', help='Path to the json file')
parser.add_argument('func_type', choices=['sobel', 'gradient'], help='Type of function to be applied')
parser.add_argument('-envelope', action='store_true', help='Apply envelope to the data', default=False)
args = parser.parse_args()


#* Load json file
json_path = args.json_path
with open (json_path) as f:
    params = json.load(f)


#* Define output directory
output_dir = os.path.join(os.path.dirname(json_path), 'gradient')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#* Load antenna settings
src_step = params['antenna_settings']['src_step']
rx_step = params['antenna_settings']['rx_step']
src_start = params['antenna_settings']['src_start']
rx_start = params['antenna_settings']['rx_start']

if src_step == rx_step:
    antenna_step = src_step
    antenna_start = (src_start + rx_start) / 2


#* Load output file
data_path = params['out_file']
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')



#* Define skip time if needed
skip_time = 0
data = data[int(skip_time/dt):]
print('Data shape after skipping time:', data.shape)


#* Calculate envelope
if args.envelope:
    data = np.abs(signal.hilbert(data, axis=0))



#* Define the function to calculate the autocorrelation
def calc_autocorrelation(Ascan): # data: 1D array
    N = len(Ascan)
    data_ave = np.mean(Ascan)
    Ascan_centered = Ascan - data_ave
    sigma = np.var(Ascan)

    # Calculate the autocorrelation using numpy.correlate
    auto_corr = np.correlate(Ascan_centered, Ascan_centered, mode='full') / (N * sigma)
    return auto_corr[N-1:]

# Calculate the autocorrelation of the data
auto_corr = np.zeros(data.shape)
#for i in tqdm(range(data.shape[1]), desc='Calculating autocorrelation'):
#    data[:, i] = calc_autocorrelation(data[:, i])


#* Replace 0 with a small number to avoid division by zero
data[data == 0] = 1e-15


#* Resampling in LPR sample interval
LPR_sample_interval = 0.3125e-9
resample_factor = int(LPR_sample_interval / dt)
#data = data[::resample_factor]
#dt = LPR_sample_interval
print('Data shape after resampling:', data.shape)

#* Apply sobel filter
def sobel():
    sobelx = cv2.Sobel(np.abs(data), cv2.CV_64F, 1, 0, ksize=5)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(np.abs(data), cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.convertScaleAbs(sobely)

    sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    plot_list = [data, sobelx, sobely, sobel_combined]
    return plot_list
#list = sobel()


#* Calculate gradient of the data
def gradient():
    gradx = np.abs(np.gradient(data, axis=1))
    grady = np.abs(np.gradient(data, axis=0))

    gradx[gradx == 0] = 1e-15
    grady[grady == 0] = 1e-15

    grad_combined = np.sqrt(gradx**2 + grady**2)

    plot_list = [data, gradx, grady, grad_combined]
    return plot_list



#* Run the function
if args.func_type == 'sobel':
    list = sobel()
elif args.func_type == 'gradient':
    list = gradient()
else:
    raise ValueError('Invalid function type. Choose from sobel, gradient, division.')

#* Plot the data
font_large = 20
font_medium = 18
font_small = 16
fig, ax = plt.subplots(2, 2, figsize=(18, 18), tight_layout=True)

for i, data in enumerate(list):
    if i == 0:
        if args.envelope:
            color = 'jet'
            vmin = 0
        else:
            color = 'seismic'
            vmin = -1
        im = ax[i//2, i%2].imshow(data, cmap=color, aspect='auto',
                                extent = [antenna_start, antenna_start + data.shape[1] * antenna_step, data.shape[0] * dt * 1e9, skip_time * 1e9],
                                vmin=vmin, vmax=20)
        #ax[i//2, i%2].set_ylim(35, 25)
    else:
        if args.func_type == 'sobel':
            im = ax[i//2, i%2].imshow(data, cmap='jet', aspect='auto',
                                extent = [antenna_start, antenna_start + data.shape[1] * antenna_step,  data.shape[0] * dt * 1e9, skip_time * 1e9],
                                vmin=0, vmax=np.amax(data))
            ax[i//2, i%2].set_title(['Ez', 'Sobel x', 'Sobel y', 'Sobel combined'][i], fontsize=font_large)
        elif args.func_type == 'gradient':
            data = 10 * np.log10(data/np.amax(data))
            im = ax[i//2, i%2].imshow(data, cmap='jet', aspect='auto',
                                    extent = [antenna_start, antenna_start + data.shape[1] * antenna_step,  data.shape[0] * dt * 1e9, skip_time * 1e9],
                                    vmin=-50, vmax=0)
            ax[i//2, i%2].set_title(['Ez', 'Gradient x', 'Gradient y', 'Gradient combined'][i], fontsize=font_large)
    ax[i//2, i%2].set_xlabel('x [m]', fontsize=font_medium)
    ax[i//2, i%2].set_ylabel('Time [ns]', fontsize=font_medium)
    ax[i//2, i%2].set_ylim(150, 0)
    ax[i//2, i%2].tick_params(labelsize=font_small)
    if data.shape[1] > 50:
        ax[i//2, i%2].set_xlim(2, 4)

    delvider = axgrid1.make_axes_locatable(ax[i//2, i%2])
    cax = delvider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical').set_label('Amplitude, [dB]', fontsize=font_medium)
    cax.tick_params(labelsize=font_small)


if args.envelope:
    output_dir = output_dir + '/envelope'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
if args.func_type == 'sobel':
    plt.savefig(output_dir + '/sobel.png', bbox_inches='tight', dpi=300)
elif args.func_type == 'gradient':
    plt.savefig(output_dir + '/gradient.png', bbox_inches='tight', dpi=300)
elif args.func_type == 'division':
    plt.savefig(output_dir + '/division.png', bbox_inches='tight', dpi=300)
plt.show()