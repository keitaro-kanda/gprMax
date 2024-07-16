import numpy as np
import argparse
import json
import os
import scipy.fftpack as fft
from tqdm import tqdm
from tools.outputfiles_merge import get_output_data
import h5py
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import matplotlib.colors as colors

#* Parse command line arguments
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.fourier jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* Open output file and read number of outputs (receivers)
data_path = params['out_file']
data = h5py.File(data_path, 'r')
nrx = data.attrs['nrx']
data.close()
data_dir_path = os.path.dirname(data_path)

"""
#* load data
data_list = []
for i in range(1, nrx+1):
    data, dt = get_output_data(data_path, i, 'Ez')
    data_list.append(data)
# data_list is 3D array, (trace number, time, rx)
"""

data, dt = get_output_data(data_path, 1, 'Ez')

fs = 1 / dt # sampling frequency
N = data.shape[0] # number of samples

#* Fourier analysis
IFFT_data = np.zeros_like(data)
for trace_num in tqdm(range(nrx), desc='fourier analysis'):
    FFT_data = fft.fft(data[:, trace_num]) # 2D array, (frequency, rx)

    #* fourier space analysis
    Amp = np.abs(FFT_data) # Amplitude Spectrum, [Intensity]
    ASD = np.sqrt(Amp **2 / (fs / N)) # Amplitude Spectrum Density, [Intensity/√Hz]
    ASD[ASD == 0] = 1e-12 # avoid log(0)
    ASD_norm = ASD / np.amax(ASD) # normalize
    PSD_norm = 10 * np.log10(ASD_norm) # normalized Power Spectrum Density, [dB/Hz]

    #* for inverse Fourier transform
    # cutoff PSD < -40dB
    #PSD_norm[PSD_norm < -50] = -120

    ASD_norm = 10 ** (PSD_norm / 10)
    #ASD = ASD_norm * ASD_max
    Amp = ASD_norm * np.sqrt(fs / N)

    IFFT_data[:, trace_num] = fft.ifft(Amp)

print('finish fourier analysis')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

plt.imshow(np.abs(IFFT_data), aspect='auto',
        extent=[0, IFFT_data.shape[1], IFFT_data.shape[0]*dt, 0],
        cmap='jet',
        norm=colors.LogNorm(vmin=1e-3, vmax=1))

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax, label='Intensity')

plt.show()


"""
#* Fourier transform
def fourier_transform():
    fourier_data_list = []
    for i in range(nrx):
        fourier_data = fft.fft(data, axis=0)
        fourier_data_list.append(fourier_data)
    return fourier_data_list

fourier_data_list = fourier_transform() # fourier_data_list is 3D array, (trace number, frequency, rx)
fourier_data_list = np.array(fourier_data_list)
#print('shape: ', fourier_data_list.shape)
"""

Amp = np.abs(FFT_data) # Amplitude Spectrum
ASD = np.sqrt(Amp **2 / (fs / N)) # Amplitude Spectrum Density
print('ASD shape: ', ASD.shape)
ASD[ASD == 0] = 1e-12 # avoid log(0)
ASD_norm = ASD / np.amax(ASD) # normalize
ASD_max = np.amax(ASD)
"""
ASD_norm = np.zeros_like(ASD)
for i in range(nrx):
    ASD_norm[:, :, i] = ASD[:, :, i] / np.amax(ASD[:, :, i]) # normalize
    amax_list = np.amax(ASD[:, :, i], axis=0)
"""
PSD_norm = 10 * np.log10(ASD_norm) # normalized Power Spectrum Density


#* make frequency axis
frequency_axis = fft.fftfreq(data.shape[0], d=dt)
# frequency_axis is 1D array, (frequency)


#* make output directory
output_dir = os.path.join(data_dir_path, 'fourier')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


#* plot
def plot():
    for rx in range(nrx):
        plt.figure(figsize=(10, 10))
        for trace_num in tqdm(range(0, nrx), desc='plotting, rx' + str(rx+1)):
            plt.plot(frequency_axis[1: int(N/2)], PSD_norm[trace_num, :, rx], label='trace No.' + str(trace_num+1))

        #plt.yscale('log')
        plt.title('Power Spectrum Density, rx' + str(rx+1), size=18)
        plt.xlabel('frequency [Hz]', size=16)
        plt.ylabel('Power Spectrum Density [dB]', size=16)
        plt.xlim(0, 1e9)
        plt.ylim(-70, 0)
        plt.grid()
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=12)

        plt.savefig(output_dir + '/fourier_rx' + str(rx+1) + '.png')
        #plt.show()
#plot()


#* cutoff PSD < -40dB
PSD_norm[PSD_norm < -40] = -120
print('PSD_norm shape: ', PSD_norm.shape)

ASD_norm = 10 ** (PSD_norm / 10)
ASD = ASD_norm * ASD_max
"""
for rx in range(nrx):
    ASD = ASD_norm[:, :, rx] * amax_list[rx]
"""
print('ASD shape: ', ASD.shape)
Amp = ASD * np.sqrt(fs / N)
print('Amp shape: ', Amp.shape)


#* inverse Fourier transform
IFFT_data = np.zeros_like(Amp)
for rx in range(nrx):
    IFFT_data[:, rx] = fft.ifft(Amp[:, rx])
print('inverse_fourier_data shape: ', IFFT_data.shape)
"""
#* inverse Fourier transform
def inverse_fourier_transform():
    inverse_fourier_data_list = []
    for i in range(nrx):
        inverse_fourier_data = fft.ifft(Amp[i], axis=0)
        inverse_fourier_data_list.append(inverse_fourier_data)
    return inverse_fourier_data_list
IFFT_data = inverse_fourier_transform() # inverse_fourier_data_list is 3D array, (trace number, time, rx)
IFFT_data = np.array(IFFT_data)
print('inverse_fourier_data_list shape: ', IFFT_data.shape)
"""

plt.figure(figsize=(10, 10))
plt.imshow(np.abs(IFFT_data), aspect='auto')
plt.colorbar()
plt.show()