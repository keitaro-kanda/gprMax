import numpy as np
import matplotlib.pyplot as plt



def ricker_gprMax(t_array, f0):
    zeta = np.pi **2 * f0**2
    xai = np.sqrt(2) / f0
    sig = - (2 * zeta * (t_array - xai)**2 - 1) * np.exp(-zeta * (t_array - xai)**2)
    return sig


t = np.arange(0, 15e-9, 1e-14)
f0 = 500e6

input = ricker_gprMax(t, f0)
waveform = np.sin(2 * np.pi * f0 * t) * input
waveform = waveform / np.max(np.abs(waveform))

input = input / np.max(np.abs(input))


#* FFT
def fft(data):
    data_fft = np.fft.fft(data)
    data_fft = np.abs(data_fft) / np.max(np.abs(data_fft))
    data_fft = data_fft[1:len(data_fft)//2]
    freq = np.fft.fftfreq(len(data), 1e-14)
    freq = freq[1:len(freq)//2]
    return data_fft, freq

fft_waveform, freq = fft(waveform)
fft_input, freq = fft(input)




#* Plot
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(t, waveform)
ax[0, 0].set_title('Waveform')
ax[0, 0].grid()

ax[0, 1].plot(t, input)
ax[0, 1].set_title('Input')
ax[0, 1].grid()

ax[1, 0].plot(freq, fft_waveform)
ax[1, 0].set_xscale('log')
ax[1, 0].set_xlim(1e8, 1e10)
ax[1, 0].grid()

ax[1, 1].plot(freq, fft_input)
ax[1, 1].set_xscale('log')
ax[1, 1].set_xlim(1e8, 1e10)
ax[1, 1].grid()

plt.show()

