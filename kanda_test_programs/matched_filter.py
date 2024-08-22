import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


dt = 0.01e-9

def ricker_gprMax(t_array, f0):
    zeta = np.pi **2 * f0**2
    xai = np.sqrt(2) / f0
    sig = - (2 * zeta * (t_array - xai)**2 - 1) * np.exp(-zeta * (t_array - xai)**2)
    return sig


def gaussian(t_array, mu, sigma):
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    b = np.exp(-0.5 * (((t_array - mu) / sigma) ** 2))
    return a * b

def fft(data):
    data_fft = np.fft.fft(data)
    data_fft = data_fft[1:len(data_fft)//2]
    data_fft = 10 * np.log10(np.abs(data_fft) / np.max(np.abs(data_fft)))
    freq = np.fft.fftfreq(len(data), dt)
    freq = freq[1:len(freq)//2]
    return data_fft, freq

#* Matched filter
def matched_filter(refer, data):
    reference_sig = np.concatenate([np.flip(refer), np.zeros(len(data) - len(refer))])
    #reference_sig = reference_sig * signal.windows.hamming(len(reference_sig))
    fft_refer = np.fft.fft(np.conj(reference_sig))
    fft_data = np.fft.fft(data)
    conv = np.fft.ifft(fft_refer * fft_data)
    return np.abs(conv)



t = np.arange(0, 40e-9, dt)

y1 = ricker_gprMax(t, 500e6)
print('y1 shape: ', y1.shape)
y1 = y1 / np.max(y1)
fft1, freq1 = fft(y1)

gauss_pulse = gaussian(t, 2.8e-9, 0.3e-9)
gauss_pulse = gauss_pulse / np.max(gauss_pulse)
fft_gauss, freq_gauss = fft(gauss_pulse)

y1_shift = np.roll(ricker_gprMax(t, 500e6), int(10e-9/dt)) + np.random.normal(0, 0.05, len(t))
print('y1_shift shape: ', y1_shift.shape)
y1_shift = y1_shift / np.max(y1_shift)
fft1_shift, freq1_shift = fft(y1_shift)


corr_filter = matched_filter(y1, y1_shift)
corr_filter = corr_filter / np.max(corr_filter)


corr_gauss = matched_filter(gauss_pulse, y1_shift)
corr_gauss = corr_gauss / np.max(corr_gauss)


y2 = ricker_gprMax(t, 250e6)
y2 = y2 / np.max(y2)
fft2, freq2 = fft(y2)

y2_corr = signal.correlate(y1_shift, y2, mode='full')
y2_corr = y2_corr / np.max(y2_corr)
t_corr = np.arange(-len(y2_corr)//2, len(y2_corr)//2) * dt
print('y2_corr shape: ', y2_corr.shape)



fig, ax = plt.subplots(2, 2, figsize=(18, 10))

ax[0, 0].plot(t, y1, label='ricker')
ax[0, 0].plot(t, y1_shift, label='shifted')
ax[0, 0].plot(t, corr_filter, label='Matched filter')
ax[0, 0].set_xlabel('Time [ns]')
ax[0, 0].set_ylabel('Amplitude')
ax[0, 0].grid()
ax[0, 0].legend()

ax[0, 1].plot(freq1, fft1, label='rigker')
ax[0, 1].plot(freq1_shift, fft1_shift, label='shifted')
ax[0, 1].vlines(500e6, -50, 0, 'r', '--')
ax[0, 1].set_xlabel('Frequency [Hz]')
ax[0, 1].set_ylabel('Amplitude [dB]')
ax[0, 1].set_xscale('log')
ax[0, 1].set_ylim(-40, 0)
ax[0, 1].grid()
ax[0, 1].legend()

ax[1, 0].plot(t, gauss_pulse, label='gaussian')
ax[1, 0].plot(t, y1_shift, label='shifted')
ax[1, 0].plot(t, corr_gauss, label='Matched filter')
ax[1, 0].set_xlabel('Time [ns]')
ax[1, 0].set_ylabel('Amplitude')
ax[1, 0].grid()
ax[1, 0].legend()

ax[1, 1].plot(freq_gauss, fft_gauss, label='gaussian')
ax[1, 1].plot(freq1_shift, fft1_shift, label='shifted')
ax[1, 1].set_xlabel('Frequency [Hz]')
ax[1, 1].set_ylabel('Amplitude [dB]')
ax[1, 1].set_xscale('log')
ax[1, 1].set_ylim(-40, 0)
ax[1, 1].grid()
ax[1, 1].legend()

plt.show()