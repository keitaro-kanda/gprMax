import numpy as np
import matplotlib.pyplot as plt



def ricker(t, sigma):
    a = 2 * np.sqrt(3 * sigma) * np.pi**0.25
    b = 1 - (t / sigma)**2
    c = np.e**(-1 * t**2 / (2 * sigma**2))
    return a * b * c

dt = 0.01e-9
t_length = 20e-9
N = int(t_length / dt)
t = np.arange(0, t_length, dt)

y1 = ricker(t, 2e-9)
y1 = y1 / np.max(y1)

#* fft
y1_fft = np.fft.fft(y1)
y1_fft = y1_fft[1:len(y1_fft)//2]
y1_fft = 10 * np.log10(np.abs(y1_fft) / np.max(np.abs(y1_fft)))
freq1 = np.fft.fftfreq(len(y1), dt)
freq1 = freq1[1:len(freq1)//2]


def ricker_gprMax(t, f0):
    zeta = np.pi **2 * f0**2 * t**2
    xai = np.sqrt(2) / f0
    sig = - (2 * zeta * (t - xai)**2 - 1) * np.exp(-zeta * (t - xai)**2)
    return sig




fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(t, y1)
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Amplitude')
ax[0].grid()

ax[1].plot(freq1, y1_fft)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Amplitude [dB]')
ax[1].set_xscale('log')
ax[1].grid()

plt.show()