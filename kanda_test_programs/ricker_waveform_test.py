import numpy as np
import matplotlib.pyplot as plt


f0 = 500e6 # [Hz]
dt = 1e-11 # [s]
t = np.arange(-10e-9, 10e-9, dt) # [s]
tau = 0 # [s]
epsilon_0 = 1
sigma = 1 / (np.sqrt(2) * np.pi * f0)


ricker = (1 - 2 * np.pi**2 * f0**2 * (t - tau)**2) * np.exp(-np.pi**2 * f0**2 * (t - tau)**2)
E_field = - 1 / epsilon_0 * (t - tau) / (np.sqrt(np.pi) * sigma) * np.exp(- np.pi**2 * f0**2 * (t - tau)**2)

#* Plot
plt.plot(t, ricker, label='Ricker Waveform')
plt.plot(t, E_field, label='E-field')
plt.legend()
plt.grid(True)
plt.show()