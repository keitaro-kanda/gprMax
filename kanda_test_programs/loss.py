import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0.5, 10.01, 0.01)
z = np.arange(0.5, 10.5, 0.5)
c = 299792458
tan_delta = 0.006
epsilon = 3.4
freq = 500e6
wavelength = c/np.sqrt(epsilon) / freq

power = np.zeros((len(z), len(x)))
for i, z_val in enumerate(z):
    r = np.sqrt(x**2 + z_val**2)
    alpha = np.pi / wavelength * np.sqrt(epsilon) * tan_delta
    power[i, :] = 1 / r**4 * np.exp(-2 * alpha * r)
    power[i, :] = 10 * np.log10(power[i, :])



color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
style_list = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
plt.figure(figsize=(12, 10))
for i in range(len(z)):
    plt.plot(x, power[i, :], label=f'z={z[i]} [m]',
            color=color_list[i], linestyle=style_list[i])

plt.xlabel('x [m]')
plt.ylabel(r'$\frac{1}{år^4}e^{-2 \alpha \tan \delta}$' + ' [dB]')
plt.xlim(0.5, 10)
#plt.ylim(-50, 0)
plt.legend()
plt.grid()

plt.show()
