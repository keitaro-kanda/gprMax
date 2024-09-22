import numpy as np
import matplotlib.pyplot as plt


theta_i = np.linspace(0, np.pi/2, 100)
sin_i = np.sin(theta_i)
cos_i = np.cos(theta_i)
epsilon_1 = 3
n1 = np.sqrt(1 / epsilon_1)
epsilon_2 = 9
n2 = np.sqrt(1 / epsilon_2)


sin_t = np.sqrt(epsilon_1 / epsilon_2) * sin_i
cos_t = np.sqrt(1 - sin_t**2)


R = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
T = 2 *n2 * cos_i / (n2 * cos_i + n1 * cos_t)


plt.plot(theta_i, R, label='R')
plt.plot(theta_i, T, label='T')
plt.legend()

plt.xticks([0, np.pi/6, np.pi/4, np.pi/2], ['0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
plt.grid(True)

plt.title(r'$\varepsilon_1 = $' + str(epsilon_1) + r', $\varepsilon_2 = $' + str(epsilon_2))
plt.xlabel(r'$\theta_i$')
plt.ylabel('Intensity')

plt.savefig('kanda_test_programs/reflectance_coefficient/R_T.png')
plt.show()

