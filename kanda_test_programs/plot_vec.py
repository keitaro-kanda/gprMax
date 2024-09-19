import numpy as np
import matplotlib.pyplot as plt

n = np.array([0, 1])
n = n / np.linalg.norm(n)

I = np.array([2, -3])
I = I / np.linalg.norm(I)

n1 = 1
n2 = 3
eta = n2 / n1



R = I - 2 * np.dot(I, n) * n
R = R / np.linalg.norm(R)
#T = 1/eta * (I - np.dot(I, n) * n) - n * np.sqrt(1 - 1/eta**2 * (1 - np.dot(I, n)**2))
T = 1/eta * I - 1/eta * (np.dot(I, n) + np.sqrt(eta**2 - 1 + np.dot(I, n)**2)) * n

cos_I = np.dot(I, n)
sin_I = np.sqrt(1 - cos_I**2)
sin_T = eta * sin_I
cos_T = np.sqrt(1 - sin_T**2)
#* nをarccos(cos_T)だけ回転したベクトルがT
#T = cos_T * n + sin_T * np.array([n[1], -n[0]])


plt.quiver(0, 0, n[0], n[1], angles='xy', scale_units='xy', scale=1, color='r', label='n')
plt.quiver(-I[0], -I[1], I[0], I[1], angles='xy', scale_units='xy', scale=1, color='b', label='I')
plt.quiver(0, 0, R[0], R[1], angles='xy', scale_units='xy', scale=1, color='g', label='R')
plt.quiver(0, 0, T[0], T[1], angles='xy', scale_units='xy', scale=1, color='y', label='T')

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.grid()
plt.legend()
plt.show()