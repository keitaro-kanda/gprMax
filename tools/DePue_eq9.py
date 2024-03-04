import numpy as np
import matplotlib.pyplot as plt

"""
This code is checking the DePue et al. method for estimating Vrms from CMP data.
"""


c = 299792458 # [m/s], speed of light in vacuum

#* De Pue eq(9)
def DePue_eq9(y1, y0, z0, z1, v1):
    # De Pue eq(9b)
    lateral_air = (y0 - y1) / 2
    sin0 = lateral_air / np.sqrt(lateral_air**2 + z0**2)
    # De Pue eq(9c)
    lateral_ground = (y1 / 2)
    sin1 = lateral_ground / np.sqrt(lateral_ground**2 + z1**2)

    return sin0 / sin1 - c / v1


y0 = [1, 5, 10, 15, 20] # [m], antenna offset
z0 = 1 # [m], antenna height
z1 = [5, 10, 20, 40] # [m], depth


fig, ax = plt.subplots(1,len(z1), figsize=(30, 10), tight_layout=True)
for depth in range(len(z1)):
    for i in range(len(y0)):
        y1 = np.linspace(y0[i]/100, y0[i], 100)
        v1_ind = 0.4
        v1 = v1_ind * c # [m/s], RMS velocity
        result = DePue_eq9(y1, y0[i], z0, z1[depth], v1)
        offset_ground = np.argmin(np.abs(result))

        ax[depth].plot(y1, result, label=f'y0 = {y0[i]}')
        fig.supxlabel('y1 [m]', fontsize=18)
        fig.supylabel(r'$\frac{sin(\theta_0)}{sin(\theta_1)} - \frac{c}{v_1}$', fontsize=18)
        ax[depth].set_title('z1: {} [m], v1: {} [/c]'.format(z1[depth], v1_ind), fontsize=20)
        ax[depth].legend()
        ax[depth].grid()
        ax[depth].set_yscale('symlog')

        # show DePue_eq9 get 0
        text_y = [-0.8, -0.4, 0.2, 0.4, 0.8]
        ax[depth].scatter(y1[offset_ground], result[np.argmin(np.abs(result))], c='k', s=30, marker='x')
        ax[depth].text(y1[offset_ground], text_y[i],
                    f'({y1[offset_ground]:.2f}, {result[np.argmin(np.abs(result))]:.4e})', fontsize=14)
plt.show()