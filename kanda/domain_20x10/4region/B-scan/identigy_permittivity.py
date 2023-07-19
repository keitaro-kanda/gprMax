import time

import matplotlib.pyplot as plt
import numpy as np

# 真空を伝播する時間
c = 299792458 # [m/s]
t_vacuum0 = 2/c # [s]

# rx=40を基準にする
rx40_time = 0.5300e-7

# rx=40以降のtime
time_array = np.array([0.5307e-7, 0.5326e-7, 0.5354e-7, 0.5397e-7, 0.5446e-7, # 45まで
                0.5507e-7, 0.5576e-7, 0.5656e-7, 0.5741e-7, 0.5833e-7, # 50まで
                0.5930e-7, 0.6033e-7, 0.6142e-7, 0.6253e-7, 0.6366e-7, # 55まで
                0.6484e-7, 0.6602e-7, 0.6720e-7, 0.6842e-7, 0.6963e-7, # 60まで
                0.7088e-7, 0.7210e-7, 0.7338e-7, 0.7463e-7, 0.7590e-7, # 65まで
])

epsilon_r1 = np.zeros(len(time_array))
epsilon_r2 = np.zeros(len(time_array))
index = np.zeros(len(time_array))

for i in range(len(time_array)):

    L = (i + 1) * 0.2 # [m]
    index[i] = L

    # 
    epsilon_r1[i] = ((time_array[i]) **2 - (rx40_time)**2) * (c /2 /L)**2 

    #tau1 = time_array[i]*(1 - 2/c/rx40_time) # 
    #tau0 = rx40_time - 2/c
    #epsilon_r2[i] = (c**4 * rx40_time**2)*(tau1**2 - tau0**2) / ((2 * l * (c * rx40_time - 2))**2)

    #
    time_1m = 2/c
    time_perp = rx40_time - time_1m # A
    time_oblique_vacuum = time_1m * time_array[i] / rx40_time # tau1'
    time_oblique_ground = time_array[i] - time_oblique_vacuum # B

    v_ground = 2 * L /np.sqrt(time_oblique_ground**2 - time_perp**2) - c * time_array[i] / time_oblique_ground
    epsilon_r2[i] = (c / v_ground)**2

mean = np.mean(epsilon_r2)
print(mean)

plt.plot(index, epsilon_r1, label='no vacuum rivised')
plt.plot(index, epsilon_r2, label='vacuum rivised')
plt.xlabel('distance from echo peak [m]', size=18)
plt.ylabel('relative permittivity', size=18)
#plt.yscale('log')
plt.legend(fontsize = 18)


plt.grid()
plt.savefig('kanda/domain_20x10/4region/B-scan/identify_permittivity.png')

plt.show()
