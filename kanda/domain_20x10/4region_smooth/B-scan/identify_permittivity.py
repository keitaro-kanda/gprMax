import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy import size

from tools.outputfiles_merge import get_output_data

# 真空を伝播する時間
c = 299792458 # [m/s]
t_vacuum0 = 2/c # [s]


# rx=11~
time_array = np.array([
                0.5456e-7, 0.5479e-7, 0.5512e-7, 0.5555e-7, 0.5609e-7, # 15まで
                0.5670e-7, 0.5743e-7, 0.5824e-7, 0.5911e-7, 0.6005e-7, # 20まで
                0.6104e-7, 0.6208e-7, 0.6316e-7, 0.6430e-7, 0.6545e-7, # 25まで
                0.6666e-7, 0.6786e-7, 0.6908e-7, 0.7031e-7, 0.7156e-7, # 30まで
                0.7281e-7, 0.7409e-7, 0.7536e-7, 0.7666e-7, 0.7793e-7, # 35まで
])
time_array = time_array - 0.0434e-7 

# rx= 10を基準にする
tau0 = 0.5444e-7 - 0.0434e-7 # [s]


epsilon_r1 = np.zeros(len(time_array))
epsilon_r2 = np.zeros(len(time_array))
index = np.zeros(len(time_array))

for i in range(len(time_array)):

    L = (i + 1) * 0.2 # [m]
    index[i] = L

    # 
    epsilon_r1[i] = ((time_array[i]) **2 - (tau0)**2) * (c /2 /L)**2 

    #tau1 = time_array[i]*(1 - 2/c/rx40_time) # 
    #tau0 = rx40_time - 2/c
    #epsilon_r2[i] = (c**4 * rx40_time**2)*(tau1**2 - tau0**2) / ((2 * l * (c * rx40_time - 2))**2)

    #
    time_1m = 2/c
    time_perp = tau0 - time_1m # A
    time_oblique_vacuum = time_1m * time_array[i] / tau0 # tau1'
    time_oblique_ground = time_array[i] - time_oblique_vacuum # B

    v_ground = 2 * L /np.sqrt( np.abs(time_oblique_ground**2 - time_perp**2)) - c * time_array[i] / time_oblique_ground
    epsilon_r2[i] = (c / v_ground)**2

mean = np.mean(epsilon_r2)
print(mean)

#plt.plot(index, epsilon_r1, label='no vacuum rivised')
plt.plot(index, epsilon_r2, label='relative permittivity')
plt.xlabel('distance from echo peak [m]', size=18)
plt.ylabel('relative permittivity', size=18)
plt.title(str(mean), size=20)
#plt.yscale('log')
plt.legend(fontsize = 18)


plt.grid()
plt.savefig('kanda/domain_20x10/4region_smooth/B-scan/permittivity')

plt.show()
