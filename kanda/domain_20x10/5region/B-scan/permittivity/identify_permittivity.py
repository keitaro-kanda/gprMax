import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

# 真空を伝播する時間
c = 299792458 # [m/s]
t_vacuum0 = 2/c # [s]


time_array_before = np.array([0.5552e-7, 0.5472e-7, 0.5399e-7, 0.5335e-7, 0.5276e-7, # 34まで
                0.5231e-7, 0.5194e-7, 0.5168e-7, 0.5154e-7])
# rx=39以降のtime
time_array = np.array([
                0.5154e-7, 0.5165e-7, 0.5191e-7, 0.5229e-7, 0.5276e-7, # 44まで
                0.5335e-7, 0.5401e-7, 0.5479e-7, 0.5564e-7, 0.5656e-7, # 49まで
                0.5753e-7, 0.5859e-7, 0.5967e-7, 0.6081e-7, 0.6199e-7, # 54まで
                0.6319e-7, 0.6441e-7, 0.6566e-7, 0.6692e-7, 0.6817e-7, # 59まで
])

# rx=39を基準にする
tau0 = 0.5147e-7


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

    v_ground = 2 * L /np.sqrt(time_oblique_ground**2 - time_perp**2) - c * time_array[i] / time_oblique_ground
    epsilon_r2[i] = (c / v_ground)**2


plt.plot(index, epsilon_r1, label='no vacuum rivised')
plt.plot(index, epsilon_r2, label='vacuum rivised')
plt.xlabel('distance from echo peak [m]', size=18)
plt.ylabel('relative permittivity', size=18)
#plt.yscale('log')
plt.legend(fontsize = 18)


plt.grid()
plt.savefig('kanda/domain_20x10/5region/B-scan/permittivity')

plt.show()
