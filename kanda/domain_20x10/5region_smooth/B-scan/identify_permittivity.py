import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

# 真空を伝播する時間
c = 299792458 # [m/s]
t_vacuum0 = 2/c # [s]


# rx=41~
time_array = np.array([
                0.5137e-7, 0.5158e-7, 0.5189e-7, 0.5239e-7, 0.5283e-7, # 45まで
                0.5342e-7, 0.5413e-7, 0.5491e-7, 0.5574e-7, 0.5665e-7, # 50まで
                0.5762e-7, 0.5866e-7, 0.5972e-7, 0.6083e-7, 0.6196e-7, # 55まで
                0.6316e-7, 0.6432e-7, 0.6555e-7, 0.6677e-7, 0.6800e-7, # 60まで
])
time_array = time_array - 0.1e-8

# rx= 40を基準にする
tau0 = 0.5125e-7 - 0.1e-8 # [s]


epsilon_r1 = np.zeros(len(time_array))
epsilon_r2 = np.zeros(len(time_array))
epsilon_r3 = np.zeros(len(time_array))
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
    theta = np.arcsin(1/(c * time_oblique_vacuum / 2)) # Θの導出
    delta_l = c * time_oblique_vacuum / 2 * np.cos(theta) # Δl
    l = L - delta_l # l

    v_ground = 2 * L /np.sqrt(time_oblique_ground**2 - time_perp**2) - c * time_array[i] / time_oblique_ground
    epsilon_r2[i] = (c / v_ground)**2

    epsilon_r3[i] = (c / 2 / l)**2 * (time_oblique_ground**2 - time_perp**2)

mean1 = np.mean(epsilon_r2)
mean2 = np.mean(epsilon_r3)
#print(mean1)
print(mean2)

#plt.plot(index, epsilon_r1, label='no vacuum rivised')
#plt.plot(index, epsilon_r2, label='relative permittivity')
plt.plot(index, epsilon_r3)
plt.xlabel('distance from echo peak [m]', size=18)
plt.ylabel('relative permittivity', size=18)
# タイトルに平均値を小数点第３位まで表示
plt.title('mean = {:.3f}'.format(mean2), size=20)
#plt.yscale('log')
plt.legend(fontsize = 18)


plt.grid()
plt.savefig('kanda/domain_20x10/5region_smooth/B-scan/permittivity.png')

plt.show()
