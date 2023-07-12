import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import size

# 真空を伝播する時間
c = 299792458 # [m/s]
t_vacuum0 = 2/c # [s]

# rx=40を基準にする
rx40_time = 0.5300e-7 - t_vacuum0

# rx=40以降のtime
time_array = [0.5307e-7, 0.5326e-7, 0.5354e-7, 0.5397e-7, 0.5446e-7, # 45まで
                0.5507e-7, 0.5576e-7, 0.5656e-7, 0.5741e-7, 0.5833e-7, # 50まで
                0.5930e-7, 0.6033e-7, 0.6142e-7, 0.6253e-7, 0.6366e-7, # 55まで
                0.6484e-7, 0.6602e-7, 0.6720e-7, 0.6842e-7, 0.6963e-7, # 60まで
                0.7088e-7, 0.7210e-7, 0.7338e-7, 0.7463e-7, 0.7590e-7, # 65まで
]

epsilon_r = np.zeros(len(time_array))
index = np.zeros(len(time_array))

for i in range(len(time_array)):

    l = (i + 1) * 0.2 # [m]
    index[i] = l

    t_vacuum = 2 / c # [s]
    epsilon_r[i] = ((time_array[i] - t_vacuum) **2 - rx40_time**2) * (c /2 /l)**2 
    

print(index)
print(epsilon_r)

plt.plot(index, epsilon_r)
plt.xlabel('distance from echo peak [m]', size=18)
plt.ylabel('relative permittivity', size=18)

plt.grid()
plt.show()
plt.savefig('kanda/domain_20x10/4region/B-scan/identify_permittivity.png')
