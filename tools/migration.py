from calendar import c

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

# 読み込みファイル名
file_name = 'kanda/domain_10x10/test/B-scan/test_B_merged.out'

# .outファイルの読み込み
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()

for rx in range(1, nrx + 1):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')

print(outputdata.shape)