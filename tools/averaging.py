from calendar import c

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tools.outputfiles_merge import get_output_data

file_name = 'kanda/domain_10x10/rock/B-scan/0.2step_n40/10x10_rock_merged.out'

# .outファイルの読み込み
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()



