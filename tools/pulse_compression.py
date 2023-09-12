import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft
from requests import get

from tools.outputfiles_merge import get_output_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Processing migration', 
                                 usage='cd gprMax; python -m tools.pulse_compression out_file source_file')
parser.add_argument('out_file', help='.out file name')
parser.add_argument('source_file', help='source file name')
parser.add_argument('-select_rx', help='option: elect rx number', default=False, action='store_true')
args = parser.parse_args()


# Open output file and read number of outputs (receivers)
file_name = args.out_file
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()
input_dir_path = os.path.dirname(file_name)


# make output directory
output_dir_path = os.path.join(input_dir_path, 'pulse_compression')
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)



# =====correlation=====
rx_start = 1
rx_end = nrx + 1
if args.select_rx:
    rx_start = 25
    rx_end = rx_start + 1

for rx in range(1, nrx+1):
    # read .out file
    output_data = get_output_data(file_name, rx, 'Ez')
    print(type(output_data))
    # read source file
    source_data = get_output_data(args.source_file, 1, 'Ez')
    print(type(source_data.type))
    
    x_index_num = output_data.shape[1]
    y_index_num = output_data.shape[0]


    # mathced filter
    corr_data = np.zeros_like(output_data)
    for i in range (output_data.shape[1]):
        output_fft = fft.fft(output_data[:, i])
        source_fft = fft.fft(source_data)

        # correlation
        corr = fft.ifft(output_fft * source_fft)
        corr_data[:, i] = 10 * np.log10(np.abs(corr)/np.amax(np.abs(corr)))

    print(corr_data.shape)
    
    fig = plt.figure(figsize=(15, 12), facecolor='w', edgecolor='w')
    plt.imshow(corr_data, aspect='auto', cmap='seismic')

