import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft
from tqdm import tqdm

from tools.outputfiles_merge import get_output_data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Processing migration', 
                                 usage='cd gprMax; python -m tools.pulse_compression out_file source_file')
parser.add_argument('out_file', help='.out file name')
parser.add_argument('source_file', help='source file name')
parser.add_argument('-select_rx', help='option: elect rx number', default=False, action='store_true')
args = parser.parse_args()


# Open output file and read number of outputs (receivers)
outfile_name = args.out_file
output_data = h5py.File(outfile_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()
input_dir_path = os.path.dirname(outfile_name)


# make output directory
output_dir_path = os.path.join(input_dir_path, 'pulse_compression_map')
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)



# =====correlation=====
rx_start = 1
rx_end = nrx + 1
if args.select_rx:
    rx_start = 25
    rx_end = rx_start + 1

for rx in range(rx_start, rx_end):
    # read .out file
    output_data, dt = get_output_data(outfile_name, rx, 'Ez')
    # read source file
    source_data, dt = get_output_data(args.source_file, 1, 'Ez')
    
    axis0_index_num = output_data.shape[0]
    axis1_index_num = output_data.shape[1]


    # mathced filter
    def matched_filter():
        correlation = np.zeros_like(output_data)
        #for i in tqdm(range(axis1_index_num), desc = 'filtering, rx' + str(rx)):
        for i in range(axis1_index_num):
            output_fft = fft.fft(output_data[:, i])
            source_fft = fft.fft(source_data[:, 1])

            # correlation
            corr = fft.ifft(output_fft * source_fft)
            correlation[:, i] = corr
        return correlation
    corr_data = matched_filter()


    def calc_power_map():
        correlation_power = np.zeros_like(corr_data)
        max_value = np.amax(np.abs(corr_data))

        for j in tqdm(range(axis1_index_num), desc = 'calc_power, rx' + str(rx)): # 列
            for k in range(axis0_index_num): # 行
                if np.abs(corr_data[k, j]) == 0:
                    correlation_power[k, j] = 10 * np.log10(1e-10 / max_value)
                else:
                    correlation_power[k, j] = 10 * np.log10(np.abs(corr_data[k, j]) / max_value)

        return correlation_power
    corr_data_power = calc_power_map()
    

    fig = plt.figure(figsize=(15, 12), facecolor='w', edgecolor='w')
    plt.imshow(corr_data_power,
               extent= [0, axis1_index_num, axis0_index_num*dt, 0], aspect='auto', cmap='rainbow',
               vmin=-50, vmax=0)
    plt.colorbar()

    plt.title('Correlation rx' + str(rx), size=20)
    plt.xlabel('Trace Number', size=15)
    plt.ylabel('Time', size=15)

    plt.savefig(output_dir_path + '/corr_map_rx' + str(rx) + '.png', dpi=150, format='png')
    if nrx == 1 or args.select_rx:
        plt.show()

