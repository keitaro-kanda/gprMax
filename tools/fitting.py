import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import argparse
from tools.outputfiles_merge import get_output_data
import h5py

#* Parse command line arguments
parser = argparse.ArgumentParser(description='Fitting hyperbola function',
                                 usage='cd gprMax; python -m tools.fitting jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* Open output file and read number of outputs (receivers)
#? h5pyを使ってデータを開ける意味はあまりないかも？nrx取得できるだけなのかな．
data_path = params['out_file']
data = h5py.File(data_path, 'r')
nrx = data.attrs['nrx']
data.close()
data_dir_path = os.path.dirname(data_path)
#* load data
data_list = []
for i in range(1, nrx+1):
    data, dt = get_output_data(data_path, i, 'Ez')
    data_list.append(data)


#* make output directory
outputdir = os.path.join(data_dir_path, 'Bscan_fitting')
if not os.path.exists(outputdir):
    os.mkdir(outputdir)


#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum


#* calculate hyperbola function
def calc_hyperbola(tau_ver, rxnumber, txnumber, Vrms):
    offset = np.abs((rxnumber - txnumber)) * params['src_step'] # [m]

    delay_time = np.sqrt(
        tau_ver **2 + (offset / (c * Vrms)) **2
    )

    return delay_time

t0 = params['t0_theory'] #[s]
Vrms = params['V_RMS_theory'] # [/c]

src_positions = np.arange(0, nrx+1, 1)

#* plotting fuction
def mpl_plot(outputdata, dt, rxnumber, rxcomponent):
    outputdata_norm = outputdata / np.amax(np.abs(outputdata)) * 100 # normalize

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # plot hyperbola
    for layers in tqdm(range(len(t0)), desc = 'rx' + str(rxnumber+1) + ' fitting'):
        hyperbola = calc_hyperbola(t0[layers], rxnumber, src_positions, Vrms[layers],)

        ax.plot(src_positions, hyperbola, linestyle='--',
                label = 't0 = ' + str(t0[layers]) + 's, Vrms = ' + str(Vrms[layers]) + 'c')
        ax.invert_yaxis()
        ax.set_ylim(outputdata_norm.shape[0] * dt, 0)


    #plot B-scan
    outputdata_norm = outputdata / np.amax(np.abs(outputdata)) * 100 # normalize
    plt.imshow(outputdata_norm,
             extent=[0, outputdata_norm.shape[1], outputdata_norm.shape[0] * dt, 0],
            interpolation='nearest', aspect='auto', cmap='seismic', vmin=-0.1, vmax=0.1)
    plt.title('rx' + str(rxnumber+1) + ')')
    plt.xlabel('trace number')
    plt.ylabel('Time [s]')
    plt.legend()

    # Grid properties
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    if 'E' in rxcomponent:
        cb.set_label('Field strength percentage [%]')
    elif 'H' in rxcomponent:
        cb.set_label('Field strength [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')


    # Save a PDF/PNG of the figure
    fig.savefig(outputdir + '/fitting_rx' + str(rxnumber+1) + '.png', dpi=150, format='png',
                 bbox_inches='tight', pad_inches=0.1)

    return plt

for rx in tqdm(range(nrx)):
    mpl_plot(data_list[rx], dt, rx, 'Ez') # rx is inputted by 0 ~ nrx
    plt.close()



