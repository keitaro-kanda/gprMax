import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import json


from .outputfiles_merge import get_output_data
import mpl_toolkits.axes_grid1 as axgrid1


parser = argparse.ArgumentParser(description='Processing estimate Vrms in De Pue et al. method',
                                usage='cd gprMax; python -m tools.plot_Bscan_CMP jsonfile')
parser.add_argument('jsonfile', help='.out file path')
args = parser.parse_args()



#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)



#* Open output file and read number of outputs (receivers)
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



#* reshepe data into CMP style
data_list_CMP = []
for i in range(int(nrx/2)):
    rx_num = i
    tx_num = nrx - i - 1

    data_list_CMP.append(data_list[rx_num][:, tx_num])

data_lits_CMP = np.fliplr(np.array(data_list_CMP).T)
data_list_CMP = data_lits_CMP / np.max(data_lits_CMP) # normalize



#* save data
outputdir = os.path.dirname(args.jsonfile)
if not os.path.exists(outputdir):
    os.mkdir(outputdir)
np.savetxt(outputdir + '/Bscan_CMP.txt', data_list_CMP, delimiter=',')

#* plot B-scan
fig = plt.figure(figsize=(10, 10), tight_layout=True)
ax = fig.add_subplot(111)

plt.imshow(data_lits_CMP, aspect='auto', cmap='seismic', interpolation='nearest',
            extent=[0, data_lits_CMP.shape[1]*params['src_step']*2, data_lits_CMP.shape[0] * dt, 0],
            vmin=-1, vmax=1)
ax.set_title('CMP B-scan', fontsize=20)
ax.set_xlabel('Offset [m]', fontsize=20)
ax.set_ylabel('Time [ns]', fontsize=20)
ax.tick_params(labelsize=18)
ax.grid(which='both', axis='both', linestyle='--', linewidth=0.5)
xticks = np.arange(params['src_step']*2, (data_lits_CMP.shape[1]+1)*params['src_step']*2, params['src_step']*2)
ax.set_xticks(np.arange(0, data_lits_CMP.shape[1]*params['src_step']*2, params['src_step']*2))
ax.set_xticklabels(xticks)

delvider = axgrid1.make_axes_locatable(ax)
cax = delvider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(cax=cax).set_label('Normalised amplitude', fontsize=20)

cax.tick_params(labelsize=18)

plt.savefig(outputdir + '/Bscan_CMP.png')
plt.show()